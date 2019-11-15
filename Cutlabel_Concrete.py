# =============================================================================
#
#        コンクリート表面の変状セグメンテーション
#
#        ①まず初めに、コンクリートの変状が撮影された画像から、YOLOv3による物体検出を行い、
#          変状のラベルとその領域を特定する。
#          変状のラベルは、以下の８つ。
#          (1)ひび割れ
#          (2)漏水・遊離石灰ｃ
#          (3)漏水・遊離石灰ｄ
#          (4)漏水・遊離石灰ｅ（つらら）
#          (5)漏水・遊離石灰ｅ（錆汁）
#          (6)剥離・鉄筋露出ｃ
#          (7)剥離・鉄筋露出ｄ
#          (8)剥離・鉄筋露出ｅ
#        ②変状のラベルが付いた切出画像に対して、変状領域をピクセル単位でU-netを使って
#          セグメンテーションする。
#　      ③セグメンテーションされた画像をもとの画像に重ね合わせる。
#
# =============================================================================

# ***** モジュールのインポート *****

# 必要な分だけメモリを確保するようにする。
import mdl_makeDir as MD
import mdl_ObjectDetectionYolo3 as Yolo3
import mdl_CutOut as CO
#from mdl_UNet import *
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from keras.backend import tensorflow_backend
import keras.backend as K

#import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import warnings
from keras.optimizers import Adam

from PIL import Image
from natsort import natsorted

# ***** Warningエラーを表示しないようにする *****

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

K.clear_session()                                                      # ResourceExhaustedError対策
tensorflow_backend.clear_session()                                     # ResourceExhaustedError対策
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))  # ResourceExhaustedErrorに対して、ｺﾒﾝﾄｱｳﾄにより解消される場合がある。
session = tf.Session(config=config)                                    # ResourceExhaustedErrorに対して、ｺﾒﾝﾄｱｳﾄにより解消される場合がある。
tensorflow_backend.set_session(session)                                # ResourceExhaustedErrorに対して、ｺﾒﾝﾄｱｳﾄにより解消される場合がある。

# ***** データが格納されているフォルダ *****
# <---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---
Data_dir = 'C:/Users/K.Nakamura/Desktop/Yolo_Unet/Segmentation/Concrete/EffloEE/'
# <---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---

# ***** 変状画像が格納されたフォルダ *****
Image_dir = Data_dir + 'Input/'
Label_dir = Data_dir + 'label/'

Output_dir = Data_dir + 'OutputImage/'
DetectionImage_dir = Output_dir + 'DetectionImage/'
CutoutImage_dir  = Output_dir + 'CutoutImage/'
Cutoutseg_dir  = Output_dir + 'Cutoutseg/'

# ***** 必要となるフォルダの作成 *****
CutoutImg_dir = []
Cutoutseg_dir = []
# -------------------------------------------------------------------------------------------------
MD.make_Directories(Data_dir, CutoutImg_dir, Cutoutseg_dir)
# -------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------

# ***** keras(YOLOv3)で使うモデルと重み *****
#model_weight = 'weights_concrete_20191104.h5'
#model_weight = 'weights_concrete_20191107.h5'
model_weight = 'weights_concrete-1F1X.h5'
# ***** オリジナルの画像データが格納されたフォルダ *****
# /--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/
input_dir = Image_dir
# /--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/

# ***** 物体検出した結果を保存するフォルダ *****
# /--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/
output_dir = DetectionImage_dir
# /--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/

# <---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---
# ***** アンカーの定義 *****
# average IOU for 9 anchors: 0.69 'weights_concrete_20191104.h5'
#anchors = [41,274, 54,55, 84,326, 85,124, 169,338, 170,159, 273,77, 329,191, 373,358]

#average IOU for 9 anchors: 0.67 'weights_concrete_20191107.h5'
#anchors = [50,180, 53,53, 66,354, 113,103, 157,187, 157,351, 297,90, 327,199, 371,359] # 1JPG-1XML

anchors = [47,140, 55,339, 56,47, 104,148, 127,339, 188,85, 224,206, 361,121, 368,354]
# ***** ラベルの定義 *****
labels  = ['Crack', 'Efflo_c', 'Efflo_d', 'Efflo_ee', 'Efflo_er', 'Rebar_c', 'Rebar_d', 'Rebar_e']
# <---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---

# Kerasへの入力画像の大きさ（416×416ピクセルの正方形のカラー画像）
# YOLOv3では、画像のサイズは32の倍数      
net_h, net_w = 416, 416

# ***** 検出クラスの閾値の設定 *****
obj_thresh = 0.75

# ***** 重なり合って同じオブジェクトの閾値の設定 *****
nms_thresh = 0.45

# ***** モデルと重みの読み込み *****
from yolo import dummy_loss
print('\nモデルとその重みを読み込みます。\n')
infer_model = load_model(model_weight,custom_objects={'dummy_loss': dummy_loss})

# ***** 1つの画像または複数の画像の検出を行う *****
image_paths,seg_paths = [],[]

if os.path.isdir(input_dir): 
    for inp_file in natsorted(os.listdir(input_dir)):
        image_paths += [input_dir + inp_file]
else:
    image_paths += [input_dir]

if os.path.isdir(Label_dir): 
    for L_file in natsorted(os.listdir(Label_dir)):
        seg_paths += [Label_dir + L_file]
else:
    seg_paths += [Label_dir]

image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg','.JPG', '.png', 'JPEG'])]
seg_paths = [L_file for L_file in seg_paths if (L_file[-4:] in ['.jpg','.JPG', '.png', 'JPEG'])]

# ***** 元画像のサイズを保存する配列 *****
image_width = np.array
image_height = np.array

# ***** 座標を保存する配列 *****
BoundingBoxes = []

#座標を保存するディレクトリ
Coordinate_dir = './Coordinate/'
try:
    os.makedirs(Coordinate_dir)
except FileExistsError:
    pass

# ***** 変状部分の検出開始 *****
print('\n変状部分の検出を行います。\n')

# ***** 保存されている各画像に対して変状部分の検出を行う *****
count = -1
for image_path in image_paths:
    image = cv2.imread(image_path)
    count += 1
    # クラス確率が閾値より大きいバウンディングボックスのみを抽出
    # ---------------------------------------------------------------------------------------------
    boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, anchors, obj_thresh, nms_thresh)[0]
    # ---------------------------------------------------------------------------------------------
    
    # ラベルを使って画像上にバウンディングボックスを描く
    # ----------------------------------------------------------------
    image,coordinates = Yolo3.draw_boxes(count, image_path, image, boxes, labels, obj_thresh) 
    # ----------------------------------------------------------------
    BoundingBoxes.append(coordinates)

    # バウンディングボックスが付けられた画像を保存
    cv2.imwrite(output_dir + image_path.split('/')[-1], np.uint8(image)) 

# ***** BoundingBoxes[画像の枚数][座標] *****
BoundingBoxes=np.asarray(BoundingBoxes)

# ***** 座標の保存 *****
print("\n座標を保存中⇒⇒⇒ ",'\t', end="")
SaveFileName = Coordinate_dir + 'coordinate'
np.save(SaveFileName,BoundingBoxes)
print("座標の保存完了")

# ***** 画像の切り出し *****
print('\n変状部分の切り出しを行います。\n')


CO.CutOut(image_paths,seg_paths, BoundingBoxes,CutoutImg_dir,Cutoutseg_dir)






