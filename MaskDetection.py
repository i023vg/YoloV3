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
import mdl_ObjectDitectionYolo3 as Yolo3
import mdl_CutOut as CO
import mdl_Segmentation as S
import mdl_makeDir as MD
import mdl_ImagePaste as IP

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from keras.backend import tensorflow_backend
import keras.backend as K

import json
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
Data_dir = 'D:/Yolo_Unet/'

# ***** 変状画像が格納されたフォルダ *****
Image_dir = Data_dir + 'OriginalImage_Ex/'
Label_dir = Data_dir + 'OriginalLabel/'

Output_dir = Data_dir + 'OutputImage/'
DetectionImage_dir = Output_dir + 'DetectionImage/'
CutoutImage_dir  = Output_dir + 'CutoutImage/'
MaskImage_dir  = Output_dir + 'MaskImage/'
Coordinate_dir = Data_dir + 'Coordinate/'
OverlayImage_dir  = Output_dir + 'OverlayImage/'
FinalImage_dir  = Output_dir + 'FinalImage/'
Cut_dir = Output_dir + 'Cut/'
final_dir = Output_dir + 'Final/'



# ***** 必要となるフォルダの作成 *****
CutoutImg_dir = []
MaskImg_dir = []
# -------------------------------------------------------------------------------------------------
MD.make_Directories(Data_dir, CutoutImg_dir, MaskImg_dir)
# -------------------------------------------------------------------------------------------------

# ***** keras(YOLOv3)で使うモデルと重み *****
model_weight = 'weights_concrete_20191107.h5'

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
# average IOU for 9 anchors: 0.69  'weights_concrete_20191104.h5'
#anchors = [41,274, 54,55, 84,326, 85,124, 169,338, 170,159, 273,77, 329,191, 373,358] # 1JPG-1XML

# average IOU for 9 anchors: 0.67  'weights_concrete_20191107.h5'
anchors = [44,65, 50,273, 98,336, 104,99, 161,170, 202,326, 296,86, 330,188, 380,356]
# ***** ラベルの定義 *****
labels  = ['Crack', 'Efflo_c', 'Efflo_d', 'Efflo_ee', 'Efflo_er', 'Rebar_c', 'Rebar_d', 'Rebar_e']
# <---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---

# Kerasへの入力画像の大きさ（416×416ピクセルの正方形のカラー画像）
# YOLOv3では、画像のサイズは32の倍数      
net_h, net_w = 416, 416

# ***** 検出クラスの閾値の設定 *****
obj_thresh = 0.50

# ***** 重なり合って同じオブジェクトの閾値の設定 *****
nms_thresh = 0.45

# ***** モデルと重みの読み込み *****
from yolo import dummy_loss
print('\nStep2 YOLOv3モデルとその重みを読み込みます。')
infer_model = load_model(model_weight,custom_objects={'dummy_loss': dummy_loss})

# ***** 1つの画像または複数の画像の検出を行う *****
image_paths = []

if os.path.isdir(input_dir): 
    for inp_file in natsorted(os.listdir(input_dir)):
        image_paths += [input_dir + inp_file]
else:
    image_paths += [input_dir]

image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg','.JPG', '.png', 'JPEG'])]


# ***** 元画像のサイズを保存する配列 *****
image_width = np.array
image_height = np.array

# ***** 座標を保存する配列 *****
BoundingBoxes = []

# ***** 変状部分の検出開始 *****
print('\nStep3 変状部分の検出を行います。\n')

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
    # ---------------------------------------------------------------------------------------------
    image,coordinates = Yolo3.draw_boxes(count, image_path, image, boxes, labels, obj_thresh) 
    # ---------------------------------------------------------------------------------------------
    BoundingBoxes.append(coordinates)

    # バウンディングボックスが付けられた画像を保存
    cv2.imwrite(output_dir + image_path.split('/')[-1], np.uint8(image)) 

# ***** BoundingBoxes[画像の枚数][座標] *****
BoundingBoxes=np.asarray(BoundingBoxes)
# ***** 座標の保存 *****
print("\n座標を保存中  ⇒⇒⇒ ",'\t', end="")
SaveFileName = Coordinate_dir + 'coordinate'
np.save(SaveFileName,BoundingBoxes)
print("座標の保存完了")

# ***** セグメンテーション用の画像の切り出し *****
print('\nSteo4 セグメンテーション用に変状部分の切り出しを行います。')

# -------------------------------------------------------------------------------------------------
# index=0：セグメンテーション用の切り出し
CO.CutOut(image_paths,BoundingBoxes,CutoutImg_dir, Data_dir,0)
# -------------------------------------------------------------------------------------------------

# ***** 推測処理 *****

#label_filename = ['Cra','Efc','Efd','Efee','Rec','Red','Ree']
label_filename = os.listdir(CutoutImage_dir)

weight_filename = os.listdir('./weights/')

init_size = (224,224)
BATCH_SIZE = 16

for i,label in enumerate(label_filename):
    f_dir = CutoutImage_dir + label
    f_name = natsorted(os.listdir(f_dir))
    #もしあるクラスのファイルに切り取り画像が無ければ次のループに入る
    if not f_name:
        strings = label + 'の切り取り画像がありません。'
        print(strings)
        continue
    
    #画像を読み込む
    print(f_dir)
    print(f_name)
    X_pred = S.loader(f_dir,f_name,init_size)

    #重みのディレクトリ指定
    weights_dir = './weights/' + weight_filename[i]
    strings = '\nラベル' + label + 'のU-Netの重みを読み込んでいます。'
    print(strings)

    #推測処理
    Y_pred = S.predict(X_pred,init_size,weights_dir,BATCH_SIZE)
    strings = '\nラベル' + label + 'のセグメンテーションが終了しました。'
    print(strings)


    #予測結果の書き出し
    strings = '\nラベル' + label + 'のマスク画像を書き出します。\n'
    print(strings)
    S.draw_masks(i,Y_pred,f_name,label,CutoutImage_dir,MaskImage_dir)
    
# ***** マスク画像と切り取り画像をpaste *****
print('\nStep5 マスク画像と切り取り画像のオーバーレイ処理を行います。\n')

# -------------------------------------------------------------------------------------------------
img_blend = IP.ImagePaste(CutoutImage_dir,MaskImage_dir,OverlayImage_dir)
img_blend.Blend(0.5) #引数はデフォルトで0.5
# -------------------------------------------------------------------------------------------------
        
# ***** 元画像にマスク画像をpaste *****
print('\nStep6 元画像にマスク画像を貼り付けます。\n')

# -------------------------------------------------------------------------------------------------
Load_FileName = Data_dir +'Coordinate/crop_coordinate.npy'
first_paste = IP.ImagePaste(Image_dir,OverlayImage_dir,FinalImage_dir)
first_paste.Paste(Load_FileName)
# -------------------------------------------------------------------------------------------------



# ***** 貼り付け用の画像の切り出し *****
print('\nStep7 貼り付け用に変状部分の切り出しを行います。\n')

# -------------------------------------------------------------------------------------------------
image_paths = []

if os.path.isdir(FinalImage_dir): 
    for inp_file in natsorted(os.listdir(FinalImage_dir)):
        image_paths += [FinalImage_dir + inp_file]
else:
    image_paths += [FinalImage_dir]

image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg','.JPG', '.png', 'JPEG'])]

#print("Bounding_shape00",BoundingBoxes.shape)
empty = []
for i in range(len(BoundingBoxes)):
    if not BoundingBoxes[i].any():
        empty.append(i)

for i,emp in enumerate(empty):
    
    if i > 0:
        emp -= i
        BoundingBoxes = np.delete(BoundingBoxes,emp)
    else:
        BoundingBoxes = np.delete(BoundingBoxes,emp)


#print("Bounding_shape00",BoundingBoxes.shape)
# index=1：貼り付け用の切り出し
CO.CutOut(image_paths,BoundingBoxes,CutoutImg_dir,Data_dir,1)
# -------------------------------------------------------------------------------------------------


# ***** 元画像にマスク画像をpaste *****
print('\nStep8 Detection画像にマスク画像を貼り付けます。\n')

# -------------------------------------------------------------------------------------------------
Load_FileName = Data_dir +'Coordinate/finalcrop_coordinate.npy'
last_paste = IP.ImagePaste(DetectionImage_dir,Cut_dir,final_dir)
last_paste.Paste(Load_FileName)
# -------------------------------------------------------------------------------------------------

print('\n処理が終了しました。\n')
