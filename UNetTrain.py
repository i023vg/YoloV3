import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
from mdl_Train import Train
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob,os,cv2
from mdl_TrainTestSplit import TrainTestSplit
from mdl_Augmentation import Augmentation
from mdl_Makedirs import Makedirs
from mdl_Category import *
from mdl_UNet import UNet
from mdl_Loader import Loader
from mdl_evaluate import evaluate
from mdl_output import output_img

# ***** Warningエラーを表示しないようにする *****
warnings.filterwarnings("ignore", category = DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

config = tf.ConfigProto()
set_session(tf.Session(config=config))

# ***** 実行環境の確認 *****
print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__)); del keras
print("tensorflow version {}".format(tf.__version__))
print("")

# ***** ハイパーパラメータの設定 *****
argmentation = False      # 画像の水増しを行うかどうか
BATCH_SIZE = 4            # 一度に読み込む画像枚数
EPOCH = 100                # モデルを学習する反復回数
class_weight = False       # クラス重み付けをするかどうか
use_model = 'UNet'        # 使用するモデルを選択する
trained_model = False     # 学習済みモデルを使用するかどうか
INIT_SIZE = (224,224)     # UNetを学習させる画像サイズ

if trained_model == True:
    # 学習済みモデルを使用するため、重みづけを使用しない
    class_weight = False

# ***** データが格納されているフォルダ *****
Data_dir = 'D:/Yolo_Unet/Segmentation/Concrete/'

# target_folderに "Crack", "EffloC", "EffloD", "EffloEE",
# "EffloER", "RebarC", "RebarD", "RebarE"のいずれかを選択する
target_folder = "RebarE"
save_dir = os.path.join(Data_dir,target_folder)

class_names,pixelID = Category(target_folder, save_dir)      # 二値分類を使用する場合
#class_names,pixelID = all_Category(target_folder, save_dir) # 多値分類を使用する場合
TrainTestSplit(save_dir)  # 元画像と教師画像を訓練画像と検証画像とテスト画像に分割する
if argmentation:
    Augmentation(save_dir)     # 訓練画像の水増しを行う

[train_img, train_seg, val_img, val_seg, test_img, test_seg,
models_dir,time_dir,train_folder, test_folder, train_pred, test_pred] = Makedirs(save_dir, argmentation)

# ***** 入力画像と教師画像の読み込み *****
print("Load start...") 
original_train, segmented_train = Loader.images_load(train_img, train_seg, class_names, pixelID,
                                                    init_size=INIT_SIZE,clipping=False)
original_val, segmented_val = Loader.images_load(val_img, val_seg, class_names, pixelID,
                                                init_size=INIT_SIZE,clipping=False)
original_test, segmented_test = Loader.images_load(test_img, test_seg, class_names, pixelID,
                                                  init_size=INIT_SIZE,clipping=False)

print("original_train data : ", end="")
print(len(original_train))
print("segmented_train data : ", end="")
print(len(segmented_train))
print("original_validation data : ", end="")
print(len(original_val))
print("segmented_validation data : ", end="")
print(len(segmented_val))
print("original_test data : ", end="")
print(len(original_test))
print("segmented_test data : ", end="")
print(len(segmented_test))
print("")

# ***** モデルの学習と正解率と損失の変化をプロット *****
model = Train(models_dir,use_model,time_dir,class_weight,trained_model,class_names,
              original_train,segmented_train, original_val,segmented_val,BATCH_SIZE, EPOCH,INIT_SIZE)

# ***** 訓練画像を用いて評価した時 *****
print("訓練画像の精度を判定")
x_pred = model.predict(original_train)
x_predi = np.argmax(x_pred, axis=3)
x_testi = np.argmax(segmented_train, axis=3)
print("x_testi.shape, x_predi.shape: ", end="")
print(x_testi.shape,x_predi.shape)

evaluate.IoU(x_testi,x_predi,train_folder,class_names)
evaluate.PixelWise_acc(x_testi,x_predi,train_folder,class_names)
evaluate.Mean_acc(x_testi,x_predi,train_folder,class_names)
evaluate.FWIU(x_testi,x_predi,train_folder,class_names)
evaluate.Fmeasure(x_testi,x_predi,train_folder,class_names)

# ***** テスト画像を用いて評価した時 *****
print("テスト画像の精度を判定")
y_pred = model.predict(original_test)
y_predi = np.argmax(y_pred, axis=3)
y_testi = np.argmax(segmented_test, axis=3)
print("y_testi.shape, y_predi.shape: ", end="")
print(y_testi.shape,y_predi.shape)

evaluate.IoU(y_testi,y_predi,test_folder,class_names)
evaluate.PixelWise_acc(y_testi,y_predi,test_folder,class_names)
evaluate.Mean_acc(y_testi,y_predi,test_folder,class_names)
evaluate.FWIU(y_testi,y_predi,test_folder,class_names)
evaluate.Fmeasure(y_testi,y_predi,test_folder,class_names)

# ***** セマンティックセグメンテーションを実施した画像を出力 *****
print("セマンティックセグメンテーションを実施した画像を出力")
output_img(x_pred,train_img,train_pred,pixelID)
output_img(y_pred,test_img,test_pred,pixelID)
