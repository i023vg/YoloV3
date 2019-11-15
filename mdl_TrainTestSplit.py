import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# ***** 元画像と教師画像を訓練画像と検証画像とテスト画像に分割する *****
#  (訓練画像 + 検証画像) : テスト画像の比 = 8 : 2
#  訓練画像 ; 検証画像 = 8 : 2
# 出力フォルダ : train_img, train_seg, val_img, val_seg, test_img, test_seg
def TrainTestSplit(save_dir):
    dir_img = os.path.join(save_dir,"Input")
    dir_seg = os.path.join(save_dir,"label")

    train_img = os.path.join(save_dir,"train_img")
    os.makedirs(train_img,exist_ok=True)
    train_seg = os.path.join(save_dir, "train_seg")
    os.makedirs(train_seg,exist_ok=True)
    val_img = os.path.join(save_dir, "val_img")
    os.makedirs(val_img,exist_ok=True)
    val_seg = os.path.join(save_dir, "val_seg")
    os.makedirs(val_seg,exist_ok=True)
    test_img = os.path.join(save_dir, "test_img")
    os.makedirs(test_img,exist_ok=True)
    test_seg = os.path.join(save_dir, "test_seg")
    os.makedirs(test_seg,exist_ok=True)

    [original_trainval, original_test, segmented_trainval,
        segmented_test] = train_test_split(os.listdir(dir_img), os.listdir(dir_seg), test_size=0.2, random_state=1)
    [original_train, original_val, segmented_train,
        segmented_val] = train_test_split(original_trainval, segmented_trainval, test_size=0.2, random_state=1)

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

    for i in range(len(original_train)):
        # 訓練画像振り分け
        timg = Image.open(os.path.join(dir_img, original_train[i]))
        timg.save(os.path.join(train_img, original_train[i]))
        tseg = Image.open(os.path.join(dir_seg,segmented_train[i]))
        #tseg.putpalette(tseg.getpalette())
        tseg.save(os.path.join(train_seg, segmented_train[i]))

    for i in range(len(original_val)):
        # 検証画像振り分け
        vimg = Image.open(os.path.join(dir_img,original_val[i]))
        vimg.save(os.path.join(val_img, original_val[i]))
        vseg = Image.open(os.path.join(dir_seg,segmented_val[i]))
        #vseg.putpalette(vseg.getpalette())
        vseg.save(os.path.join(val_seg, segmented_val[i]))

    for i in range(len(original_test)):
        # テスト画像振り分け
        img = Image.open(os.path.join(dir_img,original_test[i]))
        img.save(os.path.join(test_img, original_test[i]))
        iseg = Image.open(os.path.join(dir_seg,segmented_test[i]))
        #iseg.putpalette(iseg.getpalette())
        iseg.save(os.path.join(test_seg, segmented_test[i]))