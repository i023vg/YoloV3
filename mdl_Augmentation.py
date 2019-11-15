import os
import numpy as np
import random
from PIL import Image

#  ***** 訓練画像の水増しを行う *****
# shift_arg : 元画像に回転と平行移動を加える
# flip_arg : shift_argを実施後の画像に左右反転を行う
# 出力フォルダ : train_arg, trainseg_arg
def Augmentation(save_dir, shift_arg=True,flip_arg=True):
    # 読み込みフォルダ
    train_img = os.path.join(save_dir,"train_img")
    train_seg = os.path.join(save_dir, "train_seg")
    val_img = os.path.join(save_dir, "val_img")
    val_seg = os.path.join(save_dir, "val_seg")
    test_img = os.path.join(save_dir, "test_img")
    test_seg = os.path.join(save_dir, "test_seg")

    # 生成フォルダ
    train_arg = os.path.join(save_dir,"train_arg")
    os.makedirs(train_arg,exist_ok=True)
    trainseg_arg = os.path.join(save_dir, "trainseg_arg")
    os.makedirs(trainseg_arg,exist_ok=True)
    

    r1 = []
    for i in np.arange(-10,10,1):
        r1.append(i)
    random.choice(r1)

    # 元画像に回転、平行移動して水増しするかどうか指定する
    # 水増し画像を 'train_arg', 'trainseg_arg' フォルダに出力する
    # (元画像 + 回転2種類 + 平行移動2種類) = 5種類

    train_file = os.listdir(train_img)
    trainseg_file = os.listdir(train_seg)

    for path in train_file:
        path = path[:-4]
        input_img = Image.open(os.path.join(train_img, path + ".jpg"))
        input_seg = Image.open(os.path.join(train_seg, path + ".png"))
        input_img.save(os.path.join(train_arg, path+".jpg"))
        input_seg.save(os.path.join(trainseg_arg, path+".png"))

        if shift_arg == True:
            # 回転
            random.shuffle(r1)
            for i in range(2):
                r = r1[i]
                r_img = input_img.rotate(r)
                r_img.save(os.path.join(train_arg, "rotate_" + str(r)+ "_"+ path+ ".jpg"))
                r_seg = input_seg.rotate(r)
                r_seg.save(os.path.join(trainseg_arg, "rotate_" + str(r)+ "_"+ path+".png"))

            # 平行移動
            tr1,tr2 = [], []
            for i in np.arange(int(input_img.height*(-0.1)),int(input_img.height*0.1), 1):
                tr1.append(i)
            for i in np.arange(int(input_img.width*(-0.1)), int(input_img.width*0.1), 1):
                tr2.append(i)
            random.shuffle(tr1), random.shuffle(tr2)

            for i in range(2):
                t1,t2 = tr1[i], tr2[i]
                t_img = input_img.rotate(0, translate=(t1,t2))
                t_seg = input_seg.rotate(0, translate=(t1,t2))
                t_img.save(os.path.join(train_arg,"trans_" + str(t1)+ "+" + str(t2)+ "_" + path+".jpg"))
                t_seg.save(os.path.join(trainseg_arg,"trans_" + str(t1)+ "+" + str(t2)+ "_" + path+".png"))

    # 元画像に左右反転して水増しするかどうか指定する
    # 水増し画像を 'train_arg', 'trainseg_arg' フォルダに出力する
    # (元画像 + 回転2種類 + 平行移動2種類)*(左右反転2種類) = 10種類
    train_argfile = os.listdir(train_arg)
    trainseg_argfile = os.listdir(trainseg_arg)

    if flip_arg:
        for path in train_argfile:
            path = path[:-4]
            input_img = Image.open(os.path.join(train_arg, path+".jpg"))
            input_seg = Image.open(os.path.join(trainseg_arg, path+".png"))
            #input_img.save(os.path.join(train_arg, path+".jpg"))
            #input_seg.save(os.path.join(trainseg_arg, path+".png"))

            h_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
            h_seg = input_seg.transpose(Image.FLIP_LEFT_RIGHT)
            h_img.save(os.path.join(train_arg, "flip_"+ path +".jpg"))
            h_seg.save(os.path.join(trainseg_arg, "flip_"+ path +".png"))
    
