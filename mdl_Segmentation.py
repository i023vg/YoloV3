# =============================================================================
# U-netによるセグメンテーション
# =============================================================================

# ***** モジュールのインポート *****

from keras.models import * #Model -> * に変更(model_from_jsonメソッドを使用するため)
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
import cv2
import numpy as np
from PIL import Image


class UNet(object):
    def __init__(self,input_size):
        
        # エンコーダ作成
        inputs = Input((input_size[1],input_size[0],3))
        
        enc1 = Conv2D(64,3,padding='same',activation='relu')(inputs)
        enc1 = Conv2D(64,3,padding='same',activation='relu')(enc1)
        enc1 = BatchNormalization()(enc1)
        pool1 = MaxPooling2D(pool_size=(2,2))(enc1)
        
        enc2 = Conv2D(128,3,padding='same',activation='relu')(pool1)
        enc2 = Conv2D(128,3,padding='same',activation='relu')(enc2)
        enc2 = BatchNormalization()(enc2)
        pool2 = MaxPooling2D(pool_size=(2,2))(enc2)
        
        enc3 = Conv2D(256,3,padding='same',activation='relu')(pool2)
        enc3 = Conv2D(256,3,padding='same',activation='relu')(enc3)
        enc3 = BatchNormalization()(enc3)
        pool3 = MaxPooling2D(pool_size=(2,2))(enc3)
        
        enc4 = Conv2D(512,3,padding='same',activation='relu')(pool3)
        enc4 = Conv2D(512,3,padding='same',activation='relu')(enc4)
        enc4 = BatchNormalization()(enc4)
        pool4 = MaxPooling2D(pool_size=(2,2))(enc4)
        
        enc5 = Conv2D(1024,3,padding='same',activation='relu')(pool4)
        enc5 = Conv2D(1024,3,padding='same',activation='relu')(enc5)
        
        
        # デコーダ作成
        dec1 = Conv2DTranspose(512,2,strides=(2,2),padding='same',activation='relu')(enc5)
        dec1 = concatenate([dec1,enc4],axis=3)
        dec1 = Conv2D(512,3,padding='same',activation='relu')(dec1)
        dec1 = Conv2D(512,3,padding='same',activation='relu')(dec1)
        dec1 = BatchNormalization()(dec1)
        
        dec2 = Conv2DTranspose(256,2,strides=(2,2),padding='same',activation='relu')(dec1)
        dec2 = concatenate([dec2,enc3],axis=3)
        dec2 = Conv2D(256,3,padding='same',activation='relu')(dec2)
        dec2 = Conv2D(256,3,padding='same',activation='relu')(dec2)
        dec2 = BatchNormalization()(dec2)
        #dec2 = Dropout(0.25)(dec2)
        
        dec3 = Conv2DTranspose(128,2,strides=(2,2),padding='same',activation='relu')(dec2)
        dec3 = concatenate([dec3,enc2],axis=3)
        dec3 = Conv2D(128,3,padding='same',activation='relu')(dec3)
        dec3 = Conv2D(128,3,padding='same',activation='relu')(dec3)
        dec3 = BatchNormalization()(dec3)
        #dec3 = Dropout(0.25)(dec3)
        
        dec4 = Conv2DTranspose(64,2,strides=(2,2),padding='same',activation='relu')(dec3)
        dec4 = concatenate([dec4,enc1],axis=3)
        dec4 = Conv2D(64,3,padding='same',activation='relu')(dec4)
        dec4 = Conv2D(64,3,padding='same',activation='relu')(dec4)
        dec4 = BatchNormalization()(dec4)
        #dec4 = Dropout(0.5)(dec4)
        
        # Conv2Dの最初の引数を1にするとグレースケールの出力
        outputs = Conv2D(2,1,activation='softmax')(dec4)
        
        self.UNet = Model(inputs=inputs,outputs=outputs)
    
    # ---------------------------------------------------------------------------------------------
    def get_model(self):
        return self.UNet
 
# 自作したUNet
class UNet1(object):
    def __init__(self,input_size):
        NumFilters = 64
        PoolSize = (2,2)
        KernelSize = 3
        StrideSize = (2,2)

        # エンコーダ作成
        inputs = Input((input_size[1],input_size[0],3))
        
        enc1 = Conv2D(NumFilters,KernelSize,padding='same',activation='relu')(inputs)
        enc1 = Conv2D(NumFilters,KernelSize,padding='same',activation='relu')(enc1)
        enc1 = BatchNormalization()(enc1)
        pool1 = MaxPooling2D(pool_size=PoolSize)(enc1)
        
        enc2 = Conv2D(NumFilters*2,KernelSize,padding='same',activation='relu')(pool1)
        enc2 = Conv2D(NumFilters*2,KernelSize,padding='same',activation='relu')(enc2)
        enc2 = BatchNormalization()(enc2)
        pool2 = MaxPooling2D(pool_size=PoolSize)(enc2)
        
        enc3 = Conv2D(NumFilters*4,KernelSize,padding='same',activation='relu')(pool2)
        enc3 = Conv2D(NumFilters*4,KernelSize,padding='same',activation='relu')(enc3)
        enc3 = BatchNormalization()(enc3)
        pool3 = MaxPooling2D(pool_size=PoolSize)(enc3)
        
        enc4 = Conv2D(NumFilters*8,KernelSize,padding='same',activation='relu')(pool3)
        enc4 = Conv2D(NumFilters*8,KernelSize,padding='same',activation='relu')(enc4)
        enc4 = BatchNormalization()(enc4)
        pool4 = MaxPooling2D(pool_size=PoolSize)(enc4)
        
        enc5 = Conv2D(NumFilters*16,KernelSize,padding='same',activation='relu')(pool4)
        enc5 = Conv2D(NumFilters*16,KernelSize,padding='same',activation='relu')(enc5)
        
        
        # デコーダ作成
        dec1 = Conv2DTranspose(NumFilters*8,2,strides=StrideSize,padding='same',activation='relu')(enc5)
        dec1 = concatenate([dec1,enc4],axis=3)
        dec1 = Conv2D(NumFilters*8,KernelSize,padding='same',activation='relu')(dec1)
        dec1 = Conv2D(NumFilters*8,KernelSize,padding='same',activation='relu')(dec1)
        dec1 = BatchNormalization()(dec1)
        
        dec2 = Conv2DTranspose(NumFilters*4,2,strides=StrideSize,padding='same',activation='relu')(dec1)
        dec2 = concatenate([dec2,enc3],axis=3)
        dec2 = Conv2D(NumFilters*4,KernelSize,padding='same',activation='relu')(dec2)
        dec2 = Conv2D(NumFilters*4,KernelSize,padding='same',activation='relu')(dec2)
        dec2 = BatchNormalization()(dec2)
        
        dec3 = Conv2DTranspose(NumFilters*2,2,strides=StrideSize,padding='same',activation='relu')(dec2)
        dec3 = concatenate([dec3,enc2],axis=3)
        dec3 = Conv2D(NumFilters,KernelSize,padding='same',activation='relu')(dec3)
        dec3 = Conv2D(NumFilters,KernelSize,padding='same',activation='relu')(dec3)
        dec3 = BatchNormalization()(dec3)
        
        dec4 = Conv2DTranspose(NumFilters,2,strides=StrideSize,padding='same',activation='relu')(dec3)
        dec4 = concatenate([dec4,enc1],axis=3)
        dec4 = Conv2D(NumFilters,KernelSize,padding='same',activation='relu')(dec4)
        dec4 = Conv2D(NumFilters,KernelSize,padding='same',activation='relu')(dec4)
        dec4 = BatchNormalization()(dec4)
        
        outputs = Conv2D(2,1,activation='sigmoid')(dec4)
        
        self.UNet = Model(input=inputs,output=outputs)
    
    # ---------------------------------------------------------------------------------------------
    def get_model(self):
        return self.UNet

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# ラベルの色を設定する関数
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

def label_color(class_label):
    
    label_color=[]
    # 背景用
    label_color.append([0,0,0])
    if class_label ==1:
        label_color.append([0,0,255])
    elif class_label == 2:
        label_color.append([0,128,0])
    elif class_label == 3:
        label_color.append([0,128,128])
    elif class_label == 4:
        label_color.append([128,0,128])
    elif class_label == 5:
        label_color.append([128,128,0])
    elif class_label == 6:
        label_color.append([128,0,0])
    elif class_label == 7:
        label_color.append([64,128,0])
    elif class_label == 8:
        label_color.append([0,64,128])
    
    return label_color

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# 画像を塗りつぶす関数
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

def draw_masks(class_label,Y_pred,f_name,label,CutoutImage_dir,MaskImage_dir):
    for i in range(Y_pred.shape[0]):
        print(i,'\t',f_name[i])
        save_name = f_name[i]
        save_name = save_name[:-4]
        img = cv2.imread(CutoutImage_dir + label + '/' + f_name[i])
        y = cv2.resize(Y_pred[i],(img.shape[1],img.shape[0]))
        #y = np.amax(y,axis=2)
        #y = np.where(y>0.5,1,0)
        y = np.argmax(y,axis=2)
        pre_img = np.random.randint(9,size=(img.shape[0],img.shape[1],3))
        for cal in range(img.shape[0]):
            for raw in range(img.shape[1]):
                r=label_color(class_label+1)
                rgb = r[y[cal][raw]]
                pre_img[cal][raw] = rgb
                    
        cv2.imwrite(MaskImage_dir + label + '/' + save_name + '.png',pre_img)



# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# 画像を読み込む関数
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

def loader(f_dir,f_name,init_size):
    P=[]
    print(f_dir)
    print(f_name)
    for inp in f_name:
        print(inp)
        img = Image.open(f_dir + '/' + inp)
        img = img.resize(init_size,Image.ANTIALIAS)
        img = np.asarray(img)
        img = img / 255
        P.append(img)
    P = np.asarray(P,dtype=np.float32)

    return P


# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# 推測処理を行う関数
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

def predict(X_pred,init_size,weights_dir,BATCH_SIZE):

    #network = UNet(init_size)
    #model = network.get_model()
    #'model.json'のモデル構造はUNet1
    model = model_from_json(open('model.json', 'r').read())
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    model.load_weights(weights_dir)
    
    Y_pred = model.predict(X_pred,BATCH_SIZE)

    return Y_pred