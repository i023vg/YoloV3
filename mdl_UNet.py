import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *

# ***** Semantic Segmentationを実施する際に使用するモデル *****
class UNet():
    def __init__(self,input_height,input_width,nClasses):
        self.input_height=input_height
        self.input_width=input_width
        self.nClasses=nClasses


    def create_model(self):
        n_filters=64    # originalは64
        kernel_size=3
        stride_size = (2,2)
        PoolSize = (2,2)
        batchnorm=True

        # contracting path
        img_input = Input(shape=(self.input_height,self.input_width, 3)) ## Assume 224,224,3
        enc1 = Conv2D(n_filters,kernel_size,padding='same')(img_input)
        if batchnorm:
            enc1 = BatchNormalization()(enc1)
        enc1 = Activation("relu")(enc1)
        enc1 = Conv2D(n_filters,kernel_size,padding='same')(enc1)
        if batchnorm:
            enc1 = BatchNormalization()(enc1)
        enc1 = Activation("relu")(enc1)
        pool1 = MaxPooling2D(pool_size=PoolSize)(enc1)
    
        enc2 = Conv2D(n_filters*2,kernel_size,padding='same')(pool1)
        if batchnorm:
            enc2 = BatchNormalization()(enc2)
        enc2 = Activation("relu")(enc2)
        enc2 = Conv2D(n_filters*2,kernel_size,padding='same')(enc2)
        if batchnorm:
            enc2 = BatchNormalization()(enc2)
        enc2 = Activation("relu")(enc2)
        pool2 = MaxPooling2D(pool_size=PoolSize)(enc2)

        enc3 = Conv2D(n_filters*4,kernel_size,padding='same')(pool2)
        if batchnorm:
            enc3 = BatchNormalization()(enc3)
        enc3 = Activation("relu")(enc3)
        enc3 = Conv2D(n_filters*4,kernel_size,padding='same')(enc3)
        if batchnorm:
            enc3 = BatchNormalization()(enc3)
        enc3 = Activation("relu")(enc3)
        pool3 = MaxPooling2D(pool_size=PoolSize)(enc3)
        
        enc4 = Conv2D(n_filters*8,kernel_size,padding='same')(pool3)
        if batchnorm:
            enc4 = BatchNormalization()(enc4)
        enc4 = Activation("relu")(enc4)
        enc4 = Conv2D(n_filters*8,kernel_size,padding='same')(enc4)
        if batchnorm:
            enc4 = BatchNormalization()(enc4)
        enc4 = Activation("relu")(enc4)
        pool4 = MaxPooling2D(pool_size=PoolSize)(enc4)
        
        enc5 = Conv2D(n_filters*16,kernel_size,padding='same')(pool4)
        if batchnorm:
            enc5 = BatchNormalization()(enc5)
        enc5 = Activation("relu")(enc5)
        enc5 = Conv2D(n_filters*16,kernel_size,padding='same')(enc5)
        if batchnorm:
            enc5 = BatchNormalization()(enc5)
        enc5 = Activation("relu")(enc5)

        # expansive path
        dec1 = Conv2DTranspose(n_filters*8,kernel_size,strides=stride_size,padding='same')(enc5)
        dec1 = concatenate([dec1,enc4],axis=3)
        dec1 = Conv2D(n_filters*8,kernel_size,padding='same')(dec1)
        if batchnorm:
            dec1 = BatchNormalization()(dec1)
        dec1 = Activation("relu")(dec1)
        dec1 = Conv2D(n_filters*8,kernel_size,padding='same')(dec1)
        if batchnorm:
            dec1 = BatchNormalization()(dec1)
        dec1 = Activation("relu")(dec1)

        dec2 = Conv2DTranspose(n_filters*4,kernel_size,strides=stride_size,padding='same')(dec1)
        dec2 = concatenate([dec2,enc3],axis=3)
        dec2 = Conv2D(n_filters*4,kernel_size,padding='same')(dec2)
        if batchnorm:
            dec2 = BatchNormalization()(dec2)
        dec2 = Activation("relu")(dec2)
        dec2 = Conv2D(n_filters*4,kernel_size,padding='same')(dec2)
        if batchnorm:
            dec2 = BatchNormalization()(dec2)
        dec2 = Activation("relu")(dec2)

        dec3 = Conv2DTranspose(n_filters*2,kernel_size,strides=stride_size,padding='same')(dec2)
        dec3 = concatenate([dec3,enc2],axis=3)
        dec3 = Conv2D(n_filters*2,kernel_size,padding='same')(dec3)
        if batchnorm:
            dec3 = BatchNormalization()(dec3)
        dec3 = Activation("relu")(dec3)
        dec3 = Conv2D(n_filters*2,kernel_size,padding='same')(dec3)
        if batchnorm:
            dec3 = BatchNormalization()(dec3)
        dec3 = Activation("relu")(dec3)

        dec4 = Conv2DTranspose(n_filters,kernel_size,strides=stride_size,padding='same')(dec3)
        dec4 = concatenate([dec4,enc1],axis=3)
        dec4 = Conv2D(n_filters,kernel_size,padding='same')(dec4)
        if batchnorm:
            dec4 = BatchNormalization()(dec4)
        dec4 = Activation("relu")(dec4)
        dec4 = Conv2D(n_filters,kernel_size,padding='same')(dec4)
        if batchnorm:
            dec4 = BatchNormalization()(dec4)
        dec4 = Activation("relu")(dec4)

        outputs = Conv2D(self.nClasses,1,activation='sigmoid')(dec4)
        model = Model(inputs=[img_input],output=[outputs])
        return model

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