import numpy as np
from PIL import Image
import glob,os,cv2

# 入力画像と教師画像の読み込み
class Loader():
    # 原画像の正規化
    def Preprocessing(process, img):
        if process == 'normalize':
            img = img / 255.0
        elif process == 'standardization':
            img -= np.mean(img)
            img /= np.std(img)
        return img

    def crop_to_square(image):
        size = min(image.size)
        left, upper = (image.width - size) // 2, (image.height - size) // 2
        right, bottom = (image.width + size) // 2, (image.height + size) // 2
        return image.crop((left,upper, right,bottom))
    
    def images_load(original_dir,segmented_dir, class_names,pixelID,init_size,
                    one_hot=True, clipping=True):
        # フォルダの画像パスを取得
        origin_filenames = os.listdir(original_dir)
        segmented_filenames = os.listdir(segmented_dir)
    
        images_original, images_segmented = [], []
    
        # 入力画像の読み込み
        print("Loading original images .... ", end="", flush=True)
        for i, origin_filename in enumerate(origin_filenames):
            origin_img = Image.open(os.path.join(original_dir, origin_filename))
            # Crop処理
            if clipping == True:
                origin_img =  Loader.crop_to_square(origin_img)
            # resize処理(224,224)
            # 画像中のジャギーを目立たなくする処理
            origin_img = origin_img.resize(init_size, Image.ANTIALIAS)
            # numpy配列に変換
            origin_img = np.asarray(origin_img, dtype=np.float32)
            # normalize処理
            origin_img = Loader.Preprocessing('normalize', origin_img)
        
            images_original.append(origin_img)
        print("Completed")    
    
        # 教師画像の読み込み
        print("Loading segmented images... ", end="", flush=True)
        for i, seg_filename in enumerate(segmented_filenames):
            seg_img = Image.open(os.path.join(segmented_dir, seg_filename))
            # Crop処理
            if clipping == True:
                seg_img = Loader.crop_to_square(seg_img)
            # resize処理
            seg_img = seg_img.resize(init_size)
            # numpy配列に変換
            seg_img = np.asarray(seg_img)
            images_segmented.append(seg_img)
        print("Completed")
    
        # images_originalとimages_segmentedの長さが同じか確かめる
        assert len(images_original) == len(images_segmented)
    
        # numpy配列に変換
        images_original = np.asarray(images_original, dtype=np.float32)
        images_segmented = np.asarray(images_segmented, dtype=np.uint8)
    
        if one_hot:
            print("Casting to one-hot encoding... ", end="", flush=True)
            ID = np.zeros((images_segmented.shape[0],images_segmented.shape[1],
                           images_segmented.shape[2]), dtype=np.uint8)
            for num in range(images_segmented.shape[0]):
                for i in range(images_segmented.shape[1]):
                    for j in range(images_segmented.shape[2]):
                        for c in range(len(pixelID)):
                            if all(images_segmented[num][i][j]==pixelID[c]):
                                ID[num][i][j]=c
            identity = np.identity(len(class_names), dtype=np.uint8)
            images_segmented = identity[ID]
        else:
            images_segmented = ID
        
        print("Done")  
        print("Input_images Shape :  ", end="")
        print(images_original.shape)
        print("Segmented_images Shape : ", end="")
        print(images_segmented.shape)
        print("")
        return images_original, images_segmented
