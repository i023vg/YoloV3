import numpy as np
import os,cv2
from PIL import Image

def output_img(y_pred,pre_input,pre_output,pixelID):
    filenames_predict = os.listdir(pre_input)
    
    for num in range(y_pred.shape[0]):
        img = cv2.imread(os.path.join(pre_input,filenames_predict[num]))
        y = cv2.resize(y_pred[num],(img.shape[1],img.shape[0]))
        y = np.argmax(y, axis=2)
        pre_img = np.empty((img.shape[0],img.shape[1],3),dtype=np.uint8)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pre_img[i][j] = pixelID[y[i][j]]
        pre_img = Image.fromarray((np.uint8(pre_img)))
        
        # 画像書き込み
        pre_img.save(os.path.join(pre_output,filenames_predict[num][:-4] + '.png'))

