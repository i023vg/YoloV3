# ***** モジュールのインポート *****

import cv2
import numpy as np
from natsort import natsorted
from utils.colors import get_color
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
#
# 検出された各オブジェクトの周囲に境界ボックスやラベルを描画
#
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

def draw_boxes(count, image_path, image, boxes, labels, obj_thresh, quiet=True):
    
    # ***** 座標保存用配列 *****
    coordinates=[]
    
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', ' 
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)

        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin],
                              [box.xmin-3,        box.ymin-height-26],
                              [box.xmin+width+100, box.ymin-height-26],
                              [box.xmin+width+100, box.ymin]], dtype='int32')

            # ***** 四角描画　cv2.rectangle(画像,（左上座標）,（右下座標）, 色, 線の太さ) 
            #cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=(255, 0, 0), thickness=7)
            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=7)

            # ***** ポリゴン領域の塗りつぶし　cv2.fillPoly(画像, pts：ポリゴンの配列, npts:各ポリゴンの頂点数の配列,
            #  ncontours:領域を区切る輪郭の個数, 色, 線の種類, 頂点座標の小数点以下の桁のビット数)
            cv2.fillPoly(img=image, pts=[region], color=(255, 255, 255))  # 白

            # ***** 文字記述　cv2.putText(画像, 文字,（左下座標）, フォント, 文字の大きさ, 色, 文字の太さ, 線の種類)
            cv2.putText(img=image,
                        text=label_str,
                        org=(box.xmin+13, box.ymin - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0e-3 * image.shape[0],  # 文字の大きさ
                        color=(0,0,0),                      # 文字の色：黒
                        thickness=2)                        # 文字の太さ
            
            # ***** 座標を取得 *****
            coordinate = [label,box.xmin,box.ymin,box.xmax,box.ymax]
            print(count,'\t', end="")
            print(image_path,'\t', end="")
            print(box.xmin, '\t', end="")
            print(box.ymin, '\t', end="")
            print(box.xmax, '\t', end="")
            print(box.ymax, '\t', end="")
            print(label_str)
            coordinates.append(coordinate)
    
    # ***** 1枚の画像の中に存在するバウンディングボックスの個数分の座標 *****
    coordinates = np.asarray(coordinates)

    return image,coordinates

