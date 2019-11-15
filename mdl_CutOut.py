# =============================================================================
# 画像の切り出し
# =============================================================================

# ***** モジュールのインポート *****

import os
import numpy as np
from PIL import Image

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# 画像を切り出す関数
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

def CutOut(image_paths,BoundingBoxes,CutoutImg_dir,Data_dir,index):

    # index=0：セグメンテーション用の切り出し
    # index=1：貼り付け用の切り出し

    # ***** 座標保存用配列 *****
    coordinate = []
    box = []

    i = -1
    for image_path in image_paths:
        i += 1
        FileName = os.path.basename(image_path)
        onlyFileName,ExtName = os.path.splitext(FileName)

        img = Image.open(image_path)
        coordinate = BoundingBoxes[i]


        for j,c in enumerate(coordinate):
           
            # c[0] : クラス
            # c[1] : x1
            # c[2] : y1
            # c[3] : x2
            # c[4] : y2

            # ***** セグメンテーション用の画像の場合 *****               
            if index == 0:
                # 少し大きめに切り取る時
                c1 = int(c[1] - (c[3]-c[1])*0.05)
                c2 = int(c[2] - (c[4]-c[2])*0.05)
                c3 = int(c[3] + (c[3]-c[1])*0.05)
                c4 = int(c[4] + (c[4]-c[2])*0.05)

                #切り抜きの画像サイズを取得
                New_Width = c3 - c1
                New_Height = c4 - c2

                #ひび割れ用のトリミングサイズ調整
                if c[0] == 0:
                    if New_Width > New_Height:
                        dis_width = New_Width - New_Height
                        c4 = int(c4 + (dis_width/2))
                        c2 = int(c2 - (dis_width/2))
                    elif New_Height > New_Width:
                        dis_height = New_Height - New_Width
                        c3 = int(c3 + (dis_height/2))
                        c1 = int(c1 - (dis_height/2))

            # ***** 貼り付け用の画像の場合 *****               
            elif index == 1:
                # ちょうどの大きさで切り取る時
                c1 = c[1]
                c2 = c[2]
                c3 = c[3]
                c4 = c[4]

            # オリジナルの入力画像のサイズ（幅、高さ）を取得（Pillowを使用）
            Width, Height = img.size
        
            # トリミングが画像の範囲を越えたときの処理
            if c1 < 0:
                c1 = 0
            if c2 < 0:
                c2 = 0
            if c3 > Width:
                c3 = Width
            if c4 > Height:
                c4 = Height
        
            # 画像のトリミング（切り取り）とトリミング座標の格納
            img_crop = img.crop((c1,c2,c3,c4))
            box.append([c1,c2,c3,c4])

            # ***** 切り取られた画像の保存 *****
        
            # セグメンテーション用の画像の場合               
            if index == 0:
                ClassNo = c[0]
                if (0 <= ClassNo) and (ClassNo <=7):
                    Save_FileName = CutoutImg_dir[ClassNo] + onlyFileName + '_crop_' + str(j) + '.jpg'
                    img_crop.save(Save_FileName,quality=95)
                else:
                    print("Not found label")
                    break
                #トリミング座標の保存
                crop_coordinate = np.asarray(box)
                Save_FileName = Data_dir + 'Coordinate/crop_coordinate'
                np.save(Save_FileName,crop_coordinate)
            # 貼り付け用の画像の場合               
            elif index == 1:
                Save_FileName = Data_dir + 'OutputImage/Cut/' + onlyFileName + '_crop_'+str(j)+'.jpg'
                img_crop.save(Save_FileName,quality=95)
                #トリミング座標の保存
                crop_coordinate = np.asarray(box)
                Save_FileName = Data_dir + 'Coordinate/finalcrop_coordinate'
                np.save(Save_FileName,crop_coordinate)

