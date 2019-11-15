import os

def make_Directories(Data_dir, CutoutImg_dir, MaskImg_dir):

    print('\nStep1 必要となるフォルダを作成します。')

    # ***** 出力フォルダが作成されるフォルダ *****
    Output_dir = Data_dir + 'OutputImage/'

    try:
        os.makedirs(Output_dir)
    except FileExistsError:
        pass

    # ***** 物体検出した結果を保存するフォルダ *****
    DetectionImage_dir = Output_dir + 'DetectionImage/'
    try:
        os.makedirs(DetectionImage_dir)
    except FileExistsError:
        pass

    # ***** 切り出した画像を保存するフォルダ（変状ごとに保存） *****
    CutoutImage_dir  = Output_dir + 'CutoutImage/'
    CutoutImg_dir.append(CutoutImage_dir + 'Cra/')
    CutoutImg_dir.append(CutoutImage_dir + 'Efc/')
    CutoutImg_dir.append(CutoutImage_dir + 'Efd/')
    CutoutImg_dir.append(CutoutImage_dir + 'Efee/')
    CutoutImg_dir.append(CutoutImage_dir + 'Efer/')
    CutoutImg_dir.append(CutoutImage_dir + 'Rec/')
    CutoutImg_dir.append(CutoutImage_dir + 'Red/')
    CutoutImg_dir.append(CutoutImage_dir + 'Ree/')

    try:
        os.makedirs(CutoutImage_dir)
        os.makedirs(CutoutImg_dir[0])
        os.makedirs(CutoutImg_dir[1])
        os.makedirs(CutoutImg_dir[2])
        os.makedirs(CutoutImg_dir[3])
        os.makedirs(CutoutImg_dir[4])
        os.makedirs(CutoutImg_dir[5])
        os.makedirs(CutoutImg_dir[6])
        os.makedirs(CutoutImg_dir[7])
    except FileExistsError:
        pass

    # ***** マスク画像を保存するフォルダ（変状ごとに保存） *****
    MaskImage_dir  = Output_dir + 'MaskImage/'
    MaskImg_dir.append(MaskImage_dir + 'Cra/')
    MaskImg_dir.append(MaskImage_dir + 'Efc/')
    MaskImg_dir.append(MaskImage_dir + 'Efd/')
    MaskImg_dir.append(MaskImage_dir + 'Efee/')
    MaskImg_dir.append(MaskImage_dir + 'Efer/')
    MaskImg_dir.append(MaskImage_dir + 'Rec/')
    MaskImg_dir.append(MaskImage_dir + 'Red/')
    MaskImg_dir.append(MaskImage_dir + 'Ree/')
    try:
        os.makedirs(MaskImage_dir)
        os.makedirs(MaskImg_dir[0])
        os.makedirs(MaskImg_dir[1])
        os.makedirs(MaskImg_dir[2])
        os.makedirs(MaskImg_dir[3])
        os.makedirs(MaskImg_dir[4])
        os.makedirs(MaskImg_dir[5])
        os.makedirs(MaskImg_dir[6])
        os.makedirs(MaskImg_dir[7])
    except FileExistsError:
        pass

    # ***** 座標を保存するディレクトリ *****
    Coordinate_dir = Data_dir + 'Coordinate/'
    try:
        os.makedirs(Coordinate_dir)
    except FileExistsError:
        pass


    Cut_dir  = Output_dir + 'Cut/'
    try:
        os.makedirs(Cut_dir)
    except FileExistsError:
        pass

    Final_dir  = Output_dir + 'Final/'
    try:
        os.makedirs(Final_dir)
    except FileExistsError:
        pass

    # ***** マスク画像と切り取り画像を重ね合わせた画像を保存するフォルダ *****
    OverlayImage_dir  = Output_dir + 'OverlayImage/'
    try:
        os.makedirs(OverlayImage_dir)
    except FileExistsError:
        pass

    # ***** 最終的な画像を保存するフォルダ *****
    FinalImage_dir  = Output_dir + 'FinalImage/'
    try:
        os.makedirs(FinalImage_dir)
    except FileExistsError:
        pass

