import os

'''
pixelIDを求めるには

>>> from PIL import Image
>>> import glob,os,cv2
>>> Data_dir = 'D:/Yolo_Unet/Segmentation/Concrete/'
>>> target_folder = "Crack"
>>> save_dir = os.path.join(Data_dir,target_folder)
>>> train_seg = os.path.join(save_dir, "train_seg")
>>> img=Image.open(os.path.join(train_seg,"Cra110.png"))
>>> print(img.mode)
RGB
>>> print(img.getcolors(img.size[0]*img.size[1]))
[(19879, (255, 255, 255)), (1090329, (0, 0, 0))]
'''

# ***** 二値分類 *****
def Category(target_folder, save_dir):
    # ひび割れ
    if target_folder == "Crack":
        print(target_folder + "が選択されました。")
        dir_seg = os.path.join(save_dir, "label/")
        dir_img = os.path.join(save_dir,  "Input/")
        class_names = ["background","Crack"]
        pixelID = [[0,0,0], [128,0,0]]

    # 漏水・遊離石灰C
    elif target_folder == "EffloC":
        print(target_folder + "が選択されました。")
        dir_seg = os.path.join(save_dir, "label/")
        dir_img = os.path.join(save_dir,  "Input/")
        class_names = ["background","EffloC"]
        pixelID = [[0,0,0], [0,128,0]]

    # 漏水・遊離石灰D
    elif target_folder == "EffloD":
        print(target_folder + "が選択されました。")
        dir_seg = os.path.join(save_dir, "label/")
        dir_img = os.path.join(save_dir,  "Input/")
        class_names = ["background","EffloD"]
        pixelID = [[0,0,0], [128,128,0]]

    # 漏水・遊離石灰E_ツララ
    elif target_folder == "EffloEE":
        print(target_folder + "が選択されました。")
        dir_seg = os.path.join(save_dir, "label/")
        dir_img = os.path.join(save_dir,  "Input/")
        class_names = ["background","EffloEE"]
        pixelID = [[0,0,0], [128,0,128]]

    # 漏水・遊離石灰E_錆
    elif target_folder == "EffloER":
        print(target_folder + "が選択されました。")
        dir_seg = os.path.join(save_dir, "label/")
        dir_img = os.path.join(save_dir,  "Input/")
        class_names = ["background","EffloER"]
        pixelID = [[0,0,0], [0,128,128]]

    # 剥離鉄筋露出(Rebar)C
    elif target_folder == "RebarC":
        print(target_folder + "が選択されました。")
        dir_seg = os.path.join(save_dir, "label/")
        dir_img = os.path.join(save_dir,  "Input/")
        class_names = ["background","RebarC"]
        pixelID = [[0,0,0], [0,0,128]]

    # 剥離鉄筋露出D
    elif target_folder == "RebarD":
        print(target_folder + "が選択されました。")
        dir_seg = os.path.join(save_dir, "label/")
        dir_img = os.path.join(save_dir,  "Input/")
        class_names = ["background","RebarD"]
        pixelID = [[0,0,0], [0,128,64]]

    # 剥離鉄筋露出E
    elif target_folder == "RebarE":
        print(target_folder + "が選択されました。")
        dir_seg = os.path.join(save_dir, "label/")
        dir_img = os.path.join(save_dir,  "Input/")
        class_names = ["background","RebarE"]
        pixelID = [[0,0,0], [128,64,0]]

    return class_names,pixelID

# ***** 多値分類 *****
def all_Category(target_folder, save_dir):
    print(target_folder + "が選択されました。")
    dir_seg = os.path.join(save_dir, "Cra_label/")
    dir_img = os.path.join(save_dir,  "Cra_Input/")
    class_names = ["background", "Crack", "EffloC", "EffloD", "EffloEE",
                  "EffloER", "RebarC", "RebarD", "RebarE"]
    pixelID = [[128,0,0], [255,255,255], [0,128,0], [128,128,0], [128,0,128],
               [0,128,128], [0,0,128], [0,128,64],[128,64,0]]
    return class_names, pixelID
