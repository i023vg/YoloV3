import os
from datetime import datetime

def Makedirs(save_dir, argmentation):
    result_dir = os.path.join(save_dir,"result_dir")
    os.makedirs(result_dir, exist_ok=True)
    time_dir = os.path.join(result_dir,datetime.now().strftime("%y%m%d_%H%M"))
    os.makedirs(time_dir, exist_ok=True)
    train_pred = os.path.join(time_dir, "train_pred")
    os.makedirs(train_pred,exist_ok=True)
    test_pred = os.path.join(time_dir, "test_pred")
    os.makedirs(test_pred, exist_ok=True)

    train_folder = os.path.join(time_dir, 'train')
    os.makedirs(train_folder)
    test_folder = os.path.join(time_dir, 'test')
    os.makedirs(test_folder)

    models_dir = os.path.join(result_dir, "models")
    os.makedirs(models_dir,exist_ok=True)

    if argmentation == True:
        print('argmentationを行います')
        train_img = os.path.join(save_dir, "train_arg")
        train_seg = os.path.join(save_dir, "trainseg_arg")
    else:
        print('argmentationを行いません')
        train_img = os.path.join(save_dir, "train_img")
        train_seg = os.path.join(save_dir, "train_seg")

    val_img = os.path.join(save_dir, "val_img")
    val_seg = os.path.join(save_dir, "val_seg")
    test_img = os.path.join(save_dir, "test_img")
    test_seg = os.path.join(save_dir, "test_seg")

    return [train_img, train_seg, val_img, val_seg, test_img, test_seg,
           models_dir,time_dir,train_folder, test_folder, train_pred, test_pred]
