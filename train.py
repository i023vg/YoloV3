#! /usr/bin/env python

# 必要な分だけメモリを確保するようにする。
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.chdir('C:/Users/K.Nakamura/Desktop/Yolo_Unet/Segmentation/Concrete/Concrete/Concrete/')

import tensorflow as tf
from keras.backend import tensorflow_backend
import keras.backend as K
K.clear_session()                                                      # ResourceExhaustedError対策
tensorflow_backend.clear_session()                                     # ResourceExhaustedError対策

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))  # ResourceExhaustedErrorに対して、ｺﾒﾝﾄｱｳﾄにより解消される場合がある。
session = tf.Session(config=config)                                    # ResourceExhaustedErrorに対して、ｺﾒﾝﾄｱｳﾄにより解消される場合がある。
tensorflow_backend.set_session(session)                                # ResourceExhaustedErrorに対して、ｺﾒﾝﾄｱｳﾄにより解消される場合がある。

import argparse
import numpy as np
import json
from voc import parse_voc_annotation
from yolo import create_yolov3_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint, CustomTensorBoard
from utils.multi_gpu_model import multi_gpu_model
import tensorflow as tf
import keras
from keras.models import load_model

config_min_input_size = 288
config_max_input_size = 448
# average IOU for 9 anchors: 0.69
#config_anchors = [41,274, 54,55, 84,326, 85,124, 169,338, 170,159, 273,77, 329,191, 373,358]
#average IOU for 9 anchors: 0.67
config_anchors = [50,175, 51,52, 64,349, 112,102, 150,343, 164,177, 304,81, 324,194, 369,358]

config_labels = ['Crack', 'Efflo_c', 'Efflo_d', 'Efflo_ee', 'Efflo_er', 'Rebar_c', 'Rebar_d', 'Rebar_e']

config_train_image_folder = 'C:/Users/K.Nakamura/Desktop/Yolo_Unet/OriginalImage/'
config_train_annot_folder = 'C:/Users/K.Nakamura/Desktop/Yolo_Unet/XML150/'
config_train_cache_name =  'concrete_train.pkl'

config_train_times = 8
config_batch_size = 1
config_learning_rate = 1e-4
config_nb_epochs = 1
config_ignore_thresh = 0.5
config_gpus = '0'

config_grid_scales = [1,1,1]  # deafultは [1,1,1]
#config_grid_scales = [3,2,1]  # deafultは [1,1,1]
config_obj_scale   = 10       # deafultは 10
config_noobj_scale = 5        # deafultは 1
config_xywh_scale  = 5        # deafultは 5
config_class_scale = 5        # deafultは 5

# 生成されたアンカーより最も依存しているスケールを考える場合に、それらがすべて等しい場合には、
# grid_scalesの3つの数値をすべて同じにする。スケールを大きくすると、その特定のスケールの価値が高くなり、
# 損失が大きくなり、モデルはその重みを最適化しようとする。
# 第1のスケール：入力の1/32×1/32特徴マップ上の予測
# 第2のスケール：入力の2/32×2/32特徴マップ上の予測
# 第3のスケール：入力の4/32×4/32特徴マップ上の予測

# 多くの偽陽性が出ている場合（誤検出）は、                                      obj_scale  を大きくするとよい。
# 多くの偽陰性が出ている場合（検出漏れ）は、                                    noobj_scaleを大きくするとよい。
# prediction boxのサイズと場所が適切でない場合は、                              xywh_scale を大きくするとよい。
# prediction boxが適切でも、分類スコアが低い、あるいは分類が間違っている場合は、class_scaleを大きくするとよい。

config_tensorboard_dir = 'logs'
config_saved_weights_name = 'weights_concrete.h5' # weights_concrete-1F1X.h5
config_saved_name = 'concrete.h5'  # concrete-1F1X.h5
config_debug = True

config_valid_image_folder = ''
config_valid_annot_folder = ''
config_valid_cache_name = ''

config_valid_times = 1


def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels,
):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8*len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t'  + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image

def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(tensorboard_logs)
    
    early_stop = EarlyStopping(
        monitor     = 'loss', 
        min_delta   = 0.01, 
        patience    = 5, 
        mode        = 'min', 
        verbose     = 1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save   = model_to_save,
        filepath        = saved_weights_name,# + '{epoch:02d}.h5', 
        monitor         = 'loss', 
        verbose         = 1, 
        save_best_only  = True, 
        mode            = 'min', 
        period          = 1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'min',
        epsilon  = 0.01,
        cooldown = 0,
        min_lr   = 0
    )
    tensorboard = CustomTensorBoard(
        log_dir                = tensorboard_logs,
        write_graph            = True,
        write_images           = True,
    )    
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard]

def create_model(
    nb_class, 
    anchors, 
    max_box_per_image, 
    max_grid, batch_size, 
    warmup_batches, 
    ignore_thresh, 
    multi_gpu, 
    saved_weights_name, 
    lr,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale  
):
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            template_model, infer_model = create_yolov3_model(
                nb_class            = nb_class, 
                anchors             = anchors, 
                max_box_per_image   = max_box_per_image, 
                max_grid            = max_grid, 
                batch_size          = batch_size//multi_gpu, 
                warmup_batches      = warmup_batches,
                ignore_thresh       = ignore_thresh,
                grid_scales         = grid_scales,
                obj_scale           = obj_scale,
                noobj_scale         = noobj_scale,
                xywh_scale          = xywh_scale,
                class_scale         = class_scale
            )
    else:
        template_model, infer_model = create_yolov3_model(
            nb_class            = nb_class, 
            anchors             = anchors, 
            max_box_per_image   = max_box_per_image, 
            max_grid            = max_grid, 
            batch_size          = batch_size, 
            warmup_batches      = warmup_batches,
            ignore_thresh       = ignore_thresh,
            grid_scales         = grid_scales,
            obj_scale           = obj_scale,
            noobj_scale         = noobj_scale,
            xywh_scale          = xywh_scale,
            class_scale         = class_scale
        )  

    # load the pretrained weight if exists, otherwise load the backend weight only
    # 存在する場合はpretrained weightをロードし、そうでなければbackend weightのみをロードする
    if os.path.exists(saved_weights_name): 
        print("\nLoading pretrained weights.\n")
        template_model.load_weights(saved_weights_name)
    else:
        template_model.load_weights("backend.h5", by_name=True)       

    if multi_gpu > 1:
        train_model = multi_gpu_model(template_model, gpus=multi_gpu)
    else:
        train_model = template_model      
    
    #template_model.load_weights("backend.h5", by_name=True)       
    ##template_model.load_weights("yolo_v3-weight.h5", by_name=True)       
    #train_model = template_model      


    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)             

    return train_model, infer_model

def _main_():
    #config_path = args.conf

    #with open(config_path) as config_buffer:    
        #config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config_train_annot_folder,
        config_train_image_folder,
        config_train_cache_name,
        config_valid_annot_folder,
        config_valid_image_folder,
        config_valid_cache_name,
        config_labels
    )
    print('\nTraining on: \t' + str(labels) + '\n')

    ###############################
    #   Create the generators 
    ###############################    
    train_generator = BatchGenerator(
        instances           = train_ints, 
        anchors             = config_anchors,   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config_batch_size,
        min_net_size        = config_min_input_size,
        max_net_size        = config_max_input_size,   
        shuffle             = True, 
        jitter              = 0.3, 
        norm                = normalize
    )
    
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config_anchors,   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config_batch_size,
        min_net_size        = config_min_input_size,
        max_net_size        = config_max_input_size,   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    ###############################
    #   Create the model 
    ###############################
    
    config_warmup_epochs = 3

    #if os.path.exists(config_saved_weights_name):
    #    config_warmup_epochs = 0
    warmup_batches = config_warmup_epochs * (config_train_times*len(train_generator))   

    os.environ['CUDA_VISIBLE_DEVICES'] = config_gpus
    multi_gpu = len(config_gpus.split(','))

    train_model, infer_model = create_model(
        nb_class            = len(labels), 
        anchors             = config_anchors, 
        max_box_per_image   = max_box_per_image, 
        max_grid            = [config_max_input_size, config_max_input_size], 
        batch_size          = config_batch_size, 
        warmup_batches      = warmup_batches,
        ignore_thresh       = config_ignore_thresh,
        multi_gpu           = multi_gpu,
        saved_weights_name  = config_saved_weights_name,
        lr                  = config_learning_rate,
        grid_scales         = config_grid_scales,
        obj_scale           = config_obj_scale,
        noobj_scale         = config_noobj_scale,
        xywh_scale          = config_xywh_scale,
        class_scale         = config_class_scale,
    )

    ###############################
    #   Kick off the training
    ###############################
    callbacks = create_callbacks(config_saved_weights_name, config_tensorboard_dir, infer_model)

    train_model.fit_generator(
        generator        = train_generator, 
        steps_per_epoch  = len(train_generator) * config_train_times, 
        epochs           = config_nb_epochs + config_warmup_epochs, 
        verbose          = 2 if config_debug else 1,
        callbacks        = callbacks, 
        workers          = 4,
        max_queue_size   = 3
        #max_queue_size   = 8
    )

    # make a GPU version of infer_model for evaluation
    if multi_gpu > 1:
        infer_model = load_model(config_saved_weights_name)

    ###############################
    #   Run the evaluation
    ###############################   
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
    
    #train_model.save_weights(config_saved_weights_name)
    #train_model.save(config_saved_name)

#if __name__ == '__main__':
    #argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    #argparser.add_argument('-c', '--conf', help='path to configuration file')   

    #args = argparser.parse_args()
    #_main_(args)

_main_()
