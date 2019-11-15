import os
import numpy as np
from datetime import datetime
from keras.models import *
from mdl_UNet import *
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from mdl_Plot import *
import matplotlib.pyplot as plt

# ***** モデルの学習と正解率と損失の変化をプロット *****
def Train(models_dir, use_model, time_dir, class_weight, trained_model, class_names,
         original_train, segmented_train, original_val, segmented_val, BATCH_SIZE, EPOCH,INIT_SIZE):

    if class_weight == True:
        # 全画素数を求める
        sum_cnt = np.zeros(segmented_train.shape[3])
        for num in range(segmented_train.shape[0]):
            for i in range(segmented_train.shape[1]):
                for j in range(segmented_train.shape[2]):
                    sum_cnt += segmented_train[num][i][j]
    
        # 写真中に含まれるカテゴリの割合を算出する
        frequency = []
        for i in range(len(class_names)):
            frequency.append(sum_cnt[i]/np.sum(sum_cnt))
        plt.bar(range(0,len(class_names)),frequency)
    
        # クラス重みづけを行う
        classWeights = np.mean(frequency)/frequency
        print(classWeights)

    # モデルの読み込み
    if trained_model == True:
        print("学習済みモデルの構造と重みを読み込みます。")
        model = model_from_json(open(os.path.join(models_dir, use_model,'191002_1852.json'), 'r').read())
        model.summary()
        model.load_weights(os.path.join(models_dir, use_model,'191002_1836.h5'))

    # 学習済みモデルを使用しない場合
    else:
        # モデルの学習
        if use_model == 'UNet':
            print(use_model + 'が選択されました。')
            dir_unet = os.path.join(models_dir, use_model)
            os.makedirs(dir_unet, exist_ok=True)
            model_path = os.path.join(dir_unet, datetime.now().strftime('%y%m%d_%H%M')+'.h5')
            print('モデルの重みを保存しているパス: ', model_path)
            print("")
        
            #unet = UNet(input_height = 224, input_width  = 224, nClasses = len(class_names))
            #model = unet.create_model()
            network = UNet1(INIT_SIZE)
            model = network.get_model()
            model.summary()

        adam = optimizers.Adam()
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        # 評価に用いるモデルの保存
        checkpointer = ModelCheckpoint(model_path, monitor = 'val_loss', verbose = 0, save_best_only = True)
        # 各エポックの結果をcsvファイルに保存するコールバック関数
        csv_logger = CSVLogger(os.path.join(models_dir,use_model,datetime.now().strftime('%y%m%d_%H%M') +'.log'))

        if class_weight:
            print("クラス重み付けを行います")
            class_weights = classWeights
        else:
            print("クラス重み付けを行いません")
            class_weights = None

        print("Training start!")
        fit = model.fit(original_train, segmented_train,
                          validation_data=(original_val,segmented_val),
                          batch_size=BATCH_SIZE,
                          epochs=EPOCH,
                          verbose=1,
                          shuffle=True,
                          callbacks=[checkpointer,csv_logger],
                          class_weight=class_weights
                          )

        print("Training finish!")

        # モデルの保存
        print("モデルの保存")
        open(os.path.join(models_dir,use_model,datetime.now().strftime('%y%m%d_%H%M') +'.json'),"w").write(model.to_json())
        model.save_weights(model_path)

        # ***** 1epochごとに正解率と損失の変化をプロットする *****
        print("正解率と損失の変化をプロット")
        plot_history_acc(fit)
        plot_history_loss(fit)
        plt.savefig(os.path.join(time_dir, 'TrainPlot.png'))

    return model
    
