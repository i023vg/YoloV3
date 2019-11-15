import numpy as np
import os

# ***** Semantic segmentationにおける評価尺度 *****
class evaluate:
    def IoU(Yi,y_predi,folder,class_names):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

        with open(os.path.join(folder,'Iou.txt'),'w') as f:
            IoUs = []
            Nclass = int(np.max(Yi)) + 1
            for c in range(Nclass):
                TP = np.sum( (Yi == c)&(y_predi==c) )
                FP = np.sum( (Yi != c)&(y_predi==c) )
                FN = np.sum( (Yi == c)&(y_predi != c))
                IoU = TP/float(TP + FP + FN)
                f.write("class {:12s}: #TP={:8.0f}, #FP={:8.0f}, #FN={:8.0f}, IoU={:4.3f}".format(class_names[c],TP,FP,FN,IoU))
                f.write("\n")
                IoUs.append(IoU)
            mIoU = np.mean(IoUs)
            f.write("_________________")
            f.write("\n")
            f.write("Mean IoU: {:4.3f}".format(mIoU))
            f.write("\n")
            print("Mean IoU: {:4.3f}".format(mIoU))

    def PixelWise_acc(Yi,y_predi,folder,class_names):
        with open(os.path.join(folder,'Pixcelacc.txt'),'w') as f:
            # 画像の全画素数(縦 * 横)
            N = Yi.shape[1] * Yi.shape[2]
            # 1枚ごとの精度を保存する箱
            accuracies = []

            # 1枚ずつ精度を計算
            for i,testlabel in enumerate(Yi):
                # 正解画素数を計算
                Ni = np.sum(y_predi[i] == testlabel)
                # 精度の計算
                accuracy = Ni / N
                f.write("image {:02.0f}: #正解画素数={:6.0f}, #全画素数={:6.0f}, accuracy={:4.3f}".format(i,Ni,N,accuracy))
                f.write("\n")
                accuracies.append(accuracy)
            PixelWise_acc = np.mean(accuracies)
            f.write("_________________")
            f.write("\n")
            f.write("Pixel-wise Accuracy: {:4.3f}".format(PixelWise_acc))
            f.write("\n")
            print("Pixel-wise Accuracy: {:4.3f}".format(PixelWise_acc))
            
    def Mean_acc(Yi,y_predi,folder,class_names):
        with open(os.path.join(folder,'Mean_acc.txt'), 'w') as f:
            accuracies = []
            Nclass = int(np.max(Yi)) + 1
            for i in range(Nclass):
                # クラスiの正解画素数
                Ti = np.sum(Yi==i)
                # 予測した画像のクラスiの画素数
                Ni = np.sum((Yi==i)&(y_predi==i))
                # クラスごとの正解率の計算
                accuracy = Ni / Ti
                f.write("class {:12s}: #Ni={:6.0f}, #Ti={:6.0f}, Mean_acc={:4.3f}".format(class_names[i],Ni,Ti,accuracy))
                f.write("\n")
                accuracies.append(accuracy)
            Mean_acc = np.mean(accuracies)
            f.write("_________________")
            f.write("\n")
            f.write("Mean Accuracy: {:4.3f}".format(Mean_acc))
            f.write("\n")
            print("Mean Accuracy: {:4.3f}".format(Mean_acc))

    def FWIU(Yi,y_predi,folder,class_names):
        with open(os.path.join(folder,'FWIU.txt'), 'w') as f:
            accuracies=0
            N = Yi.shape[0]*Yi.shape[1] * Yi.shape[2]
            Nclass = int(np.max(Yi)+1)
            for i in range(Nclass):
                accuracy=0
                Ti = np.sum(Yi==i)
                Ni = np.sum((Yi==i)&(y_predi==i))
                accuracy = Ni/(Ti+np.sum(y_predi==i)-Ni)*Ti
                f.write("class {:12s}: #Ni={:7.0f}, #Ti={:7.0f}, f.w.IU={:10.3f}".format(class_names[i],Ni,Ti,accuracy))
                f.write("\n")
                accuracies+=accuracy
            fwIU = accuracies/N
            f.write("_________________")
            f.write("\n")
            f.write("N {:10f}".format(N))
            f.write("\n")
            f.write("FWIU: {:4.3f}".format(fwIU))
            f.write("\n")
            print("FWIU: {:4.3f}".format(fwIU))
            
    def Fmeasure(Yi,y_predi,folder,class_names):
        # Precision (適合率) = TP / (TP + FP)
        # Recall (再現率) = TP / (TP + FN)
        # F1 (適合率と再現率との調和平均) = 2*Precision*Recall / (Precision + Recall)
        #重み付きF値 = (1+beta) * Precision * Recall / (beta^2 * Precision + Recall)
        
        with open(os.path.join(folder,'F1.txt'),'w') as f:
            Precisions = []
            Recalls = []
            Fmeasures = []
            Nclass = int(np.max(Yi)) + 1
            for c in range(Nclass):
                TP = np.sum( (Yi == c)&(y_predi==c) )
                FP = np.sum( (Yi != c)&(y_predi==c) )
                FN = np.sum( (Yi == c)&(y_predi != c))
                Precision = TP / float(TP + FP)
                Recall = TP / float(TP + FN)
                Fmeasure = 2*Precision*Recall / float(Precision + Recall)
                f.write("class {:12s}: #Precision={:4.3f}, #Recall={:4.3f}, #F1={:4.3f}".format(class_names[c],Precision, Recall, Fmeasure))
                f.write("\n")
                Precisions.append(Precision)
                Recalls.append(Recall)
                Fmeasures.append(Fmeasure)
            mPrecision = np.mean(Precisions)
            mRecall = np.mean(Recalls)
            mFmeasure = np.mean(Fmeasures)
            f.write("_________________")
            f.write("\n")
            f.write("Mean Precision: {:4.3f}, Mean Recall: {:4.3f}, Mean F1: {:4.3f}".format(mPrecision, mRecall, mFmeasure))
            f.write("\n")
            print("Mean Precision: {:4.3f}, Mean Recall: {:4.3f}, Mean F1: {:4.3f}".format(mPrecision, mRecall, mFmeasure))
            print("")