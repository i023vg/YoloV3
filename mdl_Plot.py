import matplotlib.pyplot as plt
import os

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# ***** モデルの正解率の変化をプロット *****
def plot_history_acc(fit):
    axL.plot(fit.history['acc'], label="acc for training")
    axL.plot(fit.history['val_acc'], label="acc for validation")
    axL.set_title('model accuracy')
    axL.set_ylabel('accuracy')
    axL.set_xlabel('epoch')
    axL.legend()

 # ***** モデルの損失の変化をプロット *****
def plot_history_loss(fit):
    axR.plot(fit.history['loss'], label="loss for training")
    axR.plot(fit.history['val_loss'], label="loss for validation")
    axR.set_title('model loss')
    axR.set_ylabel('loss')
    axR.set_xlabel('epoch')
    axR.legend()