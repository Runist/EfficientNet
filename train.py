# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2020-11-17 11:52
# @Software: PyCharm
# @Brief: 训练脚本
from dataReader import read_data, get_batch_data
import EfficientNet as efn
from tensorflow.keras import losses, optimizers, callbacks
import tensorflow as tf
import os


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_dir = r'D:\Python_Code\BasicNet\dataset\train'
    val_dir = r'D:\Python_Code\BasicNet\dataset\validation'
    epochs = 50
    batch_size = 4
    lr = 1e-4
    num_classes = 5
    resolution = 224

    train_data = read_data(train_dir)
    val_data = read_data(val_dir)

    train_step = len(train_data) // batch_size
    val_step = len(val_data) // batch_size

    train_dataset = get_batch_data(train_data, resolution, num_classes, batch_size, mode='train')
    val_dataset = get_batch_data(val_data, resolution, num_classes, batch_size, mode='validation')

    model = efn.EfficientNetB0(alpha=1.0, beta=1.0, r=resolution, classes=num_classes)
    model.compile(optimizer=optimizers.Adam(lr),
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    cbk = [callbacks.ModelCheckpoint("Efn-B0.h5", save_weights_only=True, save_best_only=True),
           callbacks.ReduceLROnPlateau(monitor='loss',
                                       factor=0.5,
                                       patience=2)]
    model.fit(train_dataset,
              steps_per_epoch=train_step,
              epochs=epochs,
              validation_data=val_dataset,
              validation_steps=val_step,
              callbacks=cbk,
              verbose=1)


if __name__ == '__main__':
    main()
