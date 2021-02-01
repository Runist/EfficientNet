# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2020-11-25 14:11
# @Software: PyCharm
# @Brief: 预测脚本
import EfficientNet as efn
from dataReader import parse
import tensorflow as tf
import numpy as np
import os


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    img_path = r'daisy.jpg'
    resolution = 224
    class_name = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    num_classes = len(class_name)

    model = efn.EfficientNetB0(alpha=1.0, beta=1.0, r=resolution, classes=num_classes)
    model.load_weights("Efn-B0.h5")

    image = parse(img_path, resolution, mode='test')
    image = tf.expand_dims(image, axis=0)
    logit = model.predict(image)
    pred = np.squeeze(tf.nn.softmax(logit))
    index = int(np.argmax(pred))
    conf = np.max(pred)

    print("Input image is {}, predict is {}, confidence is {:.2f}".format(os.path.basename(img_path),
                                                                          class_name[index], conf))


if __name__ == '__main__':
    main()
