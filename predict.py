# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2020-11-25 14:11
# @Software: PyCharm
# @Brief: 预测脚本
import EfficientNet as efn
from dataReader import parse
from tensorflow.keras import losses, optimizers, callbacks
import tensorflow as tf
import os


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    image_dir = r'./dataset/test/'
    label = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    num_classes = len(label)

    model = efn.EfficientNetB3(alpha=1.2, beta=1.4, r=300, classes=num_classes)
    model.load_weights("Efn-B3_02.h5")

    image_list = os.listdir(image_dir)
    image_list.sort(key=lambda x: int(x[:-4]))

    csv = open("result.csv", mode='w', encoding='utf-8')
    i = 0
    for img in image_list:
        img_path = os.path.join(image_dir, img)
        image = parse(img_path, resolution=300, mode='validation')

        logit = model(tf.expand_dims(image, axis=0))
        pred = tf.nn.softmax(logit)
        index = tf.argmax(pred[0]).numpy()
        result = label[index]
        i += 1

        csv.writelines("{},{}\n".format(i, result))
        print(result)

    csv.close()


if __name__ == '__main__':
    main()
