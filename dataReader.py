# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020-11-16 16:35
# @Software: PyCharm
# @Brief: 数据读取
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2 as cv


def read_csv_data(csv_path, label):
    csv = pd.read_csv(csv_path)

    rate = int(0.9 * len(csv))
    train_csv = np.array(csv[1: rate])
    validation_csv = np.array(csv[rate:])

    train_data = list()
    validation_data = list()

    for line in train_csv:
        img_path = "./dataset/train/" + line[0]
        index = label.index(line[1])
        train_data.append((img_path, index))

    for line in validation_csv:
        img_path = "./dataset/train/" + line[0]
        index = label.index(line[1])
        validation_data.append((img_path, index))

    return train_data, validation_data


def read_data(path):
    """
    读取数据，传回图片完整路径列表 和 仅有数字索引列表
    :param path: 数据集路径
    :return: 图片路径列表、数字索引列表
    """
    data = list()
    class_list = os.listdir(path)

    for i, value in enumerate(class_list):
        dirs = os.path.join(path, value)
        for pic in os.listdir(dirs):
            pic_full_path = os.path.join(dirs, pic)
            data.append((pic_full_path, i))

    return data


def parse(img_path, resolution, class_num=None, label=None, mode='train'):
    """
    对数据集批量处理的函数
    :param img_path: 图片路径
    :param resolution: 图片分辨率
    :param class_num: 类别数量
    :param label: 图片标签对应数字所引
    :param mode: 读取数据后用于训练还是验证
    :return: 单个图片和分类
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [resolution, resolution])

    # 在训练集中，有50%的概率增强数据
    if mode == 'train' and np.random.random() < 0.5:
        # h, w, c = image.shape
        # grad_img = image.numpy().copy()
        #
        # for i in range(w):
        #     v = (256 / w)*i
        #     grad_img[:, i, :] = v
        #
        # image = cv.addWeighted(src1=image.numpy(), src2=grad_img, alpha=1.0, beta=0.5, gamma=0.0)
        # image = tf.convert_to_tensor(image, dtype=tf.float32)

        if np.random.random() < 0.5:
            image = tf.image.random_flip_left_right(image)
        if np.random.random() < 0.5:
            image = tf.image.random_flip_up_down(image)
        # if np.random.random() < 0.5:
        #     image = tf.image.random_brightness(image, 0.2)
        # if np.random.random() < 0.5:
        #     image = tf.image.random_contrast(image, 0.3, 2.0)
        # if np.random.random() < 0.5:
        #     image = tf.image.random_hue(image, 0.15)
        # if np.random.random() < 0.5:
        #     image = tf.image.random_saturation(image, 0.3, 2.0)
        # if np.random.random() < 0.5:
        #     image = tf.image.random_crop(image, (100, 100, 3))

    image /= 255.

    if class_num:
        label = tf.one_hot(label, depth=class_num)
        return image, label

    return image


def get_batch_data(data, resolution, num_classes, batch_size, mode):
    """
    获取一个batch
    :param data: 二维数据列表
    :param resolution: 图片分辨率
    :param num_classes: 类别数量
    :param batch_size: 批处理数量
    :param mode: 读取数据后用于训练还是验证
    :return:
    """
    n = len(data)
    i = 0
    while True:
        images = []
        labels = []

        for b in range(batch_size):
            np.random.shuffle(data)
            img, lab = parse(data[i][0], resolution, class_num=num_classes, label=data[i][1], mode=mode)

            images.append(img.numpy())
            labels.append(lab.numpy())
            i = (i+1) % n

        image_data = np.array(images)
        label_data = np.array(labels)

        yield image_data, label_data

