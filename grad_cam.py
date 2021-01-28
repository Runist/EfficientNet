# -*- coding: utf-8 -*-
# @File : grad_cam.py
# @Author: Runist
# @Time : 2020-11-23 13:41
# @Software: PyCharm
# @Brief:
import EfficientNet as efn
import tensorflow as tf
from tensorflow.keras import Input, Model, preprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def read_image(img_path, resolution):
    # 读取数据、要与模型训练数据输入一致
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [resolution, resolution])
    image = image / 255.

    return image


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):

    # 首先，我们创建一个模型，将输入图像映射到最后一个conv层的激活层
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    # 其次，我们创建一个模型，将最后一个conv层的激活映射到最终的类预测
    classifier_input = Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = Model(classifier_input, x)

    # 然后，针对最后一个conv层的激活，计算输入图像的最高预测类的梯度
    with tf.GradientTape() as tape:
        # 计算最后一个conv层的激活层
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # 计算分类预测
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_logits_value = preds[:, top_pred_index]

    # 这是相对于最后一个conv层的输出特征图的最高预测类的梯度
    grads = tape.gradient(top_logits_value, last_conv_layer_output)

    # pooled_grads是一个向量，其中每个项都是特征图通道上渐变的平均强度
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 对于最高预测类别，我们将特征映射数组中的每个通道乘以“渐变的平均强度”
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # 生成的特征图的通道方向均值是我们的类激活热图
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # 出于可视化的目的，我们还将归一化0和1之间的热图，将heatmap元素中大于0的部分除去heatmap的最大值
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def main():

    img_path = r"D:\Python_Code\BasicNet\dataset\sunflower.jpg"
    resolution = 224
    last_conv_layer_name = 'top_activation'
    classifier_layer_names = ['avg_pool', 'top_dropout', 'logits']

    # 读取图像
    image = read_image(img_path, resolution)

    # 建立模型
    model = efn.EfficientNetB0(alpha=1.0, beta=1.0, r=224, classes=5)
    model.summary()
    model.load_weights("Efn-B0.h5")

    # 生成热力图
    heatmap = make_gradcam_heatmap(tf.expand_dims(image, axis=0), model,
                                   last_conv_layer_name, classifier_layer_names)
    plt.matshow(heatmap)
    plt.show()

    # 将热图重新缩放到0-255的范围内
    heatmap = np.uint8(255 * heatmap)
    # 使用喷射色图为热图着色
    jet = cm.get_cmap("jet")    # 生成一个颜色实例

    # 使用颜色图的RGB值
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = preprocessing.image.array_to_img(jet_heatmap)     # 转成PIL格式
    jet_heatmap = jet_heatmap.resize((resolution, resolution))
    jet_heatmap = preprocessing.image.img_to_array(jet_heatmap)

    # 将热图叠加在原始图像上
    superimposed_img = jet_heatmap * 0.01 + image
    superimposed_img = preprocessing.image.array_to_img(superimposed_img)

    # 显示叠加的图像
    superimposed_img.show()


if __name__ == '__main__':
    main()
