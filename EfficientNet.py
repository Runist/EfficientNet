# -*- coding: utf-8 -*-
# @File : EfficientNet.py
# @Author: Runist
# @Time : 2020-11-05 11:05
# @Software: PyCharm
# @Brief: EfficientNet基本结构
from tensorflow.keras import layers, models
# collections是Python内建的一个集合模块，提供了许多有用的集合类。
import collections
import math
import string


# namedtuple是一个函数，它用来创建一个自定义的tuple对象，并且规定了tuple元素的个数，并可以用属性而不是索引来引用tuple的某个元素。
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'seq_ratio'
])

# 每个MBConv的参数
DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=1, seq_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=2, seq_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=2, seq_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=2, seq_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=1, seq_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=2, seq_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=1, seq_ratio=0.25)
]


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet实际上使用未截断的正态分布来初始化conv层，但是keras.initializers.VarianceScaling使用截断了的分布。
        # 我们决定使用自定义初始化程序，以实现更好的可序列化性
        'distribution': 'normal'
    }
}


DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def round_filters(filters, width_coefficient, depth_divisor):
    """
    计算卷积核缩放后的数量，但要保证能被8整除
    :param filters: 原本卷积核个数
    :param width_coefficient: 网络宽度的缩放系数
    :param depth_divisor:
    :return:
    """

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)

    # 确保卷积核数不低于原来的90%
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor

    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """
    基于深度乘数的重复次数的整数。
    :param repeats: 重复次数
    :param depth_coefficient: 网络深度的缩放系数
    :return:
    """
    # 向上取整
    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix=''):

    has_seq = (block_args.seq_ratio is not None) and (0 < block_args.seq_ratio <= 1)
    filters = block_args.input_filters * block_args.expand_ratio

    if block_args.expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=prefix + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=-1, name=prefix + 'expand_bn')(x)
        x = layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(block_args.kernel_size,
                               strides=block_args.strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + 'dw_conv')(x)
    x = layers.BatchNormalization(axis=-1, name=prefix + 'bn')(x)
    x = layers.Activation(activation, name=prefix + 'activation')(x)

    # 压缩后再放大，作为一个调整系数
    if has_seq:
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.seq_ratio))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'seq_squeeze')(x)

        se_tensor = layers.Reshape((1, 1, filters), name=prefix + 'seq_reshape')(se_tensor)
        se_tensor = layers.Conv2D(num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'seq_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'seq_expand')(se_tensor)

        x = layers.multiply([x, se_tensor], name=prefix + 'seq_excite')

    # 利用1x1卷积对特征层进行压缩
    x = layers.Conv2D(block_args.output_filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=prefix + 'project_conv')(x)
    x = layers.BatchNormalization(axis=-1, name=prefix + 'project_bn')(x)

    # 实现残差网络
    if block_args.id_skip and block_args.strides == 1 and block_args.input_filters == block_args.output_filters:

        if drop_rate and (drop_rate > 0):
            layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=prefix + 'drop')(x)

        x = layers.add([x, inputs], name=prefix + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 input_shape=None,
                 model_name='EfficientNet',
                 classes=1000):
    """
    EfficientNet模型结构
    :param width_coefficient: float，网络宽度的缩放系数
    :param depth_coefficient: float，网络深度的缩放系数
    :param resolution: int，图片分辨率
    :param dropout_rate: 最后一层前的dropout系数
    :param drop_connect_rate: 跳过连接时的概率
    :param depth_divisor: int
    :param blocks_args: 用于构造块模块的BlockArgs列表
    :param input_shape: 模型输入shape
    :param model_name: string，模型名字
    :param classes: 分类数量
    :return:
    """
    if input_shape:
        img_input = layers.Input(shape=input_shape)
    else:
        img_input = layers.Input(shape=(resolution, resolution, 3))

    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor),
                      kernel_size=3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(img_input)
    x = layers.BatchNormalization(axis=-1, name='stem_bn')(x)
    x = layers.Activation('swish', name='stem_activation')(x)

    # 计算MBConv总共重复的次数
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0

    for idx, block_args in enumerate(blocks_args):

        # 根据深度乘法器更新块输入和输出卷积核个数
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters, width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters, width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # 逐层增加drop_rate的概率
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        # 第一个MBConv块需要注意步长和过滤器尺寸的增加
        x = mb_conv_block(x, block_args,
                          activation='swish',
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))

        block_num += 1
        if block_args.num_repeat > 1:
            # 因为前面修改过卷积核的个数，所以后面的卷积核个数也需要修改，保证MBConv卷积最后输入输出一样
            block_args = block_args._replace(input_filters=block_args.output_filters, strides=1)

            for b_idx in range(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(idx + 1, string.ascii_lowercase[b_idx + 1])

                x = mb_conv_block(x,
                                  block_args,
                                  activation='swish',
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1

    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor),
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)

    x = layers.BatchNormalization(axis=-1, name='top_bn')(x)
    x = layers.Activation('swish', name='top_activation')(x)

    # 利用GlobalAveragePooling2D代替全连接层
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='top_dropout')(x)

    x = layers.Dense(classes,
                     # activation='softmax',
                     kernel_initializer=DENSE_KERNEL_INITIALIZER,
                     name='logits')(x)

    model = models.Model(img_input, x, name=model_name)
    model.summary()

    return model


def EfficientNetB0(
        alpha=1.0, beta=1.0, r=224,
        input_shape=None,
        classes=1000,
):
    return EfficientNet(
        alpha, beta, r, 0.2,
        model_name='efficientnet-b0',
        input_shape=input_shape,
        classes=classes)


def EfficientNetB1(
        alpha=1.0, beta=1.1, r=240,
        input_shape=None,
        classes=1000,
):
    return EfficientNet(
        alpha, beta, r, 0.2,
        model_name='efficientnet-b1',
        input_shape=input_shape,
        classes=classes)


def EfficientNetB2(
        alpha=1.1, beta=1.2, r=260,
        input_shape=None,
        classes=1000,
):
    return EfficientNet(
        alpha, beta, r, 0.3,
        model_name='efficientnet-b2',
        input_shape=input_shape,
        classes=classes)


def EfficientNetB3(
        alpha=1.2, beta=1.4, r=300,
        input_shape=None,
        classes=1000,
):
    return EfficientNet(
        alpha, beta, r, 0.3,
        model_name='efficientnet-b3',
        input_shape=input_shape,
        classes=classes)


def EfficientNetB4(
        alpha=1.4, beta=1.8, r=380,
        input_shape=None,
        classes=1000,
):
    return EfficientNet(
        alpha, beta, r, 0.4,
        model_name='efficientnet-b4',
        input_shape=input_shape,
        classes=classes)


def EfficientNetB5(
        alpha=1.6, beta=2.2, r=456,
        input_shape=None,
        classes=1000,
):
    return EfficientNet(
        alpha, beta, r, 0.4,
        model_name='efficientnet-b5',
        input_shape=input_shape,
        classes=classes)


def EfficientNetB6(
        alpha=1.8, beta=2.6, r=528,
        input_shape=None,
        classes=1000,
):
    return EfficientNet(
        alpha, beta, r, 0.5,
        model_name='efficientnet-b6',
        input_shape=input_shape,
        classes=classes)


def EfficientNetB7(
        alpha=2.0, beta=3.1, r=600,
        input_shape=None,
        classes=1000,
):
    return EfficientNet(
        alpha, beta, r, 0.5,
        model_name='efficientnet-b7',
        input_shape=input_shape,
        classes=classes)


def customEfficientNet(
        alpha, beta, r,
        input_shape=None,
        classes=1000,
):
    return EfficientNet(
        alpha, beta, r, 0.2,
        model_name='efficientnet-b7',
        input_shape=input_shape,
        classes=classes)
