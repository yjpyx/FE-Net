import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate, Conv2D, Input,
                                    Lambda, MaxPooling2D, UpSampling2D)
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, Activation
from tensorflow.keras.models import Model

from nets.backbone import (DarknetConv2D, DarknetConv2D_BN_SiLU,
                           Multi_Concat_Block, SiLU, Transition_Block,
                           darknet_body)
from nets.yolo_training import yolo_loss

# def SPPCSPC(x, c2, n=1, shortcut=False, g=1, e=0.5, k=(5,9,13), weight_decay=5e-4, name=""):
#     c_ = int(2 * c2 * e)  # hidden channels
#     x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
#     x1 = DarknetConv2D_BN_SiLU(c_, (3, 3), weight_decay=weight_decay, name = name + '.cv3')(x1)
#     x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv4')(x1)
#
#     y1 = Concatenate(axis=-1)([x1] + [MaxPooling2D(pool_size=(m, m), strides=(1, 1), padding='same')(x1) for m in k])
#     y1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv5')(y1)
#     y1 = DarknetConv2D_BN_SiLU(c_, (3, 3), weight_decay=weight_decay, name = name + '.cv6')(y1)
#
#     y2 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
#     out = Concatenate(axis=-1)([y1, y2])
#     out = DarknetConv2D_BN_SiLU(c2, (1, 1), weight_decay=weight_decay, name = name + '.cv7')(out)
#     #可以换用不同的结构诸如SPP、SPPF、ASPP、以及将SPPCSPC和ASPP结合
#     return out
# def SPPCSPC(x, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), weight_decay=5e-4, name=""):
#     #SPP结构
#     #---------------------------------------------------#
#     #   使用了SPP结构，即不同尺度的最大池化后堆叠。
#     #---------------------------------------------------#
#     out_channels  = int(4 * c2 * e)
#     x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
#     maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
#     maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
#     maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
#     x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
#     out = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
#     return out
# def SPPCSPC(x, c2, n=1, shortcut=False, g=1, e=0.5, k=(1, 3, 5), weight_decay=5e-4, name=""):
#     #SPPF结构
#     #---------------------------------------------------#
#     #   使用了SPP结构，即不同尺度的最大池化后堆叠。
#     #---------------------------------------------------#
#     out_channels  = int(4 * c2 * e)
#     x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
#     maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)
#     maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool1)
#     maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(maxpool2)
#     maxpool4 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(maxpool3)
#     maxpool5 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(maxpool4)
#     maxpool6 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(maxpool5)
#     x = Concatenate()([x, maxpool1, maxpool2, maxpool3,maxpool4,maxpool5,maxpool6])
#     out = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
#     return out



# def SPPCSPC(x, c2, n=1, shortcut=False, g=1, e=0.5, k=(1, 3, 5), weight_decay=5e-4, name=""):
#     #ASPP结构
#     out_channels = int(4 * c2 * e)
#     # 创建ASPP模块的四个分支
#     branch_outputs = []
#
#     # 分支1：使用3x3卷积
#     x1 = DepthwiseConv2D((3, 3),dilation_rate=2,padding='same', depthwise_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
#     x1 = BatchNormalization()(x1)
#     x1 = Activation('relu')(x1)
#     # Inter-channel Convolution
#     x1 = Conv2D(c2, (1, 1), padding='same', use_bias=False,
#                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1)
#     x1 = BatchNormalization()(x1)
#     x1 = Activation('relu')(x1)
#     branch_outputs.append(x1)
#
#     # 分支2：使用3x3卷积
#     x2 = DepthwiseConv2D((3, 3), dilation_rate=4, padding='same',
#                          depthwise_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
#     x2 = BatchNormalization()(x2)
#     x2 = Activation('relu')(x2)
#     # Inter-channel Convolution
#     x2 = Conv2D(c2, (1, 1), padding='same', use_bias=False,
#                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2)
#     x2 = BatchNormalization()(x2)
#     x2 = Activation('relu')(x2)
#     branch_outputs.append(x2)
#
#     # 分支3：使用3x3卷积
#     x3 = DepthwiseConv2D((3, 3), dilation_rate=6, padding='same',
#                          depthwise_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
#     x3 = BatchNormalization()(x3)
#     x3 = Activation('relu')(x3)
#     # Inter-channel Convolution
#     x3 = Conv2D(c2, (1, 1), padding='same', use_bias=False,
#                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3)
#     x3 = BatchNormalization()(x3)
#     x3 = Activation('relu')(x3)
#     branch_outputs.append(x3)
#
#     # 分支4：使用1x1卷积
#     x4 = Conv2D(c2, (1, 1), padding='same', use_bias=False,
#                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
#     x4 = BatchNormalization()(x4)
#     x4 = Activation('relu')(x4)
#     branch_outputs.append(x4)
#
#     # 将四个分支输出堆叠在一起
#     out = tf.keras.layers.concatenate(branch_outputs)
#     out = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name=name + '.cv2')(out)
#     return out

# def SPPCSPC(x, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), weight_decay=5e-4, name=""):
#     #ours+SPPF结构
#     c_ = int(2 * c2 * e)  # hidden channels
#     branch_outputs = []
#     x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
#     x1 = DarknetConv2D_BN_SiLU(c_, (3, 3), weight_decay=weight_decay, name = name + '.cv3')(x1)
#     x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv4')(x1)
#     maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x1)
#     maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool1)
#     maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(maxpool2)
#     maxpool4 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(maxpool3)
#     maxpool5 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(maxpool4)
#     maxpool6 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(maxpool5)
#     y1= Concatenate()([x1, maxpool1, maxpool2, maxpool3,maxpool4,maxpool5,maxpool6])
#     y1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv5')(y1)
#     y1 = DarknetConv2D_BN_SiLU(c_, (3, 3), weight_decay=weight_decay, name = name + '.cv6')(y1)
#
#     y2 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
#     out = Concatenate(axis=-1)([y1, y2])
#     out = DarknetConv2D_BN_SiLU(c2, (1, 1), weight_decay=weight_decay, name = name + '.cv7')(out)
#     #可以换用不同的结构诸如SPP、SPPF、ASPP、以及将SPPCSPC和ASPP结合
#     return out
# def SPPCSPC(x, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), weight_decay=5e-4, name=""):
#     #ours+SPPF结构SPPF_t-CSPC
#     c_ = int(2 * c2 * e)  # hidden channels
#     branch_outputs = []
#     x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
#     x1 = DarknetConv2D_BN_SiLU(c_, (3, 3), weight_decay=weight_decay, name = name + '.cv3')(x1)
#     x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv4')(x1)
#     maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x1)
#     maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool1)
#     maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool2)
#     maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool3)
#     maxpool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool4)
#     maxpool6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool5)
#     maxpool7 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool6)
#     maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool7)
#     maxpool9 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool8)
#     maxpool10 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool9)
#     y1= Concatenate()([x1, maxpool1, maxpool2, maxpool4,maxpool6,maxpool8,maxpool10])
#     y1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv5')(y1)
#     y1 = DarknetConv2D_BN_SiLU(c_, (3, 3), weight_decay=weight_decay, name = name + '.cv6')(y1)
#
#     y2 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
#     out = Concatenate(axis=-1)([y1, y2])
#     out = DarknetConv2D_BN_SiLU(c2, (1, 1), weight_decay=weight_decay, name = name + '.cv7')(out)
#     #可以换用不同的结构诸如SPP、SPPF、ASPP、以及将SPPCSPC和ASPP结合
#     return out
def SPPCSPC(x, c2, n=1, shortcut=False, g=1, e=0.5, k=(1, 3, 5), weight_decay=5e-4, name=""):
    #SPPF_t结构
    #---------------------------------------------------#
    #   使用了SPP结构，即不同尺度的最大池化后堆叠。
    #---------------------------------------------------#
    out_channels  = int(4 * c2 * e)
    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool1)
    maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool2)
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool3)
    maxpool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool4)
    maxpool6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool5)
    maxpool7 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool6)
    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool7)
    maxpool9 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool8)
    maxpool10 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(maxpool9)
    x= Concatenate()([x, maxpool1, maxpool2, maxpool4,maxpool6,maxpool8,maxpool10])
    out = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    return out
# def SPPCSPC(x, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), weight_decay=5e-4, name=""):
#     #ours+ASPP结构
#     c_ = int(2 * c2 * e)  # hidden channels
#     branch_outputs = []
#     x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
#     x1 = DarknetConv2D_BN_SiLU(c_, (3, 3), weight_decay=weight_decay, name = name + '.cv3')(x1)
#     x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv4')(x1)
#     x1 = DepthwiseConv2D((3, 3),dilation_rate=2,padding='same', depthwise_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
#     x1 = BatchNormalization()(x1)
#     x1 = Activation('relu')(x1)
#         # Inter-channel Convolution
#     x1 = Conv2D(c2, (1, 1), padding='same', use_bias=False,
#                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x1)
#     x1 = BatchNormalization()(x1)
#     x1 = Activation('relu')(x1)
#     branch_outputs.append(x1)
#
#         # 分支2：使用3x3卷积
#     x2 = DepthwiseConv2D((3, 3), dilation_rate=4, padding='same',
#                              depthwise_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
#     x2 = BatchNormalization()(x2)
#     x2 = Activation('relu')(x2)
#         # Inter-channel Convolution
#     x2 = Conv2D(c2, (1, 1), padding='same', use_bias=False,
#                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x2)
#     x2 = BatchNormalization()(x2)
#     x2 = Activation('relu')(x2)
#     branch_outputs.append(x2)
#
#         # 分支3：使用3x3卷积
#     x3 = DepthwiseConv2D((3, 3), dilation_rate=6, padding='same',
#                              depthwise_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
#     x3 = BatchNormalization()(x3)
#     x3 = Activation('relu')(x3)
#         # Inter-channel Convolution
#     x3 = Conv2D(c2, (1, 1), padding='same', use_bias=False,
#                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x3)
#     x3 = BatchNormalization()(x3)
#     x3 = Activation('relu')(x3)
#     branch_outputs.append(x3)
#
#         # 分支4：使用1x1卷积
#     x4 = Conv2D(c2, (1, 1), padding='same', use_bias=False,
#                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
#     x4 = BatchNormalization()(x4)
#     x4 = Activation('relu')(x4)
#     branch_outputs.append(x4)
#     y1= tf.keras.layers.concatenate(branch_outputs)
#     y1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv5')(y1)
#     y1 = DarknetConv2D_BN_SiLU(c_, (3, 3), weight_decay=weight_decay, name = name + '.cv6')(y1)
#
#     y2 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
#     out = Concatenate(axis=-1)([y1, y2])
#     out = DarknetConv2D_BN_SiLU(c2, (1, 1), weight_decay=weight_decay, name = name + '.cv7')(out)
#     #可以换用不同的结构诸如SPP、SPPF、ASPP、以及将SPPCSPC和ASPP结合
#     return out
def fusion_rep_vgg(fuse_layers, trained_model, infer_model):
    for layer_name, use_bias, use_bn in fuse_layers:

        conv_kxk_weights = trained_model.get_layer(layer_name + '.rbr_dense.0').get_weights()[0]
        conv_1x1_weights = trained_model.get_layer(layer_name + '.rbr_1x1.0').get_weights()[0]

        if use_bias:
            conv_kxk_bias = trained_model.get_layer(layer_name + '.rbr_dense.0').get_weights()[1]
            conv_1x1_bias = trained_model.get_layer(layer_name + '.rbr_1x1.0').get_weights()[1]
        else:
            conv_kxk_bias = np.zeros((conv_kxk_weights.shape[-1],))
            conv_1x1_bias = np.zeros((conv_1x1_weights.shape[-1],))

        if use_bn:
            gammas_kxk, betas_kxk, means_kxk, var_kxk = trained_model.get_layer(layer_name + '.rbr_dense.1').get_weights()
            gammas_1x1, betas_1x1, means_1x1, var_1x1 = trained_model.get_layer(layer_name + '.rbr_1x1.1').get_weights()

        else:
            gammas_1x1, betas_1x1, means_1x1, var_1x1 = [np.ones((conv_1x1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_weights.shape[-1],)),
                                                         np.zeros((conv_1x1_weights.shape[-1],)),
                                                         np.ones((conv_1x1_weights.shape[-1],))]
            gammas_kxk, betas_kxk, means_kxk, var_kxk = [np.ones((conv_kxk_weights.shape[-1],)),
                                                         np.zeros((conv_kxk_weights.shape[-1],)),
                                                         np.zeros((conv_kxk_weights.shape[-1],)),
                                                         np.ones((conv_kxk_weights.shape[-1],))]
        gammas_res, betas_res, means_res, var_res = [np.ones((conv_1x1_weights.shape[-1],)),
                                                     np.zeros((conv_1x1_weights.shape[-1],)),
                                                     np.zeros((conv_1x1_weights.shape[-1],)),
                                                     np.ones((conv_1x1_weights.shape[-1],))]

        # _fuse_bn_tensor(self.rbr_dense)
        w_kxk = (gammas_kxk / np.sqrt(np.add(var_kxk, 1e-3))) * conv_kxk_weights
        b_kxk = (((conv_kxk_bias - means_kxk) * gammas_kxk) / np.sqrt(np.add(var_kxk, 1e-3))) + betas_kxk
        
        # _fuse_bn_tensor(self.rbr_dense)
        kernel_size = w_kxk.shape[0]
        in_channels = w_kxk.shape[2]
        w_1x1 = np.zeros_like(w_kxk)
        w_1x1[kernel_size // 2, kernel_size // 2, :, :] = (gammas_1x1 / np.sqrt(np.add(var_1x1, 1e-3))) * conv_1x1_weights
        b_1x1 = (((conv_1x1_bias - means_1x1) * gammas_1x1) / np.sqrt(np.add(var_1x1, 1e-3))) + betas_1x1

        w_res = np.zeros_like(w_kxk)
        for i in range(in_channels):
            w_res[kernel_size // 2, kernel_size // 2, i % in_channels, i] = 1
        w_res = ((gammas_res / np.sqrt(np.add(var_res, 1e-3))) * w_res)
        b_res = (((0 - means_res) * gammas_res) / np.sqrt(np.add(var_res, 1e-3))) + betas_res

        weight = [w_res, w_1x1, w_kxk]
        bias = [b_res, b_1x1, b_kxk]
        
        infer_model.get_layer(layer_name).set_weights([np.array(weight).sum(axis=0), np.array(bias).sum(axis=0)])

def RepConv(x, c2, mode="train", weight_decay=5e-4, name=""):
    if mode == "predict":
        out = DarknetConv2D(c2, (3, 3), name = name, use_bias=True, weight_decay=weight_decay, padding='same')(x)
        out = SiLU()(out)
    elif mode == "train":
        x1 = DarknetConv2D(c2, (3, 3), name = name + '.rbr_dense.0', use_bias=False, weight_decay=weight_decay, padding='same')(x)
        x1 = BatchNormalization(momentum = 0.97, epsilon = 0.001, name = name + '.rbr_dense.1')(x1)
        x2 = DarknetConv2D(c2, (1, 1), name = name + '.rbr_1x1.0', use_bias=False, weight_decay=weight_decay, padding='same')(x)
        x2 = BatchNormalization(momentum = 0.97, epsilon = 0.001, name = name + '.rbr_1x1.1')(x2)
        
        out = Add()([x1, x2])
        out = SiLU()(out)
    return out

#---------------------------------------------------#
#   Panet网络的构建，并且获得预测结果
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes, phi, weight_decay=5e-4, mode="train"):
    #-----------------------------------------------#
    #   定义了不同yolov7版本的参数
    #-----------------------------------------------#
    transition_channels = {'l' : 32, 'x' : 40}[phi]
    block_channels      = 32
    panet_channels      = {'l' : 32, 'x' : 64}[phi]
    e       = {'l' : 2, 'x' : 1}[phi]
    n       = {'l' : 4, 'x' : 6}[phi]
    ids     = {'l' : [-1, -2, -3, -4, -5, -6], 'x' : [-1, -3, -5, -7, -8]}[phi]

    inputs      = Input(input_shape)
    #---------------------------------------------------#   
    #   生成主干模型，获得三个有效特征层，他们的shape分别是：
    #   80, 80, 256
    #   40, 40, 1024
    #   20, 20, 1024
    #---------------------------------------------------#
    feat1, feat2, feat3 = darknet_body(inputs, transition_channels, block_channels, n, phi, weight_decay)

    # 20, 20, 1024 -> 20, 20, 512
    P5          = SPPCSPC(feat3, transition_channels * 16, weight_decay=weight_decay, name="sppcspc")
    P5_conv     = DarknetConv2D_BN_SiLU(transition_channels * 8, (1, 1), weight_decay=weight_decay, name="conv_for_P5")(P5)
    P5_upsample = UpSampling2D()(P5_conv)
    P4          = Concatenate(axis=-1)([DarknetConv2D_BN_SiLU(transition_channels * 8, (1, 1), weight_decay=weight_decay, name="conv_for_feat2")(feat2), P5_upsample])
    P4          = Multi_Concat_Block(P4, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids, weight_decay=weight_decay, name="conv3_for_upsample1")

    P4_conv     = DarknetConv2D_BN_SiLU(transition_channels * 4, (1, 1), weight_decay=weight_decay, name="conv_for_P4")(P4)
    P4_upsample = UpSampling2D()(P4_conv)
    P3          = Concatenate(axis=-1)([DarknetConv2D_BN_SiLU(transition_channels * 4, (1, 1), weight_decay=weight_decay, name="conv_for_feat1")(feat1), P4_upsample])
    P3          = Multi_Concat_Block(P3, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids, weight_decay=weight_decay, name="conv3_for_upsample2")
        
    P3_downsample = Transition_Block(P3, transition_channels * 4, weight_decay=weight_decay, name="down_sample1")
    P4 = Concatenate(axis=-1)([P3_downsample, P4])
    P4 = Multi_Concat_Block(P4, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids, weight_decay=weight_decay, name="conv3_for_downsample1")
    # pts = P4
    # P4 = calculate_saliency_map(pts)

    P4_downsample = Transition_Block(P4, transition_channels * 8, weight_decay=weight_decay, name="down_sample2")
    P5 = Concatenate(axis=-1)([P4_downsample, P5])
    P5 = Multi_Concat_Block(P5, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids, weight_decay=weight_decay, name="conv3_for_downsample2")



    if phi == "l":
        P3 = RepConv(P3, transition_channels * 8, mode, weight_decay=weight_decay, name="rep_conv_1")
        P4 = RepConv(P4, transition_channels * 16, mode, weight_decay=weight_decay, name="rep_conv_2")

        P5 = RepConv(P5, transition_channels * 32, mode, weight_decay=weight_decay, name="rep_conv_3")
         # Input feature map ps with shape [batch_size, height, width, channels]
        # Example usage\
        #IIE模块一
        # ps = P5  # Input feature map ps with shape [batch_size, height, width, channels]
        # # Apply spatial mask
        # mask = spatial_mask(ps)
        #
        # # Reverse gate
        #
        # # Add the result with the feature map of the previous layer
        # prev_ps = P4  # Feature map of the previous layer with shape [batch_size, height*2, width*2, channels]
        # prev_ps = DarknetConv2D_BN_SiLU(1024, (1, 1), weight_decay=weight_decay, name="1_P5")(prev_ps)
        # pes = reverse_gate(prev_ps, mask)
        #
        # P4_1 = tf.add(pes, prev_ps)
        # P4_1= DarknetConv2D_BN_SiLU(512, (1, 1), weight_decay=weight_decay, name="2_P5")(P4_1)
        # # #OSE模块P4
        # pts=P4
        # P4=calculate_saliency(pts)
        #
        # #IIE模块二
        # #p3优化
        # ps_1 = P4  # Input feature map ps with shape [batch_size, height, width, channels]
        # # Apply spatial mask
        # mask = spatial_mask(ps_1)
        # prev_ps = P3
        # prev_ps = DarknetConv2D_BN_SiLU(512, (1, 1), weight_decay=weight_decay, name="1_P4")(prev_ps)
        # # Reverse gate
        # pes = reverse_gate(prev_ps, mask)
        #
        # # Add the result with the feature map of the previous layer
        #  # Feature map of the previous layer with shape [batch_size, height*2, width*2, channels]
        # P3_1 = tf.add(pes, prev_ps)
        # P3_1 = DarknetConv2D_BN_SiLU(256, (1, 1), weight_decay=weight_decay, name="2_P4")(P3_1)
        #
        # #OSE模块P3
        # pts = P3
        # P3 = calculate_saliency(pts)
        # # # #OSE模块P5
        # P5 = calculate_saliency(P5)
    else:
        P3 = DarknetConv2D_BN_SiLU(transition_channels * 8, (3, 3), strides=(1, 1), weight_decay=weight_decay, name="rep_conv_1")(P3)
        P4 = DarknetConv2D_BN_SiLU(transition_channels * 16, (3, 3), strides=(1, 1), weight_decay=weight_decay, name="rep_conv_2")(P4)
        P5 = DarknetConv2D_BN_SiLU(transition_channels * 32, (3, 3), strides=(1, 1), weight_decay=weight_decay, name="rep_conv_3")(P5)

    # len(anchors_mask[2]) = 3
    # 5 + num_classes -> 4 + 1 + num_classes
    # 4是先验框的回归系数，1是sigmoid将值固定到0-1，num_classes用于判断先验框是什么类别的物体
    # bs, 20, 20, 3 * (4 + 1 + num_classes)
    out2 = DarknetConv2D(len(anchors_mask[2]) * (5 + num_classes), (1, 1), weight_decay=weight_decay, strides = (1, 1), name = 'yolo_head_P3')(P3)
    out1 = DarknetConv2D(len(anchors_mask[1]) * (5 + num_classes), (1, 1), weight_decay=weight_decay, strides = (1, 1), name = 'yolo_head_P4')(P4)
    out0 = DarknetConv2D(len(anchors_mask[0]) * (5 + num_classes), (1, 1), weight_decay=weight_decay, strides = (1, 1), name = 'yolo_head_P5')(P5)
    return Model(inputs, [out0, out1, out2])


def spatial_mask(ps):
    # Upsample the feature map using upsampling rate of 2

    upsampled_ps = UpSampling2D(size=(2, 2))(ps)

    # Apply max pooling and global pooling to obtain spatial layout characteristics
    max_pool = tf.reduce_max(upsampled_ps, axis=-1, keepdims=True)
    global_pool = tf.reduce_mean(upsampled_ps, axis=(1, 2), keepdims=True)

    # Element-wise multiplication and sigmoid activation
    mask = tf.sigmoid(max_pool * global_pool)

    return mask


def reverse_gate(prev_ps, mask):
    # Subtract the calculated mask from a feature map with all elements equal to 1
    ones = tf.ones_like(mask)
    reverse_mask = ones - mask
    # Element-wise multiplication with the previous feature map
    pes = prev_ps * reverse_mask

    return pes


def calculate_saliency(pts, alpha=0.3):
    Tws = tf.zeros_like(pts)

    # 获取特征图的高度和宽度
    height, width = tf.shape(pts)[1], tf.shape(pts)[2]

    # 创建一个周围是1，其余地方是0的掩码
    mask = tf.ones((height - 2, width - 2))
    mask = tf.pad(mask, [[1, 1], [1, 1]])
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.tile(mask, [1, 1, tf.shape(pts)[3]])

    # 将Tws和掩码逐元素相乘
    Tws = Tws + mask

    # 将Tws和pts逐元素相乘
    product = tf.multiply(Tws, pts)

    # 求和
    sum_vbgs = tf.reduce_sum(product, axis=[1, 2, 3])

    # 求平均
    average_vbgs = sum_vbgs / tf.cast((height - 2) * (width - 2) * tf.shape(pts)[3], tf.float32)

    # 计算pts和Vbgs之间的差值△Ts
    delta_ts = pts - average_vbgs[:, tf.newaxis, tf.newaxis, tf.newaxis]

    # 定义一个函数，计算1-e^(-alpha * delta_ts)（当delta_ts > 0时）
    def activation_function(delta_ts, alpha):
        return tf.where(delta_ts > 0, 1 - tf.exp(-alpha * delta_ts), tf.zeros_like(delta_ts))
    # 计算激活函数的输出
    pds = activation_function(delta_ts, alpha) * pts + pts

    return pds


def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), 2)) for l in range(len(anchors_mask))] + [Input(shape = [None, 5])]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {
            'input_shape'       : input_shape, 
            'anchors'           : anchors, 
            'anchors_mask'      : anchors_mask, 
            'num_classes'       : num_classes, 
            'label_smoothing'   : label_smoothing, 
            'balance'           : [0.4, 1.0, 4],
            'box_ratio'         : 0.05,
            'obj_ratio'         : 1 * (input_shape[0] * input_shape[1]) / (640 ** 2), 
            'cls_ratio'         : 0.5 * (num_classes / 80)
        }
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
