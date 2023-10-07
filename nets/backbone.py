from functools import wraps

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate, Conv2D, Input,Layer,
                          MaxPooling2D, ZeroPadding2D,Dense,GlobalMaxPooling2D, UpSampling2D, Lambda, MaxPooling2D)
from tensorflow.keras.regularizers import l2
from utils.utils import compose
import tensorflow as tf




class SiLU(Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

#------------------------------------------------------#
#   单次卷积DarknetConv2D
#   如果步长为2则自己设定padding方式。
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 0))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'   
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)
    
#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + SiLU
#---------------------------------------------------#
def DarknetConv2D_BN_SiLU(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    if "name" in kwargs.keys():
        no_bias_kwargs['name'] = kwargs['name'] + '.conv'
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(momentum = 0.97, epsilon = 0.001, name = kwargs['name'] + '.bn'),
        SiLU())
#*args传递的是位置参数，打印每个参数的值； **kwargs把要传递的值打包成字典
def Transition_Block(x, c2, weight_decay=5e-4, name = ""):
    #----------------------------------------------------------------#
    #   利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
    #----------------------------------------------------------------#
    x_1 = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x_1 = DarknetConv2D_BN_SiLU(c2, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x_1)
    
    x_2 = DarknetConv2D_BN_SiLU(c2, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    x_2 = ZeroPadding2D(((1, 1),(1, 1)))(x_2)
    x_2 = DarknetConv2D_BN_SiLU(c2, (3, 3), strides=(2, 2), weight_decay=weight_decay, name = name + '.cv3')(x_2)
    y = Concatenate(axis=-1)([x_2, x_1])
    return y

def Multi_Concat_Block(x, c2, c3, n=4, e=1, ids=[0], weight_decay=5e-4, name = ""):
    c_ = int(c2 * e)
        
    x_1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    x_2 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    
    x_all = [x_1, x_2]
    for i in range(n):
        x_2 = DarknetConv2D_BN_SiLU(c2, (3, 3), weight_decay=weight_decay, name = name + '.cv3.' + str(i))(x_2)
        x_all.append(x_2)
    y = Concatenate(axis=-1)([x_all[id] for id in ids])
    y = DarknetConv2D_BN_SiLU(c3, (1, 1), weight_decay=weight_decay, name = name + '.cv4')(y)
    return y

#---------------------------------------------------#
#   CSPdarknet的主体部分
#   输入为一张640x640x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
def darknet_body(x, transition_channels, block_channels, n, phi, weight_decay=5e-4):
    #-----------------------------------------------#
    #   输入图片是640, 640, 3
    #-----------------------------------------------#
    ids = {
        'l' : [-1, -3, -5, -6],
        'x' : [-1, -3, -5, -7, -8], 
    }[phi]
    #---------------------------------------------------#
    #   base_channels 默认值为64
    #---------------------------------------------------#
    # 320, 320, 3 => 320, 320, 64
    x = DarknetConv2D_BN_SiLU(transition_channels, (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'backbone.stem.0')(x)
    x = ZeroPadding2D(((1, 1),(1, 1)))(x)
    x = DarknetConv2D_BN_SiLU(transition_channels * 2, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'backbone.stem.1')(x)
    x = DarknetConv2D_BN_SiLU(transition_channels * 2, (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'backbone.stem.2')(x)
    
    # 320, 320, 64 => 160, 160, 128
    x = ZeroPadding2D(((1, 1),(1, 1)))(x)
    x = DarknetConv2D_BN_SiLU(transition_channels * 4, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'backbone.dark2.0')(x)
    x = Multi_Concat_Block(x, block_channels * 2, transition_channels * 8, n=n, ids=ids, weight_decay=weight_decay, name = 'backbone.dark2.1')
    #加注意力机制
    # x = cbam_block(x,name='dark2')
    # x = se_block(x,name='dark2')

    # x=ppam(x,channel=transition_channels * 8,name='dark2')
    # 160, 160, 128 => 80, 80, 256
    x = Transition_Block(x, transition_channels * 4, weight_decay=weight_decay, name = 'backbone.dark3.0')
    x = Multi_Concat_Block(x, block_channels * 4, transition_channels * 16, n=n, ids=ids, weight_decay=weight_decay, name = 'backbone.dark3.1')
    #加注意力机制
    # x=cbam_block(x,name='dark3')
    # x=se_block(x,name='dark3')
    # x = ppam(x, channel=transition_channels * 16, name='dark3')
    feat1 = x

    # 80, 80, 256 => 40, 40, 512
    x = Transition_Block(x, transition_channels * 8, weight_decay=weight_decay, name = 'backbone.dark4.0')
    x = Multi_Concat_Block(x, block_channels * 8, transition_channels * 32, n=n, ids=ids, weight_decay=weight_decay, name = 'backbone.dark4.1')
    #加注意力机制
    # x = se_block(x,name='dark4')
    # x = cbam_block(x, name='dark4')
    # x = ppam(x, channel=transition_channels * 32, name='dark4')
    feat2 = x
    
    # 40, 40, 512 => 20, 20, 1024
    x = Transition_Block(x, transition_channels * 16, weight_decay=weight_decay, name = 'backbone.dark5.0')
    x = Multi_Concat_Block(x, block_channels * 8, transition_channels * 32, n=n, ids=ids, weight_decay=weight_decay, name = 'backbone.dark5.1')
    #加注意力机制
    # x = se_block(x,name='dark5')
    # x = cbam_block(x, name='dark5')
    # x = ppam(x, channel=transition_channels * 32, name='dark5')
    feat3 = x

    # ps = feat3   # Input feature map ps with shape [batch_size, height, width, channels]
    # # Apply spatial mask
    # mask = spatial_mask(ps)
    #
    # # Reverse gate
    #
    # # Add the result with the feature map of the previous layer
    # prev_ps = feat2   # Feature map of the previous layer with shape [batch_size, height*2, width*2, channels]
    # prev_ps = DarknetConv2D_BN_SiLU(1024, (1, 1), weight_decay=weight_decay, name="1_P5")(prev_ps)
    # pes = reverse_gate(prev_ps, mask)
    #
    # feat2 = tf.add(pes, prev_ps)
    # feat2 = DarknetConv2D_BN_SiLU(512, (1, 1), weight_decay=weight_decay, name="2_P5")(feat2)
    #
    # ps_1 = feat2  # Input feature map ps with shape [batch_size, height, width, channels]
    # # Apply spatial mask
    # mask = spatial_mask(ps_1)
    # prev_ps = feat1
    # prev_ps = DarknetConv2D_BN_SiLU(512, (1, 1), weight_decay=weight_decay, name="1_P4")(prev_ps)
    # # Reverse gate
    # pes = reverse_gate(prev_ps, mask)
    #
    # # Add the result with the feature map of the previous layer
    #  # Feature map of the previous layer with shape [batch_size, height*2, width*2, channels]
    # feat1 = tf.add(pes, prev_ps)
    # feat1 = DarknetConv2D_BN_SiLU(256, (1, 1), weight_decay=weight_decay, name="2_P4")(feat1)
    return feat1, feat2, feat3
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


def channel_attention(input_feature, ratio=8, name=""):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros',
                             name="channel_attention_shared_one_" + str(name))
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros',
                             name="channel_attention_shared_two_" + str(name))

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature)

    avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
    max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)

    avg_pool = shared_layer_one(avg_pool)
    max_pool = shared_layer_one(max_pool)

    avg_pool = shared_layer_two(avg_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    return tf.keras.layers.multiply([input_feature, cbam_feature])


def spatial_attention(input_feature, name=""):
    kernel_size = 7

    cbam_feature = input_feature

    avg_pool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])

    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name="spatial_attention_" + str(name))(concat)
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    return tf.keras.layers.multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8, name=""):
    cbam_feature = channel_attention(cbam_feature, ratio, name=name)
    x = spatial_attention(cbam_feature, name=name)
    return x
def se_block(input_feature, ratio=16, name=""):
    channel = input_feature.shape[-1]

    se_feature = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
    se_feature = tf.keras.layers.Reshape((1, 1, channel))(se_feature)

    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=False,
                       bias_initializer='zeros',
                       name="se_block_one_" + str(name))(se_feature)

    se_feature = Dense(channel,
                       kernel_initializer='he_normal',
                       use_bias=False,
                       bias_initializer='zeros',
                       name="se_block_two_" + str(name))(se_feature)
    se_feature = tf.keras.layers.Activation('sigmoid')(se_feature)

    x = tf.keras.layers.multiply([input_feature, se_feature])
    return x

def ppam(x, channel,name=""):
    # 多尺度池化
    pooled1 = tf.keras.layers.MaxPool2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    pooled1 = tf.keras.layers.GlobalAveragePooling2D()(pooled1)
    pooled1=tf.keras.layers.Reshape((1, 1,channel))(pooled1)
    pooled2 = tf.keras.layers.MaxPool2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    pooled2 = tf.keras.layers.GlobalAveragePooling2D()(pooled2)
    pooled2 = tf.keras.layers.Reshape((1, 1, channel))(pooled2)
    # 全局平均池化
    reduced = tf.keras.layers.GlobalAveragePooling2D()(x)
    reduced = tf.keras.layers.Reshape((1, 1, channel))(reduced)
    # 对每个分支进行3x3卷积操作得到权重，并使用Sigmoid进行归一化
    weights1 = tf.keras.layers.Conv2D(channel, (3, 3), padding='same', activation='relu')(pooled1)
    weights2 = tf.keras.layers.Conv2D(channel, (3, 3), padding='same', activation='relu')(pooled2)
    weights3 = tf.keras.layers.Conv2D(channel, (3, 3), padding='same', activation='relu')(reduced)
    weighted_sum = tf.add_n([weights1, weights2, weights3])
    weight = tf.keras.layers.Activation('sigmoid')(weighted_sum)

    x = tf.multiply(weight, x)

    return x