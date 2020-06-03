import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

def softmaxND(target, axes=(-1,), name='softmax'): # Defaults to normal Softmax behaviour, but arbitrary axes can be used to "softmax" over.
    with tf.name_scope(name):
        max_axes = tf.math.reduce_max(target, axes, keepdims=True)
        target_exp = tf.math.exp(target-max_axes)
        normalize = tf.math.reduce_sum(target_exp, axes, keepdims=True)
        softmax = target_exp / normalize
    return softmax

class CBAttention(Layer):

    def __init__(self, **kwargs):
        super(CBAttention, self).__init__()
        self.layers = []
        if 'amplification' in kwargs:
            self.amplification = kwargs.pop('amplification')
        else:
            self.amplification = 1   
        if 'input_shape' in kwargs:
            input_shape = kwargs.pop('input_shape')
            self.build(input_shape, **kwargs)

    def build(self, input_shape, **kwargs):
        if 'amplification' in kwargs:
            self.amplification = kwargs.pop('amplification')
        if 'hidden_c_reduction' in kwargs:
            c_mid_layer_s = input_shape[-1] // kwargs['hidden_c_reduction']
        else:
            c_mid_layer_s = input_shape[-1]
        if 's_expansion' in kwargs:
            s_in_layer_channels = kwargs['s_expansion']
        else:
            s_in_layer_channels = 1
        self.layers = [
                        tf.keras.layers.Dense(c_mid_layer_s , activation='elu', input_shape=(input_shape[-1],), name = '{}/c_mlp/dense_1'.format(self.name)),
                        tf.keras.layers.Dense(input_shape[-1], activation='elu', input_shape=(c_mid_layer_s ,), name = '{}/c_mlp/dense_2'.format(self.name)),
                        tf.keras.layers.Dense(input_shape[-1], activation='elu', input_shape=(input_shape[-1],), name = '{}/c_mlp/dense_3'.format(self.name)),
                        tf.keras.layers.Add(input_shape=(input_shape[-1],), name = '{}/c_mlp/add'.format(self.name)),
                        tf.keras.layers.Softmax(input_shape=(input_shape[-1],), name = '{}/c_mlp/softmax'.format(self.name)),
                        tf.keras.layers.Concatenate(input_shape=input_shape[1:3]+[1], name = '{}/s_cnn/concatenate'.format(self.name)),
                        tf.keras.layers.Conv2D(s_in_layer_channels, 7, padding='same', activation='elu', input_shape=input_shape[1:3]+[2], name = '{}/s_cnn/conv2d_1'.format(self.name)),
                        tf.keras.layers.Conv2D(1, 7, padding='same', activation='elu', input_shape=input_shape[1:3]+[s_in_layer_channels], name = '{}/s_cnn/conv2d_2'.format(self.name)),
                        tf.keras.layers.Lambda(softmaxND, arguments = {'axes': (-3, -2)}, input_shape=input_shape[1:3]+[1], name = '{}/s_cnn/lambda/softmaxnd'.format(self.name))
                      ]
        
 
    def call(self, inputs):
        cs = [tf.keras.backend.mean(inputs, (-3, -2)), tf.keras.backend.max(inputs, (-3, -2))]
        for i in range(3):
            for j in range(2):
                cs[j] = self.layers[i](cs[j])
        c_weights = self.layers[4](self.layers[3](cs))
        if self.amplification > 0:
            c_weights = tf.math.multiply((c_weights.shape[-1]) ** (self.amplification / 2), c_weights)
        c_weighted = tf.math.multiply(inputs, c_weights[:,np.newaxis,np.newaxis,:])
        s_coeffs = self.layers[5]([tf.keras.backend.mean(c_weighted, (-1,), keepdims=True), tf.keras.backend.max(c_weighted, (-1,), keepdims=True)])
        for i in range(2):
            s_coeffs = self.layers[6 + i](s_coeffs)
        s_shape = s_coeffs.shape
        s_weights = self.layers[8](s_coeffs)
        if self.amplification > 0:
            s_weights = tf.math.multiply((s_shape[-3]*s_shape[-2]) ** (self.amplification / 2), s_weights)
        return tf.math.multiply(c_weighted, s_weights)

