import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class CBAttention(Layer):

    def __init__(self, **kwargs):
        super(CBAttention, self).__init__()
        self.layers = []
        if 'input_shape' in kwargs:
            input_shape = kwargs.pop('input_shape')
            self.build(input_shape, **kwargs)

    def build(self, input_shape, **kwargs):
        if 'hidden_c_reduction' in kwargs:
            c_mid_layer_s = input_shape[-1] // kwargs['hidden_c_reduction']
        else:
            c_mid_layer_s = input_shape[-1]
        if 's_expansion' in kwargs:
            s_in_layer_channels = kwargs['s_expansion']
        else:
            s_in_layer_channels = 1
        self.layers = [
                        tf.keras.layers.GlobalAveragePooling2D(input_shape=input_shape[1:]),
                        tf.keras.layers.GlobalMaxPool2D(input_shape=input_shape[1:]),
                        tf.keras.layers.Dense(c_mid_layer_s , activation='elu', input_shape=(input_shape[-1],), name = '{}/c_mlp/dense_1'.format(self.name)),
                        tf.keras.layers.Dense(input_shape[-1], activation='elu', input_shape=(c_mid_layer_s ,), name = '{}/c_mlp/dense_2'.format(self.name)),
                        tf.keras.layers.Dense(input_shape[-1], activation='elu', input_shape=(input_shape[-1],), name = '{}/c_mlp/dense_3'.format(self.name)),
                        tf.keras.layers.Add(input_shape=(input_shape[-1],),name = '{}/c_mlp/add'.format(self.name)),
                        tf.keras.layers.Activation('sigmoid', input_shape=(input_shape[-1],), name = '{}/c_mlp/sigmoid'.format(self.name)),
                        tf.keras.layers.Multiply(input_shape=input_shape[1:], name = '{}/multiply_1'.format(self.name)),
                        tf.keras.layers.Concatenate(input_shape=input_shape[1:3]+[1], name = '{}/s_cnn/concatenate'.format(self.name)),
                        tf.keras.layers.Conv2D(s_in_layer_channels, 7, padding='same', activation='elu', input_shape=input_shape[1:3]+[2], name = '{}/s_cnn/conv2d_1'.format(self.name)),
                        tf.keras.layers.Conv2D(1, 7, padding='same', activation='elu', input_shape=input_shape[1:3]+[s_in_layer_channels], name = '{}/s_cnn/conv2d_2'.format(self.name)),
                        tf.keras.layers.Activation('sigmoid', input_shape=input_shape[1:3]+[1], name = '{}/s_cnn/sigmoid'.format(self.name)),
                        tf.keras.layers.Multiply(input_shape=input_shape[1:], name = '{}/multiply_2'.format(self.name))
                      ]
        
 
    def call(self, inputs):
        cs = [self.layers[0](inputs), self.layers[1](inputs)]
        for i in range(3):
            for j in range(2):
                cs[j] = self.layers[2 + i](cs[j])
        c_weights = self.layers[6](self.layers[5](cs))
        # c_weighted = tf.math.multiply(inputs, c_weights[:,np.newaxis,np.newaxis,:])
        c_weighted = self.layers[7]([inputs, c_weights[:,np.newaxis,np.newaxis,:]])
        s_coeffs = self.layers[8]([tf.keras.backend.mean(c_weighted, (-1,), keepdims=True), tf.keras.backend.max(c_weighted, (-1,), keepdims=True)])
        for i in range(2):
            s_coeffs = self.layers[9 + i](s_coeffs)
        s_weights = self.layers[11](s_coeffs)
        return self.layers[12]([c_weighted, s_weights])
        
