# -*- coding: utf-8 -*-
# this script implements one-to-one connection convolutional layer discribed in Alotaibi2017A
# Improved gait recognition based on specialized deep convolutional neural network
# reference: convolutional.py in keras source code

from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.engine.base_layer import InputSpec
from keras.utils import conv_utils

# 121 means one-to-one connection :)
class Conv2D121(Layer):

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 rank=2,
                 padding='valid',
                 data_format=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
                 ):
        super(Conv2D121, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        # normalize_padding: 检查padding的值，只有['valid', 'same', 'causal']三个值合法
        self.padding = conv_utils.normalize_padding(padding)
        # data_format: 检查
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.use_bias = use_bias,
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        self.input_dim = input_dim
        if input_dim != self.filters:
            raise ValueError('Because nature of one-to-one connnection, '
                             'input dimension must be equal to filters number')
        kernel_shape = self.kernel_size + (1, 1)

        self.kernels = []

        for i in range(input_dim):
            self.kernels.append(self.add_weight(
                shape=kernel_shape,
                # initializer=self.kernel_initializer,
                initializer=self.kernel_initializer,
                name='kernel' + str(i),
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint
            ))

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.input_dim,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        if self.rank != 2:
            raise ValueError('currently this layer only support 2D data.')

        input_slices = []

        # now we need to slice the input_dim dimension input and do convolution
        for i in range(self.input_dim):
            slice = K.expand_dims(inputs[:, :, :, i], axis=3)
            input_slices.append(slice)

        output_slices = []

        for i in range(self.input_dim):
            slice = K.conv2d(
                input_slices[i],
                self.kernels[i],
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
            )
            output_slices.append(slice)

        output = K.concatenate(output_slices, axis=3)

        if (self.use_bias):
            output = K.bias_add(
                output,
                self.bias,
                data_format=self.data_format
            )

        return output

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)