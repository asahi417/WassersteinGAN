import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops


class BaseModel:
    """
    Base Model for GAN:
        - Critics (Discriminator): CNN (DCGAN)
        - Generator: MLP, CNN (DCGAN)
    """

    def __init__(self,
                 critic_mode: str = 'cnn',
                 generator_mode: str = 'cnn',
                 ):
        """
        :param critic_mode:
        :param generator_mode:
        """

        # choose critic mode
        self.critic_mode = critic_mode
        if self.critic_mode == 'cnn':
            self.critic = self.__critic_cnn
        else:
            raise ValueError('unknown critic mode `%s`' % critic_mode)

        # choose generator mode
        self.generator_mode = generator_mode
        if self.generator_mode == 'mlp':
            self.generator = self.__generator_mlp
        elif self.generator_mode == 'cnn':
            self.generator = self.__generator_cnn
        else:
            raise ValueError('unknown generator mode `%s`' % critic_mode)

    def __generator_mlp(self,
                        inputs,
                        output_shape: list,
                        hidden_unit: int = 512,
                        layer_size: int = 4,
                        activation: str = 'relu'):
        """ Fully-connected generator """

        activation_fn = self.check_activation(activation)
        unit_size = self.check_input_dimension(inputs, dim=2)
        layer = inputs

        for i in range(layer_size):
            if i == 0:
                layer = self.full_connected(layer, [unit_size, hidden_unit], scope='fc_%i' % (i+1))
                layer = activation_fn(layer)
            elif i == layer_size-1:
                layer = self.full_connected(layer, [hidden_unit, np.prod(output_shape)], scope='fc_%i' % (i+1))
                layer = tf.tanh(layer)
                layer = tf.reshape(layer, [-1] + output_shape)
            else:
                layer = self.full_connected(layer, [hidden_unit, hidden_unit], scope='fc_%i' % (i+1))
                layer = activation_fn(layer)
        return layer

    def __critic_cnn(self,
                     inputs,
                     is_training=None,
                     batch_norm: bool = True,
                     batch_norm_decay: float = 0.999,
                     channel: list = None,
                     filter_width: list = None,
                     stride: list = None,
                     leaky_relu_alpha: float = 0.2,
                     reuse: bool = None
                     ):
        """DCGAN discriminator:
        - Batch norm for all layer except first layer1.
        - WGAN don't use last output sigomid activation."""

        channel = [64, 128, 256, 512] if channel is None else channel

        filter_width = [5, 5, 5, 5] if filter_width is None else filter_width
        stride = [2, 2, 2, 2] if stride is None else stride
        
        def leaky_relu(x):
            return tf.maximum(tf.minimum(0.0, leaky_relu_alpha * x), x)

        def bn(input_layer):
            if batch_norm:
                if is_training is None:
                    raise ValueError('Specify train phase by `is_training`')
                return tf.contrib.layers.batch_norm(input_layer, decay=batch_norm_decay, is_training=is_training)
            else:
                return input_layer

        # add initial channel size
        input_channel = self.check_input_dimension(inputs, dim=4)
        channel = [input_channel] + channel

        layer = inputs
        for i in range(len(filter_width)):
            layer = self.convolution(
                layer,
                weight_shape=[filter_width[i], filter_width[i], channel[i], channel[i+1]],
                stride=[stride[i], stride[i]],
                scope='conv_%i' % i,
                reuse=reuse
            )
            if i != 0:  # no batch norm for input layer
                layer = bn(layer)
            layer = leaky_relu(layer)

        # flatten and get logit
        sh = layer.get_shape().as_list()
        flatten_size = np.prod(sh[1:])
        layer = tf.reshape(layer, [-1, flatten_size])
        logit = self.full_connected(layer,
                                    weight_shape=[flatten_size, 1],
                                    reuse=reuse)
        return logit

    def __generator_cnn(self,
                        inputs,
                        output_width: int,
                        output_channel: int,
                        is_training=None,
                        batch_norm: bool = True,
                        batch_norm_decay: float = 0.999,
                        activation: str = 'relu',
                        filter_width: list = None,
                        stride: list = None
                        ):
        """ DCGAN Generator: Batch norm for all layer except last layer.

        :param inputs: input tensor
        :param output_width: width length of output
        :param output_channel: channel size of output
        :param is_training:
        :param batch_norm:
        :param batch_norm_decay:
        :param activation:
        :param filter_width:
        :param stride:
        :return:
        """

        channel = [1024, 512, 256, 128, output_channel]

        filter_width = [5, 5, 5, 5] if filter_width is None else filter_width
        stride = [2, 2, 2, 2] if stride is None else stride

        def bn(input_layer):
            if batch_norm:
                if is_training is None:
                    raise ValueError('Specify train phase by `is_training`')
                return tf.contrib.layers.batch_norm(input_layer, decay=batch_norm_decay, is_training=is_training)
            else:
                return input_layer

        def next_output_shape(output_shape, fractional_stride):
            return int(np.ceil(output_shape / fractional_stride))

        activation_fn = self.check_activation(activation)
        unit_size = self.check_input_dimension(inputs, dim=2)
        batch_size = self.dynamic_batch_size(inputs)

        # calculate valid output shape for transposed convolution
        width = output_width
        output_width = [output_width]
        for _stride in stride:
            width = next_output_shape(width, _stride)
            output_width.append(width)
        output_width = output_width[::-1]

        layer = self.full_connected(inputs,
                                    weight_shape=[unit_size, output_width[0] * output_width[0] * channel[0]],
                                    bias=False,
                                    scope='fc')
        layer = tf.reshape(layer, [-1, output_width[0], output_width[0], channel[0]])
        layer = bn(layer)
        layer = activation_fn(layer)

        # print(layer.shape)

        for i in range(len(filter_width)):
            # convolution_trans
            layer = self.convolution_trans(
                layer,
                weight_shape=[filter_width[i], filter_width[i], channel[i+1], channel[i]],
                output_shape=[batch_size, output_width[i+1], output_width[i+1], channel[i+1]],
                stride=[stride[i], stride[i]],
                scope='conv_%i' % (i+1),
                bias=False
            )
            # activation: except last layer, which uses `tanh`, every layer uses preset activation
            if i == len(filter_width) - 1:
                layer = tf.tanh(layer)
            else:
                # batch normalization, except for output layer
                layer = bn(layer)
                layer = activation_fn(layer)

            # print(layer.shape)
        return layer

    @staticmethod
    def check_input_dimension(inputs, dim: int):
        """Check tensor dimension. If dimension is matched, return last dimension shape, else raise error """
        tensor_shape = inputs.get_shape().as_list()
        if len(tensor_shape) != dim:
            raise ValueError('Expect %i dimension tensor but get %s' % (dim, str(tensor_shape)))
        else:
            return tensor_shape[-1]

    @staticmethod
    def check_activation(activation_name):
        if activation_name == 'relu':
            return tf.nn.relu
        else:
            raise ValueError('unknown activation `%s`' % activation)

    @staticmethod
    def full_connected(x,
                       weight_shape,
                       scope=None,
                       bias=True,
                       reuse=None):
        """ fully connected layer
        - weight_shape: input size, output size
        - priority: batch norm (remove bias) > dropout and bias term
        """
        with tf.variable_scope(scope or "fully_connected", reuse=reuse):
            w = tf.get_variable("weight", shape=weight_shape, dtype=tf.float32)
            x = tf.matmul(x, w)
            if bias:
                b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
                return tf.add(x, b)
            else:
                return x

    @staticmethod
    def convolution(x,
                    weight_shape,
                    stride,
                    padding="SAME",
                    scope=None,
                    bias=True,
                    reuse=None):
        """2d convolution
         Parameter
        -------------------
        weight_shape: width, height, input channel, output channel
        stride (list): [stride for axis 1, stride for axis 2]
        """
        with tf.variable_scope(scope or "2d_convolution", reuse=reuse):
            w = tf.get_variable('weight', shape=weight_shape, dtype=tf.float32)
            x = tf.nn.conv2d(x, w, strides=[1, stride[0], stride[1], 1], padding=padding)
            if bias:
                b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
                return tf.add(x, b)
            else:
                return x

    @staticmethod
    def convolution_trans(x,
                          weight_shape,
                          output_shape,
                          stride,
                          padding="SAME",
                          scope=None,
                          bias=True,
                          reuse=None):
        """2d fractinally-strided convolution (transposed-convolution)

         Parameter
        --------------------
        weight_shape: width, height, output channel, input channel
        stride (list): [stride for axis 1, stride for axis 2]
        output_shape (list): [batch, width, height, output channel]
        """
        with tf.variable_scope(scope or "convolution_trans", reuse=reuse):
            w = tf.get_variable('weight', shape=weight_shape, dtype=tf.float32)
            x = tf.nn.conv2d_transpose(x,
                                       w,
                                       output_shape=output_shape,
                                       strides=[1, stride[0], stride[1], 1],
                                       padding=padding,
                                       data_format="NHWC")
            if bias:
                b = tf.get_variable("bias", initializer=[0.0] * weight_shape[2])
                return tf.add(x, b)
            else:
                return x

    @staticmethod
    def dynamic_batch_size(inputs):
        """ Dynamic batch size, which is able to use in a model without deterministic batch size.
        See https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn.py
        """
        while nest.is_sequence(inputs):
            inputs = inputs[0]
        return array_ops.shape(inputs)[0]