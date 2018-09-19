import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops

# for 64 x 64 image
CNN_CHANNEL = [1024, 512, 256, 128, 3]
CNN_WIDTH = [4, 8, 16, 32, 64]

# for 128 x 128 image
# CNN_CHANNEL = [1024, 512, 256, 128, 64, 3]
# CNN_WIDTH = [4, 8, 16, 32, 64, 128]


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
        if self.generator_mode == 'cnn':
            self.generator = self.__generator_cnn
        else:
            raise ValueError('unknown generator mode `%s`' % critic_mode)

    def __critic_cnn(self,
                     inputs,
                     is_training=None,
                     batch_norm: bool = True,
                     batch_norm_decay: float = 0.999,
                     leaky_relu_alpha: float = 0.2,
                     reuse: bool = None,
                     scope: str = None
                     ):
        """DCGAN discriminator:
        - Batch norm for all layer except first layer1.
        - WGAN don't use last output sigomid activation."""

        def leaky_relu(x):
            return tf.maximum(tf.minimum(0.0, leaky_relu_alpha * x), x)

        with tf.variable_scope(scope or "dcgan_critic", reuse=reuse):

            inv_cnn_channel = CNN_CHANNEL[::-1]
            for i in range(len(inv_cnn_channel)):
                next_channel = inv_cnn_channel[i + 1] if i != len(inv_cnn_channel) - 1 else 1
                stride = [2, 2] if i != len(inv_cnn_channel) - 1 else [4, 4]
                inputs = self.convolution(
                    inputs,
                    weight_shape=[4, 4, inv_cnn_channel[i], next_channel],
                    stride=stride,
                    scope='conv_%i' % i,
                    padding='SAME'
                )
                if i != 0:  # no batch norm for input layer
                    if batch_norm:
                        inputs = self.bn(inputs, is_training=is_training, batch_norm_decay=batch_norm_decay)
                    inputs = leaky_relu(inputs)

            # flatten and get logit
            logit = tf.reshape(inputs, [self.dynamic_batch_size(inputs), 1])
        return logit

    def __generator_cnn(self,
                        inputs,
                        is_training=None,
                        batch_norm: bool = True,
                        batch_norm_decay: float = 0.999,
                        activation: str = 'relu',
                        ):
        """ DCGAN Generator: Batch norm for all layer except last layer.
        - channel ans filter width of transposed CNN is fixed as it produce 64 x 64 x 3 image
        - each elements are activated by tanh (so in range of -1 to 1)

        :param inputs: input tensor (batch, latent dim)
        :param is_training:
        :param batch_norm:
        :param batch_norm_decay:
        :param activation:
        :return:
        """

        activation_fn = self.check_activation(activation)
        tmp_ch = self.check_input_dimension(inputs, dim=2)
        batch_size = self.dynamic_batch_size(inputs)

        layer = tf.reshape(inputs, [batch_size, 1, 1, tmp_ch])
        for i, (ch, wid) in enumerate(zip(CNN_CHANNEL, CNN_WIDTH)):
            layer = self.convolution_trans(
                layer,
                weight_shape=[4, 4, ch, tmp_ch],
                output_shape=[batch_size, wid, wid, ch],
                stride=[4, 4] if i == 0 else [2, 2],
                scope='conv_%i' % i,
                padding='SAME',
                bias=False
            )
            tmp_ch = ch
            # activation: except last layer, which uses `tanh`, every layer uses preset activation
            if i == len(CNN_CHANNEL) - 1:
                layer = tf.tanh(layer)
            else:
                if batch_norm:
                    layer = self.bn(layer, is_training=is_training, batch_norm_decay=batch_norm_decay)
                layer = activation_fn(layer)

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
            raise ValueError('unknown activation `%s`' % activation_name)

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

    @staticmethod
    def bn(input_layer, is_training, batch_norm_decay):
            if is_training is None:
                raise ValueError('Specify train phase by `is_training`')
            return tf.contrib.layers.batch_norm(input_layer,
                                                decay=batch_norm_decay,
                                                is_training=is_training,
                                                updates_collections=None,
                                                scale=True)

