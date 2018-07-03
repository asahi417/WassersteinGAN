import tensorflow as tf
import numpy as np
import os
from .util import create_log
from .base_model import BaseModel


class WassersteinGAN:

    def __init__(self,
                 config: dict,
                 config_critic: dict = None,
                 config_generator: dict = None,
                 gradient_clip: int = 10,
                 mini_batch: int = 10,
                 optimizer: str = 'sgd',
                 load_model: str = None,
                 debug: bool = True,
                 n_thread: int = 4):
        """
        :param config:
            n_z=128
        :param config_critic:
            mode='cnn'
            parameter=dict(batch_norm=True, batch_norm_decay=0.999)
        :param config_generator:
            mode='cnn'
            parameter=dict(batch_norm=True, batch_norm_decay=0.999)
        :param load_model:
        :param debug:
        """

        # get generator and critic
        self.__config_critic = config_critic
        self.__config_generator = config_generator
        self.__base_model = BaseModel(critic_mode=config_critic['mode'], generator_mode=config_generator['mode'])

        # hyper parameters
        self.__config = config
        self.__clip = gradient_clip
        self.__mini_batch = mini_batch
        self.__optimizer = optimizer
        self.__logger = create_log() if debug else None

        self.__n_thread = n_thread

        self.__log('BUILD WassersteinGAN GRAPH: generator (%s), critic (%s)' % (generator_mode, critic_mode))
        self.__build_graph()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        # Load model
        if load_model is not None and isinstance(load_model, str):
            self.saver.restore(self.session, load_model)
        else:
            self.session.run(tf.global_variables_initializer())

    def __build_graph(self):
        """ Create Network, Define Loss Function and Optimizer """

        # initializer
        initializer = tf.contrib.layers.variance_scaling_initializer(seed=0)

        ############
        # TFRecord #
        ############
        def record_parser(example_proto):
            features = dict(
                shape=tf.FixedLenFeature((), tf.int64, default_value=0),
                image=tf.FixedLenFeature((), tf.string, default_value="")
            )
            parsed_features = tf.parse_single_example(example_proto, features)
            return parsed_features["image"], parsed_features["shape"]

        def read_image(image, _shape):
            _shape = tf.cast(_shape, tf.int32)
            image = tf.decode_raw(image, tf.uint8)
            image = tf.cast(image, tf.float32)
            image = tf.reshape(image, _shape)
            return image

        # load tfrecord instance
        self.tfrecord_name = tf.placeholder(tf.string, name='tfrecord_dataset_name')
        data_set_api = tf.data.TFRecordDataset(self.tfrecord_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(record_parser, self.__n_thread)
        # formatting data
        data_set_api = data_set_api.map(read_image, self.__n_thread)
        # set batch size
        data_set_api = data_set_api.batch(self.__mini_batch)
        # repeating
        data_set_api = data_set_api.repeat(-1)
        # make iterator
        iterator = tf.contrib.data.Iterator.from_structure(data_set_api.output_types, data_set_api.output_shapes)
        # get next input
        input_image = iterator.get_next()
        # initialize iterator
        self.data_iterator = iterator.make_initializer(data_set_api)
        # get random variable
        random_samples = tf.random_normal((self.__mini_batch, self.__config["n_z"]), mean=0, stddev=1, dtype=tf.float32)

        ##############
        # main graph #
        ##############
        # place holder
        self.input_shape = input_image.get_shape().as_list()[1:]
        self.input_image = tf.placeholder_with_default(input_image, [None] + self.input_shape, name="input")
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.is_training = tf.placeholder_with_default(False, [])
        self.random_samples = tf.placeholder_with_default(random_samples,
                                                          shape=[None, self.__config["n_z"]],
                                                          name='random_samples')

        input_image = self.input_image / 255  # make pixel to be in [0, 1]

        with tf.variable_scope("generator", initializer=initializer):
            self.generated_image = self.__base_model.generator(self.random_samples,
                                                               output_width=64,
                                                               output_channel=3,
                                                               is_training=self.is_training,
                                                               **self.__config_generator)

        with tf.variable_scope("critic", initializer=initializer):
            logit_input = self.__base_model.critic(input_image,
                                                   is_training=self.is_training,
                                                   **self.__config_critic)
            logit_random = self.__base_model.critic(self.generated_image,
                                                    is_training=self.is_training,
                                                    reuse=True,
                                                    **self.__config_critic)

        ################
        # optimization #
        ################

        # trainable variable
        var_generator = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        var_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # loss
        self.loss_critic = tf.reduce_mean(logit_input) - tf.reduce_mean(logit_random)
        self.loss_generator = -tf.reduce_mean(logit_random)
        # self.loss_critic = tf.where(tf.is_nan(loss_critic), 0.0, loss_critic)
        # self.loss_generator = tf.where(tf.is_nan(loss_generator), 0.0, loss_generator)

        # optimizer
        if self.__optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.__optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError('unknown optimizer !!')

        # train operation
        self.train_op_generator = optimizer.minimize(self.loss_generator, var_list=var_generator)

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_critic, var_critic), self.__clip)
        self.train_op_critic = optimizer.apply_gradients(zip(grads, var_critic))

        ###########
        # logging #
        ###########

        self.__log('input shape: %s' % str(self.input_shape))
        self.n_var = 0
        for var in tf.trainable_variables():
            sh = var.get_shape().as_list()
            self.__log('%s: %s' % (var.name, str(sh)))
            self.n_var += np.prod(sh)

        self.__log('total variables: %i' % self.n_var)

        # saver
        self.saver = tf.train.Saver()

    def __log(self, statement):
        if self.__logger is not None:
            self.__logger.info(statement)

    @property
    def path_to_dataset(self):
        return self.tfrecord_name

    @property
    def image(self):
        return self.input_image

    @property
    def trainable_variables(self):
        return self.n_var

    @property
    def input_image_shape(self):
        return self.input_shape

    def train(self,
              checkpoint: str,
              learning_rate: float = None,
              checkpoint_warm_start: str = None,
              progress_interval: int = 10):

        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint, exist_ok=True)

        # train
        self.session.run(init_op,
                         feed_dict={filenames: [INPUT_TFRECORD_TRAIN]})


