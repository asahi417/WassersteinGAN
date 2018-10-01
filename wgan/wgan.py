import tensorflow as tf
import numpy as np
import os
from PIL import Image
from .util import create_log
from .base_model import BaseModel
from .dataset_tool import tfrecord_parser


class WassersteinGAN:

    def __init__(self,
                 n_critic: int,
                 checkpoint_dir: str,
                 config: dict,
                 learning_rate: float = None,
                 config_critic: dict = None,
                 config_generator: dict = None,
                 gradient_penalty: float = None,
                 gradient_clip: float = None,
                 batch: int = 10,
                 optimizer: str = 'sgd',
                 debug: bool = True,
                 n_thread: int = 4,
                 down_scale: int = None,
                 initializer: str = 'variance_scaling',
                 overdose: bool=False):
        """
        :param config:
            n_z=128
            image_shape=[128, 128, 3]
        :param config_critic:
            mode='cnn'
            parameter=dict(batch_norm=True, batch_norm_decay=0.999)
        :param config_generator:
            mode='cnn'
            parameter=dict(batch_norm=True, batch_norm_decay=0.999)
        :param debug:
        """

        # get generator and critic
        self.__checkpoint_dir = checkpoint_dir
        self.__checkpoint = '%s/model.ckpt' % checkpoint_dir
        self.__overdose = overdose

        self.__config_critic = config_critic['parameter']
        self.__config_generator = config_generator['parameter']
        self.__base_model = BaseModel(critic_mode=config_critic['mode'], generator_mode=config_generator['mode'])

        # hyper parameters
        self.__ini_learning_rate = learning_rate
        self.__n_critic = n_critic
        self.__config = config
        self.__clip = gradient_clip
        self.__gp = gradient_penalty
        self.__batch = batch
        self.__down_scale = down_scale
        self.__optimizer = optimizer
        self.__logger = create_log() if debug else None

        self.__n_thread = n_thread
        self.__initializer = initializer

        self.__log('BUILD WassersteinGAN GRAPH: generator (%s), critic (%s)'
                   % (config_generator['mode'], config_critic['mode']))
        # self.__log('parameter: clip(%0.7f) batch (%i) opt (%s)' % (gradient_clip, batch, optimizer))
        self.__build_graph()
        self.__summary = tf.summary.merge_all()
        self.__session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.__writer = tf.summary.FileWriter('%s/summary' % self.__checkpoint_dir, self.__session.graph)

        # Load model
        if os.path.exists('%s.meta' % self.__checkpoint):
            self.__log('load variable from %s' % self.__checkpoint)
            self.__saver.restore(self.__session, self.__checkpoint)
            self.__warm_start = True
        else:
            os.makedirs(self.__checkpoint_dir, exist_ok=True)
            self.__session.run(tf.global_variables_initializer())
            self.__warm_start = False

    def __build_graph(self):
        """ Create Network, Define Loss Function and Optimizer """

        # initializer
        if self.__initializer == 'variance_scaling':
            initializer = tf.contrib.layers.variance_scaling_initializer()
        elif self.__initializer == 'truncated_normal':
            initializer = tf.initializers.truncated_normal(stddev=0.02)
        else:
            raise ValueError('unknown initializer: %s' % self.__initializer)

        # load tfrecord instance
        self.__tfrecord_name = tf.placeholder(tf.string, name='tfrecord_dataset_name')
        data_set_api = tf.data.TFRecordDataset(self.__tfrecord_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(tfrecord_parser(self.__config['image_shape']), self.__n_thread)
        # set batch size
        data_set_api = data_set_api.shuffle(buffer_size=50000)
        data_set_api = data_set_api.batch(self.__batch)
        # make iterator
        iterator = tf.data.Iterator.from_structure(data_set_api.output_types, data_set_api.output_shapes)
        # get next input
        tf_record_input = iterator.get_next()
        # initialize iterator
        self.__data_iterator = iterator.make_initializer(data_set_api)

        ##############
        # main graph #
        ##############
        # place holder
        self.__input_image = tf.placeholder_with_default(
            tf_record_input, [None] + self.__config['image_shape'], name="input")

        self.__learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # get random variable
        dynamic_batch = self.__base_model.dynamic_batch_size(self.__input_image)
        self.batch_generator = tf.placeholder_with_default(dynamic_batch, shape=[], name='batch')
        random_samples = tf.random_normal((self.batch_generator, self.__config["n_z"]),
                                          mean=0, stddev=1, dtype=tf.float32)
        self.random_samples = tf.placeholder_with_default(
            random_samples, shape=[None, self.__config["n_z"]], name='random_samples')

        # make pixel to be in [-1, 1]
        input_image = tf.cast(self.__input_image, tf.float32)
        input_image = input_image * 2 / 255 - 1

        with tf.variable_scope("generator", initializer=initializer):
            self.__generated_image = self.__base_model.generator(
                self.random_samples,
                is_training=True,
                **self.__config_generator)
            self.__generated_image_dev = self.__base_model.generator(
                self.random_samples,
                is_training=False,
                reuse=True,
                **self.__config_generator)

        with tf.variable_scope("critic", initializer=initializer):
            logit_input = self.__base_model.critic(input_image, **self.__config_critic)
            logit_random = self.__base_model.critic(self.__generated_image, reuse=True, **self.__config_critic)

            # Gradient Penalty for critic loss
            if self.__gp is not None:
                epsilon = tf.random_uniform([], 0.0, 1.0)
                penalty_target = self.__generated_image * (1 - epsilon) - input_image * epsilon

                logit_interpolate = self.__base_model.critic(penalty_target, reuse=True, **self.__config_critic)
                gradients = tf.gradients(logit_interpolate, penalty_target)[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = self.__gp * tf.reduce_mean((slopes - 1.) ** 2)
            else:
                gradient_penalty = 0

        ################
        # optimization #
        ################

        # trainable variable
        var_generator = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        var_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # loss
        self.__loss_critic = - tf.reduce_mean(logit_input - logit_random) + gradient_penalty
        self.__loss_generator = - tf.reduce_mean(logit_random)

        # optimizer
        if self.__optimizer == 'sgd':
            optimizer_critic = tf.train.GradientDescentOptimizer(self.__learning_rate)
            optimizer_generator = tf.train.GradientDescentOptimizer(self.__learning_rate)
        elif self.__optimizer == 'adam':
            optimizer_critic = tf.train.AdamOptimizer(self.__learning_rate, beta1=0.5)
            optimizer_generator = tf.train.AdamOptimizer(self.__learning_rate, beta1=0.5)
        elif self.__optimizer == 'rmsprop':
            optimizer_critic = tf.train.RMSPropOptimizer(self.__learning_rate)
            optimizer_generator = tf.train.RMSPropOptimizer(self.__learning_rate)
        else:
            raise ValueError('unknown optimizer !!')

        # train operation
        self.__train_op_generator = optimizer_generator.minimize(self.__loss_generator, var_list=var_generator)

        if self.__clip is not None:
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.__loss_critic, var_critic), self.__clip)
            self.__train_op_critic = optimizer_critic.apply_gradients(zip(grads, var_critic))
        else:
            self.__train_op_critic = optimizer_critic.minimize(self.__loss_critic, var_list=var_critic)

        ###########
        # logging #
        ###########

        self.__log('variable critic')
        n_var = 0
        for var in var_critic:
            sh = var.get_shape().as_list()
            self.__log('%s: %s' % (var.name, str(sh)))
            n_var += np.prod(sh)
            # write for tensorboard visualization
            # variable_summaries(var, var.name.split(':')[0].replace('/', '-'))

        self.__log('variable generator')
        for var in var_generator:
            sh = var.get_shape().as_list()
            self.__log('%s: %s' % (var.name, str(sh)))
            n_var += np.prod(sh)
            # write for tensorboard visualization
            # variable_summaries(var, var.name.split(':')[0].replace('/', '-'))

        self.__log('total variables: %i' % n_var)

        # saver
        self.__saver = tf.train.Saver()

    def train(self,
              epoch: int,
              path_to_tfrecord: str,
              progress_interval: int = None,
              output_generated_image: bool = False):

        self.__logger = create_log('%s/log' % self.__checkpoint_dir)
        self.__log('checkpoint (%s), epoch (%i), learning rate (%0.7f), n critic (%i)'
                   % (self.__checkpoint_dir, epoch, self.__ini_learning_rate, self.__n_critic))

        if self.__warm_start:
            meta = np.load('%s/meta.npz' % self.__checkpoint_dir)
            ini_epoch = meta['epoch']
            loss = meta['loss'].tolist()
            learning_rate = meta['learning_rate']
        else:
            ini_epoch = 0
            loss = []
            if self.__learning_rate is None:
                raise ValueError('provide learning rate !')
            learning_rate = self.__ini_learning_rate

        epoch += ini_epoch

        feed_critic = {self.__learning_rate: learning_rate}
        feed_generator = {self.__learning_rate: learning_rate, self.batch_generator: self.__batch}

        for e in range(ini_epoch, epoch):
            # initialize tfrecorder: initialize each epoch to shuffle data
            self.__session.run(self.__data_iterator, feed_dict={self.__tfrecord_name: [path_to_tfrecord]})
            loss_generator = []
            loss_critic = []
            n = 0
            while True:
                n += 1
                try:
                    # train critic
                    __n_critic = self.__n_critic

                    if self.__overdose:
                        # get the discriminator properly trained at the start
                        # https://github.com/cameronfabbri/Wasserstein-GAN-Tensorflow/blob/master/train.py#L110
                        if n < 25 and e == 0:
                            __n_critic = 100
                        elif n % 500 == 0:
                            __n_critic = 100

                    for _ in range(__n_critic):
                        val = [self.__train_op_critic, self.__loss_critic]
                        _, tmp_loss = self.__session.run(val, feed_dict=feed_critic)
                        loss_critic.append(tmp_loss)

                    # train generator
                    val = [self.__train_op_generator, self.__loss_generator, self.__generated_image_dev]
                    _, loss_gen, gen_img = self.__session.run(val, feed_dict=feed_generator)
                    loss_generator.append(loss_gen)

                    # print progress in epoch
                    if progress_interval is not None and n % progress_interval == 0:
                        if np.isnan(np.average(loss_generator)) and np.isnan(np.average(loss_critic)):
                            print()
                            raise ValueError('loss for generator and critic are nan')
                        if np.isnan(np.average(loss_critic)):
                            print()
                            raise ValueError('loss for critic is nan')
                        if np.isnan(np.average(loss_generator)):
                            print()
                            raise ValueError('loss for generator is nan')

                        print('epoch %i-%i: [generator: %0.3f, critics: %0.3f]\r'
                              % (e, n, np.average(loss_generator), np.average(loss_critic)), end='', flush=True)

                except tf.errors.OutOfRangeError:
                    print()
                    loss_generator, loss_critic = np.average(loss_generator), np.average(loss_critic)
                    self.__log('epoch %i: loss generator (%0.3f), loss critics (%0.3f)'
                               % (e, loss_generator, loss_critic))
                    loss.append([loss_generator, loss_critic])

                    if output_generated_image:
                        # output generated images
                        gen_img = (gen_img + 1) / 2
                        gen_img = np.rint(gen_img * 255).astype('uint8')
                        for _i in range(10):
                            Image.fromarray(gen_img[_i], 'RGB').save(
                                '%s/gen_img_%i-%i.png' % (self.__checkpoint_dir, e, _i))
                    break

        self.__saver.save(self.__session, self.__checkpoint)
        np.savez('%s/meta.npz' % self.__checkpoint_dir,
                 learning_rate=learning_rate,
                 loss=loss,
                 epoch=e+1)

    def generate_image(self, random_variable=None):
        if random_variable is None:
            random_variable = np.random.randn(self.__batch, self.__config["n_z"])
        result = self.__session.run(self.__generated_image,
                                    feed_dict={self.random_samples: random_variable})
        # print(result.shape, np.max(result), np.min(result), np.mean(result))
        result = (result + 1) / 2
        return [np.rint(_r * 255).astype('uint8') for _r in result]

    def __log(self, statement):
        if self.__logger is not None:
            self.__logger.info(statement)

    @property
    def input_image_shape(self):
        return self.__config['image_shape']



