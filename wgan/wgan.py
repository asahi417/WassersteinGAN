import tensorflow as tf
import numpy as np
import os
from .util import create_log, variable_summaries
from .base_model import BaseModel
from PIL import Image


class WassersteinGAN:

    def __init__(self,
                 checkpoint_dir: str,
                 n_critic: int,
                 config: dict,
                 learning_rate: float = None,
                 config_critic: dict = None,
                 config_generator: dict = None,
                 gradient_clip: float = 1,
                 batch: int = 10,
                 optimizer: str = 'sgd',
                 # load_model: str = None,
                 debug: bool = True,
                 n_thread: int = 4,
                 down_scale: int = None):
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
        :param load_model:
        :param debug:
        """

        # get generator and critic
        self.__checkpoint_dir = checkpoint_dir
        self.__checkpoint = '%s/model.ckpt' % checkpoint_dir

        self.__config_critic = config_critic['parameter']
        self.__config_generator = config_generator['parameter']
        self.__base_model = BaseModel(critic_mode=config_critic['mode'], generator_mode=config_generator['mode'])

        # hyper parameters
        self.__learning_rate = learning_rate
        self.__n_critic = n_critic
        self.__config = config
        self.__clip = gradient_clip
        self.__batch = batch
        self.__down_scale = down_scale
        self.__optimizer = optimizer
        self.__logger = create_log() if debug else None

        self.__n_thread = n_thread

        self.__log('BUILD WassersteinGAN GRAPH: generator (%s), critic (%s)'
                   % (config_generator['mode'], config_critic['mode']))
        self.__log('parameter: clip(%0.7f) batch (%i) opt (%s)' % (gradient_clip, batch, optimizer))
        self.__build_graph()
        self.__summary = tf.summary.merge_all()
        self.__writer_critic = tf.summary.FileWriter('%s/summary_critic' % self.__checkpoint_dir, self.__session.graph)
        self.__writer_generator = tf.summary.FileWriter('%s/summary_generator' % self.__checkpoint_dir, self.__session.graph)

        self.__session = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        # Load model
        if os.path.exists('%s.meta' % self.__checkpoint):
            self.__log('load variable from %s' % self.__checkpoint)
            self.__saver.restore(self.__session, self.__checkpoint)
            self.__warm_start = True
        else:
            os.makedirs(self.__checkpoint_dir, exist_ok=True)
            self.__session.run(tf.global_variables_initializer())
            self.__warm_start = False

    def __record_parser(self, example_proto):
        features = dict(image=tf.FixedLenFeature([], tf.string, default_value=""))
        parsed_features = tf.parse_single_example(example_proto, features)
        feature_image = tf.decode_raw(parsed_features["image"], tf.uint8)
        feature_image = tf.cast(feature_image, tf.float32)
        image = tf.reshape(feature_image, self.__config['image_shape'])
        return image

    def __build_graph(self):
        """ Create Network, Define Loss Function and Optimizer """

        # initializer
        initializer = tf.contrib.layers.variance_scaling_initializer(seed=0)
        # load tfrecord instance
        self.__tfrecord_name = tf.placeholder(tf.string, name='tfrecord_dataset_name')
        data_set_api = tf.data.TFRecordDataset(self.__tfrecord_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(self.__record_parser, self.__n_thread)
        # set batch size
        data_set_api = data_set_api.shuffle(buffer_size=10000, seed=0)
        data_set_api = data_set_api.batch(self.__batch)
        # make iterator
        iterator = tf.contrib.data.Iterator.from_structure(data_set_api.output_types, data_set_api.output_shapes)
        # get next input
        tf_record_input = iterator.get_next()
        # initialize iterator
        self.__data_iterator = iterator.make_initializer(data_set_api)

        # get random variable
        # random_samples = tf.random_normal((self.__batch, self.__config["n_z"]), mean=0, stddev=1, dtype=tf.float32)
        random_samples = tf.random_uniform((self.__batch, self.__config["n_z"]), minval=-1, maxval=1, dtype=tf.float32)

        ##############
        # main graph #
        ##############
        # place holder
        self.__input_image = tf.placeholder_with_default(
            tf_record_input, [None] + self.__config['image_shape'], name="input")

        self.__learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        tf.summary.scalar('meta_learning_rate', self.__learning_rate)

        self.__is_training = tf.placeholder_with_default(True, [])
        self.random_samples = tf.placeholder_with_default(
            random_samples, shape=[None, self.__config["n_z"]], name='random_samples')

        # make pixel to be in [0, 1]
        input_image = self.__input_image / 255

        # bilinear interpolation for down scale
        height, width, ch = self.__config['image_shape']
        assert height == width
        if self.__down_scale is not None:
            image_shape = np.rint(width / (2*self.__down_scale))
            size = tf.cast(tf.constant([image_shape, image_shape]), tf.int32)
            input_image = tf.image.resize_images(input_image, size)
        else:
            image_shape = width

        with tf.variable_scope("generator", initializer=initializer):
            self.__generated_image = self.__base_model.generator(self.random_samples,
                                                                 output_width=image_shape,
                                                                 output_channel=ch,
                                                                 is_training=self.__is_training,
                                                                 **self.__config_generator)

        with tf.variable_scope("critic", initializer=initializer):
            logit_input = self.__base_model.critic(input_image,
                                                   is_training=self.__is_training,
                                                   **self.__config_critic)
            # self.tmp = logit_input
            logit_random = self.__base_model.critic(self.__generated_image,
                                                    is_training=self.__is_training,
                                                    reuse=True,
                                                    **self.__config_critic)

        ################
        # optimization #
        ################

        # trainable variable
        var_generator = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        var_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # loss
        self.__loss_critic = tf.reduce_mean(logit_random) - tf.reduce_mean(logit_input)
        tf.summary.scalar('eval_loss_critic', self.__loss_critic)

        self.__loss_generator = -tf.reduce_mean(logit_random)
        tf.summary.scalar('eval_loss_generator', self.__loss_generator)

        # optimizer
        if self.__optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.__learning_rate)
        elif self.__optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.__learning_rate)
        else:
            raise ValueError('unknown optimizer !!')

        # train operation
        self.__train_op_generator = optimizer.minimize(self.__loss_generator, var_list=var_generator)

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.__loss_critic, var_critic), self.__clip)
        self.__train_op_critic = optimizer.apply_gradients(zip(grads, var_critic))

        ###########
        # logging #
        ###########

        n_var = 0
        for var in tf.trainable_variables():
            sh = var.get_shape().as_list()
            self.__log('%s: %s' % (var.name, str(sh)))
            n_var += np.prod(sh)
            # write for tensorboard visualization
            variable_summaries(var, var.name.split(':')[0].replace('/', '-'))

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
                   % (self.__checkpoint_dir, epoch, self.__learning_rate, self.__n_critic))

        if self.__warm_start:
            meta = np.load('%s/meta.npz' % self.__checkpoint_dir)
            ini_epoch = meta['epoch']
            i_summary_critic = meta['i_summary_critic']
            i_summary_generator = meta['i_summary_generator']
            loss = meta['loss'].tolist()
            learning_rate = meta['learning_rate']
        else:
            ini_epoch = 0
            i_summary_critic = 0
            i_summary_generator = 0
            loss = []
            if self.__learning_rate is None:
                raise ValueError('provide learning rate !')
            learning_rate = self.__learning_rate

        epoch += ini_epoch
        feed = {self.__learning_rate: learning_rate}

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
                    for _ in range(self.__n_critic):
                        val = [self.__train_op_critic, self.__loss_critic, self.__summary]
                        _, tmp_loss, summary = self.__session.run(val, feed_dict=feed)
                        # write tensorboard writer
                        self.__writer_critic.add_summary(summary, i_summary_critic)
                        i_summary_critic += 1
                        loss_critic.append(tmp_loss)

                    # train generator
                    val = [self.__train_op_generator, self.__loss_generator, self.__summary]
                    _, loss_gen, summary = self.__session.run(val, feed_dict=feed)
                    # write tensorboard writer
                    self.__writer_generator.add_summary(summary, i_summary_generator)
                    i_summary_generator += 1
                    loss_generator.append(loss_gen)

                    # print progress in epoch
                    if progress_interval is not None and n % progress_interval == 0:
                        print('epoch %i-%i: [generator: %0.3f, critics: %0.3f]\r'
                              % (e, n, np.average(loss_generator), np.average(loss_critic)), end='', flush=True)

                except tf.errors.OutOfRangeError:
                    print()
                    loss_generator, loss_critic = np.average(loss_generator), np.average(loss_critic)
                    self.__log('epoch %i: loss generator (%0.3f), loss critics (%0.3f)'
                               % (e, loss_generator, loss_critic))
                    loss.append([loss_generator, loss_critic])
                    if output_generated_image:
                        img = self.generate_image()
                        Image.fromarray(img, 'RGB').save('%s/generated_img_%i.png' % (self.__checkpoint_dir, e))
                    break

        self.__saver.save(self.__session, self.__checkpoint)
        np.savez('%s/meta.npz' % self.__checkpoint_dir,
                 learning_rate=self.__learning_rate,
                 i_summary_critic=i_summary_critic,
                 i_summary_generator=i_summary_generator,
                 loss=loss,
                 epoch=e+1,
                 n_critic=self.__n_critic)

    def generate_image(self, random_variable=None):
        if random_variable is None:
            random_variable = np.random.randn(1, self.__config["n_z"])
        result = self.__session.run([self.__generated_image], feed_dict={self.random_samples: random_variable})
        return np.rint(result[0][0] * 255).astype('uint8')

    def __log(self, statement):
        if self.__logger is not None:
            self.__logger.info(statement)

    @property
    def input_image_shape(self):
        return self.__config['image_shape']



