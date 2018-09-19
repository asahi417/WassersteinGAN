"""
about batching --> https://stackoverflow.com/questions/44331612/how-to-set-a-number-for-epoch-in-tf-python-io-tf-record-iterator
about buffer size of shuffle --> https://github.com/tensorflow/tensorflow/issues/14857

pixel of each image is in range of [0 225]
"""

import os
import argparse
import tensorflow as tf
from PIL import Image
import numpy as np

PATH_TFRECORD = os.getenv('PATH_TFRECORD', './datasets/tfrecords')
OUTPUT = os.getenv('OUTPUT', './scripts/check_tfrecords')

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT, exist_ok=True)


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-n', '--num', help='number.', default=10, type=int, **share_param)
    parser.add_argument('-c', '--crop', help='number.', default=128, type=int, **share_param)
    parser.add_argument('-r', '--resize', help='number.', default=64, type=int, **share_param)
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    return parser.parse_args()


class TestTFRecord:

    def __init__(self,
                 batch_size,
                 image_shape,
                 n_thread=1):

        self.__n_thread = n_thread
        self.__mini_batch = batch_size
        self.__image_shape = image_shape

        self.__build_graph()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.session.run(tf.global_variables_initializer())

    def __record_parser(self, example_proto):
        features = dict(image=tf.FixedLenFeature([], tf.string, default_value=""))
        parsed_features = tf.parse_single_example(example_proto, features)
        feature_image = tf.decode_raw(parsed_features["image"], tf.uint8)
        feature_image = tf.cast(feature_image, tf.float32)
        image = tf.reshape(feature_image, self.__image_shape)
        return image

    def __build_graph(self):

        ############
        # TFRecord #
        ############

        # load tfrecord instance
        self.tfrecord_name = tf.placeholder(tf.string, name='tfrecord_dataset_name')
        data_set_api = tf.data.TFRecordDataset(self.tfrecord_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(self.__record_parser, self.__n_thread)
        # set batch size
        # data_set_api = data_set_api.shuffle(buffer_size=10000, seed=0)
        data_set_api = data_set_api.batch(self.__mini_batch)
        # make iterator
        iterator = tf.data.Iterator.from_structure(data_set_api.output_types, data_set_api.output_shapes)
        # get next input
        input_image = iterator.get_next()

        # initialize iterator
        self.data_iterator = iterator.make_initializer(data_set_api)
        input_image = input_image * 2 / 255 - 1
        self.input_image = tf.cast(input_image, tf.float32)


if __name__ == '__main__':
    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))
    recorder = TestTFRecord(image_shape=[args.resize, args.resize, 3], batch_size=1)
    recorder.session.run(recorder.data_iterator,
                         feed_dict={recorder.tfrecord_name: '%s/%s-c%i-r%i.tfrecord'
                                                            % (PATH_TFRECORD, args.data, args.crop, args.resize)})
    for e in range(args.num):
        img = recorder.session.run(recorder.input_image)
        img = img[0]
        img = (img + 1) / 2
        img = np.rint(img * 255).astype('uint8')
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img.save('%s/%s-c%i-r%i-%i.png' % (OUTPUT, args.data, args.crop, args.resize, e))
