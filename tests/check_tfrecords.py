"""
about batching --> https://stackoverflow.com/questions/44331612/how-to-set-a-number-for-epoch-in-tf-python-io-tf-record-iterator
about buffer size of shuffle --> https://github.com/tensorflow/tensorflow/issues/14857
"""

import tensorflow as tf
from PIL import Image


class TestTFRecord:

    def __init__(self,
                 n_thread,
                 batch_size,
                 tfrecord_path):

        self.__n_thread = n_thread
        self.__mini_batch = batch_size

        self.__build_graph()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.session.run(tf.global_variables_initializer())

        self.session.run(self.data_iterator, feed_dict={self.tfrecord_name: tfrecord_path})

    def __build_graph(self):

        ############
        # TFRecord #
        ############
        def record_parser(example_proto):
            features = dict(
                # shape=tf.FixedLenFeature([], tf.int64, default_value=0),
                image=tf.FixedLenFeature([], tf.string, default_value="")
            )

            parsed_features = tf.parse_single_example(example_proto, features)
            feature_image = tf.decode_raw(parsed_features["image"], tf.uint8)
            feature_image = tf.cast(feature_image, tf.float32)
            # feature_shape = tf.cast(parsed_features["shape"], tf.int32)
            # print(feature_shape.get_shape())
            image = tf.reshape(feature_image, [128, 128, 3])
            return image

        # load tfrecord instance
        self.tfrecord_name = tf.placeholder(tf.string, name='tfrecord_dataset_name')
        # self.shuffle_data = tf.placeholder_with_default(False, [])
        data_set_api = tf.data.TFRecordDataset(self.tfrecord_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(record_parser, self.__n_thread)
        # set batch size
        data_set_api = data_set_api.shuffle(buffer_size=1000, seed=0)
        data_set_api = data_set_api.batch(self.__mini_batch)
        # make iterator
        iterator = tf.contrib.data.Iterator.from_structure(data_set_api.output_types, data_set_api.output_shapes)
        # get next input
        self.input_image = iterator.get_next()
        # initialize iterator
        self.data_iterator = iterator.make_initializer(data_set_api)

        ##############
        # main graph #
        ##############
        # place holder
        self.input_shape = data_set_api.output_shapes.as_list()[1:]
        self.input_image_pl = tf.placeholder_with_default(self.input_image, [None] + self.input_shape, name="input")
        self.output = tf.multiply(self.input_image_pl, 1)


if __name__ == '__main__':
    recorder = TestTFRecord(n_thread=4, batch_size=10, tfrecord_path='./datasets/tfrecords/celeba-128.tfrecord')
    for e in range(2):
        n = 0
        while True:
            try:
                img_1, img_2 = recorder.session.run([recorder.input_image, recorder.input_image_pl])
                if n <= 2:
                    img = Image.fromarray(img_1[0].astype('uint8'), 'RGB')
                    img.save('./tests/celeba_example/tfrecord_%i_%i.png' % (e, n))
                    img = Image.fromarray(img_2[0].astype('uint8'), 'RGB')
                    img.save('./tests/celeba_example/tfrecord_pl_%i_%i.png' % (e, n))
                    img = recorder.session.run(recorder.input_image_pl, {recorder.input_image_pl: [img_1[1]]})
                    img = Image.fromarray(img[0].astype('uint8'), 'RGB')
                    img.save('./tests/celeba_example/tfrecord_pl_recover_%i_%i.png' % (e, n))
                n += 1
            except tf.errors.OutOfRangeError:
                print('errorrrrr')
                break
        print(n)

