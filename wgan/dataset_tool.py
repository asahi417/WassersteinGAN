"""
Data organizing function: highly influenced by following script.
https://github.com/tkarras/progressive_growing_of_gans/blob/master/dataset_tool.py
"""

import os
from glob import glob
from time import time
from PIL import Image
import numpy as np
import tensorflow as tf


def raise_error(condition, msg):
    if condition:
        raise ValueError(msg)


def shuffle_data(data, seed=0):
    np.random.seed(seed)
    np.random.shuffle(data)
    return data


class TFRecorder:
    """ Formatting data as TFrecord """

    def __init__(self,
                 dataset_name: str,
                 path_to_dataset: str,
                 tfrecord_dir: str,
                 print_progress: bool = True,
                 progress_interval: int = 10):

        raise_error(dataset_name not in ['celeba', 'mnist'], 'unknown data: %s' % dataset_name)
        self.dataset_name = dataset_name
        self.path_to_dataset = path_to_dataset
        self.path_to_save = '%s/%s' % (tfrecord_dir, dataset_name)

        if not os.path.exists(tfrecord_dir):
            os.makedirs(tfrecord_dir, exist_ok=True)
        self.print_progress = print_progress
        self.progress_interval = progress_interval

    def my_print(self, *args, **kwargs):
        if self.print_progress:
            print(*args, **kwargs)

    def create(self, *args, **kwargs):
        if self.dataset_name == 'celeba':
            return self.__celeba(*args, **kwargs)
        elif self.dataset_name == 'mnist':
            pass
        else:
            raise ValueError('unknown data: %s' % self.dataset_name)

    def __celeba(self,
                 center_x: int = 89,
                 center_y: int = 121,
                 shuffle_seed: int = 0,
                 down_scale: int = 0,
                 validation_split: float = None):

        original_shape = 128 / 2 ** down_scale
        self.my_print('shape: %i' % original_shape)

        image_files = glob('%s/*.png' % self.path_to_dataset)
        image_files = shuffle_data(image_files, seed=shuffle_seed)

        def write(image_filenames, name, mode):
            full_size = len(image_filenames)
            self.my_print('writing celeba as tfrecord (%s): size %i' % (mode, full_size))
            compress_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
            with tf.python_io.TFRecordWriter(name, options=compress_opt) as writer:
                time_stamp = time()
                time_stamp_start = time()
                for n, single_image_path in enumerate(image_filenames):
                    img = np.asarray(Image.open(single_image_path))

                    # cropping
                    img = img[center_y - 64: center_y + 64, center_x - 64: center_x + 64]
                    img = np.rint(img).clip(0, 255).astype(np.uint8)
                    # downscale
                    if down_scale != 0:
                        for _ in range(down_scale):
                            img = img[0::2, 0::2, :] + img[0::2, 1::2, :] + img[1::2, 0::2, :] + img[1::2, 1::2, :]
                        img = img * 0.25**down_scale
                        img = np.rint(img).clip(0, 255).astype(np.uint8)

                    if n % self.progress_interval == 0:

                        progress_perc = n / full_size * 100
                        cl_time = time() - time_stamp
                        whole_time = time() - time_stamp_start
                        time_per_sam = cl_time / self.progress_interval
                        self.my_print('%s: %d / %d (%0.1f %%), %0.4f sec/image (%0.1f sec) \r'
                                      % (mode, n, full_size, progress_perc, time_per_sam, whole_time),
                                      end='', flush=True)
                        time_stamp = time()

                    ex = tf.train.Example(
                        features=tf.train.Features(
                            feature=dict(
                                image=tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))
                            )))
                    writer.write(ex.SerializeToString())

        # apply full data
        write(image_files, '%s-%i.tfrecord' % (self.path_to_save, original_shape), mode='full')

        if validation_split is not None:
            raise_error(validation_split > 1, 'validation_split has to be in [0,1]')

            # apply train data
            write(image_files[int(np.rint(len(image_files) * validation_split)):],
                  '%s-%i-train.tfrecord' % (self.path_to_save, original_shape),
                  mode='train')

            # apply test data
            write(image_files[:int(np.rint(len(image_files) * validation_split))],
                  '%s-%i-test.tfrecord' % (self.path_to_save, original_shape),
                  mode='test')


# def create_mnist(tfrecord_dir, mnist_dir, shuffle_seed=0):
#     import gzip
#     with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
#         images = np.frombuffer(file.read(), np.uint8, offset=16)
#     with gzip.open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as file:
#         labels = np.frombuffer(file.read(), np.uint8, offset=8)
#     images = images.reshape(-1, 1, 28, 28)
#     images = np.pad(images, [(0, 0), (0, 0), (2, 2), (2, 2)], 'constant', constant_values=0)
#     assert images.shape == (60000, 1, 32, 32) and images.dtype == np.uint8
#     assert labels.shape == (60000,) and labels.dtype == np.uint8
#     assert np.min(images) == 0 and np.max(images) == 255
#     assert np.min(labels) == 0 and np.max(labels) == 9
#     onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
#     onehot[np.arange(labels.size), labels] = 1.0
#
#     with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
#         print('Loading MNIST from "%s"' % mnist_dir)
#         order = tfr.randomized_index(seed=shuffle_seed)
#         for idx in range(order.size):
#             tfr.add_image(images[order[idx]])
#         tfr.add_labels(onehot[order])


if __name__ == '__main__':
    path_celeba = './datasets/celeba'
    path_tfrecord = './datasets/tfrecords/celeba'
    create_celeba(path_tfrecord, path_celeba)
