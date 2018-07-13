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
                 validation_split: float = None):

        self.my_print('shape: 128')

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
        write(image_files, '%s.tfrecord' % self.path_to_save, mode='full')

        if validation_split is not None:
            raise_error(validation_split > 1, 'validation_split has to be in [0,1]')

            # apply train data
            write(image_files[int(np.rint(len(image_files) * validation_split)):],
                  '%s-train.tfrecord' % self.path_to_save, mode='train')

            # apply test data
            write(image_files[:int(np.rint(len(image_files) * validation_split))],
                  '%s-test.tfrecord' % self.path_to_save, mode='test')

