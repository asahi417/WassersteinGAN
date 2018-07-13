import wgan
import os

PATH_TFRECORD = os.getenv('PATH_TFRECORD', './datasets/tfrecords')
DATASET = os.getenv('DATASET', 'celeba')
# DOWNSCALE = os.getenv('DOWNSCALE', '0')
PATH_DATA = dict(celeba='./datasets/celeba/img/img_align_celeba',
                 lsun='./datasets/LSUN/bedroom_train_lmdb/data.mdb')


if __name__ == '__main__':
    recorder = wgan.TFRecorder(dataset_name=DATASET,
                               path_to_dataset=PATH_DATA,
                               tfrecord_dir=PATH_TFRECORD)
    recorder.create()
    # recorder.create(validation_split=0.2)
