import wgan
import os

PATH_DATA = os.getenv('PATH_DATA', './datasets/celeba/img/img_align_celeba')
PATH_TFRECORD = os.getenv('PATH_TFRECORD', './datasets/tfrecords')
DATASET = os.getenv('DATASET', 'celeba')
DOWNSCALE = os.getenv('DOWNSCALE', '0')

if __name__ == '__main__':
    recorder = wgan.TFRecorder(dataset_name=DATASET,
                               path_to_dataset=PATH_DATA,
                               tfrecord_dir=PATH_TFRECORD)
    recorder.create(down_scale=int(DOWNSCALE))
    # recorder.create(validation_split=0.2)
