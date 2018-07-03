import wgan

PATH_CELEBA = './datasets/celeba/img/img_align_celeba'
PATH_TFRECORD = './datasets/tfrecords'

if __name__ == '__main__':
    recorder = wgan.TFRecorder(dataset_name='celeba',
                               path_to_dataset=PATH_CELEBA,
                               tfrecord_dir=PATH_TFRECORD)
    recorder.create(validation_split=0.2)
