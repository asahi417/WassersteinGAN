import wgan
import toml
import os

MODEL = os.getenv('MODEL', 'wgan')
EPOCH = int(os.getenv('EPOCH', '60'))
DATASET = os.getenv('DATASET', 'celeba')

TFRECORD = os.getenv('TFRECORD', './datasets/tfrecords/%s.tfrecord' % DATASET)
CHECKPOINT = os.getenv('CHECKPOINT', './checkpoint/%s-%s' % (MODEL, DATASET))
HYPERPARAMETER = os.getenv('HYPERPARAMETER', './bin/hyperparameter/%s-%s.toml' % (MODEL, DATASET))


if __name__ == '__main__':
    param = toml.load(open(HYPERPARAMETER))
    path_ckpt, _ = wgan.checkpoint_version(CHECKPOINT, param)

    model = wgan.WassersteinGAN(checkpoint_dir=path_ckpt, **param)
    model.train(epoch=EPOCH,
                path_to_tfrecord=TFRECORD,
                progress_interval=1,
                output_generated_image=True
                )
