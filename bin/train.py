import wgan
import toml
import os

HYPERPARAMETER = os.getenv('HYPERPARAMETER', './bin/hyperparameter')
IMAGE_SIZE = os.getenv('IMAGE_SIZE', '128')
MODEL = os.getenv('MODEL', 'wgan')
LEARNING_RATE = os.getenv('LEARNING_RATE', '0.000005')
CHECKPOINT = os.getenv('CHECKPOINT', './checkpoint')
EPOCH = os.getenv('EPOCH', '30')
TFRECORD = os.getenv('TFRECORD', './datasets/tfrecords')
DATASET = os.getenv('DATASET', 'celeba')
N_CRITIC = os.getenv('N_CRITIC', '1')


if __name__ == '__main__':
    param = toml.load(open('%s/%s-%s.toml' % (HYPERPARAMETER, MODEL, IMAGE_SIZE)))
    # print(param)
    model = wgan.WassersteinGAN(**param)
    model.train(checkpoint='%s/%s-%s-%s' % (CHECKPOINT, MODEL, DATASET, IMAGE_SIZE),
                epoch=int(EPOCH),
                path_to_tfrecord='%s/%s-%s.tfrecord' % (TFRECORD, DATASET, IMAGE_SIZE),
                n_critic=int(N_CRITIC),
                learning_rate=float(LEARNING_RATE),
                progress_interval=1,
                output_generated_image=True)
