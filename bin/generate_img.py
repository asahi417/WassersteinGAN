import wgan
import toml
import os
import argparse
from PIL import Image


def get_path(data, model, crop, resize):
    if crop is None:
        tfrecord = os.getenv('TFRECORD', './datasets/tfrecords/%s-r%i.tfrecord' % (data, resize))
        ckpt = os.getenv('CHECKPOINT', './checkpoint/%s-%s-r%i' % (model, data, resize))
    else:
        tfrecord = os.getenv('TFRECORD', './datasets/tfrecords/%s-c%i-r%i.tfrecord' % (data, crop, resize))
        ckpt = os.getenv('CHECKPOINT', './checkpoint/%s-%s-c%i-r%i' % (model, data, crop, resize))
    param = os.getenv('HYPERPARAMETER', './bin/hyperparameter/%s-%s.toml' % (model, data))
    return tfrecord, ckpt, param


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-m', '--model', help='Model.', required=True, type=str, **share_param)
    parser.add_argument('-e', '--epoch', help='Epoch.', required=True, type=int, **share_param)
    parser.add_argument('-v', '--version', help='number.', default=None, type=int, **share_param)
    parser.add_argument('-c', '--crop', help='number.', default=None, type=int, **share_param)
    parser.add_argument('-r', '--resize', help='number.', default=64, type=int, **share_param)
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))
    _tfrecord, _ckpt, _param = get_path(args.data, args.model, args.crop, args.resize)

    path_ckpt, _param = wgan.checkpoint_version(_ckpt, version=args.version)

    if args.model == 'wgan':
        model_instance = wgan.WassersteinGAN(checkpoint_dir=path_ckpt, **_param)
    elif args.model == 'dcgan':
        model_instance = wgan.DCGAN(checkpoint_dir=path_ckpt, **_param)
    else:
        raise ValueError('unknown model!')
    while True:
        _inp = input('q to break >>>')
        if _inp == 'q':
            break
        images = model_instance.generate_image()
        for n, img in enumerate(images):
            Image.fromarray(img, 'RGB').save('./bin/%s-%s-%i.png' % (args.model, args.data, n))
