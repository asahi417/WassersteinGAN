import wgan
import toml
import os
import argparse
import numpy as np
from PIL import Image


def get_path(data, model):
    tfrecord = os.getenv('TFRECORD', './datasets/tfrecords/%s.tfrecord' % data)
    ckpt = os.getenv('CHECKPOINT', './checkpoint/%s-%s' % (model, data))
    param = os.getenv('HYPERPARAMETER', './bin/hyperparameter/%s-%s.toml' % (model, data))
    return tfrecord, ckpt, param


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-m', '--model', help='Model.', required=True, type=str, **share_param)
    parser.add_argument('-n', '--num', help='num to generate.', required=True, type=int, **share_param)
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))
    _tfrecord, _ckpt, _param = get_path(args.data, args.model)

    _param = toml.load(open(_param))
    path_ckpt, _ = wgan.checkpoint_version(_ckpt, _param)

    if args.model == 'wgan':
        model_instance = wgan.WassersteinGAN(checkpoint_dir=path_ckpt, **_param)
    elif args.model == 'dcgan':
        model_instance = wgan.DCGAN(checkpoint_dir=path_ckpt, **_param)
    else:
        raise ValueError('unknown model!')
    for _n in range(args.num):
        img = model_instance.generate_image()
        print(img)
        # print(img.shape, np.max(img), np.min(img))
        # img = (img + 1)/2 * 255
        Image.fromarray(img, 'RGB').save('./bin/%s-%s-%i.png' % (args.model, args.data, _n))
