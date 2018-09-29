""" script to generate image from trained model """

import wgan
import os
import argparse
# from PIL import Image
import numpy as np
import scipy.misc as misc


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
    parser.add_argument('-v', '--version', help='number.', default=None, type=int, **share_param)
    parser.add_argument('-c', '--crop', help='number.', default=None, type=int, **share_param)
    parser.add_argument('-r', '--resize', help='number.', default=64, type=int, **share_param)
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    parser.add_argument('--sub_version', help='sub version.', default=None, type=int, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))
    _tfrecord, _ckpt, _param = get_path(args.data, args.model, args.crop, args.resize)
    path_ckpt, _param = wgan.checkpoint_version(_ckpt, version=args.version, sub_version=args.sub_version)
    version = '%i' % args.version if args.sub_version is None else '%i.%i' % (args.version, args.sub_version)

    if args.model == 'wgan':
        model_instance = wgan.WassersteinGAN(checkpoint_dir=path_ckpt, **_param)
    elif args.model == 'dcgan':
        model_instance = wgan.DCGAN(checkpoint_dir=path_ckpt, **_param)
    else:
        raise ValueError('unknown model!')

    n = 16  # cols
    m = 4  # rows
    img_size = (64, 64, 3)

    canvas = 255 * np.ones((m * img_size[0] + (10 * m) + 10, n * img_size[1] + (10 * n) + 10, 3), dtype=np.uint8)

    start_x = 10
    start_y = 10

    x = 0
    y = 0

    images = model_instance.generate_image()

    for img in images:

        end_x = start_x + 64
        end_y = start_y + 64

        canvas[start_y:end_y, start_x:end_x, :] = img

        if x < n:
            start_x += 64 + 10
            x += 1
        if x == n:
            x = 0
            start_x = 10
            start_y = end_y + 10
            end_y = start_y + 64

    misc.imsave('./bin/img/generated_img/%s-%s-v%s.jpg' % (args.model, args.data, version), canvas)

    # Image.fromarray(img, 'RGB').save('./bin/img/generated_img/%s-%s-%i.png' % (args.model, args.data, n))
