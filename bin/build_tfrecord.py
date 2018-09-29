""" script to build tfrecord """

import wgan
import os
import argparse

PATH_TFRECORD = os.getenv('PATH_TFRECORD', './datasets/tfrecords')
PATH_DATA = dict(
    celeba='./datasets/celeba/img/img_align_celeba',
    lsun='./datasets/lsun/train',
    celeba_v1='./datasets/celeba_v1/128_crop'
)


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-c', '--crop', help='number.', default=None, type=int, **share_param)
    parser.add_argument('-r', '--resize', help='number.', default=64, type=int, **share_param)
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))
    recorder = wgan.TFRecorder(dataset_name=args.data,
                               path_to_dataset=PATH_DATA[args.data],
                               tfrecord_dir=PATH_TFRECORD)
    recorder.create(
        crop_value=args.crop,
        resize_value=args.resize
    )
