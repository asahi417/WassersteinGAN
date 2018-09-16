import wgan
import os
import argparse

PATH_TFRECORD = os.getenv('PATH_TFRECORD', './datasets/tfrecords')
PATH_DATA = dict(celeba='./datasets/celeba/img/img_align_celeba',
                 lsun='./datasets/lsun/data_train')


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    # parser.add_argument('--split', help='split.', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))
    recorder = wgan.TFRecorder(dataset_name=args.data,
                               path_to_dataset=PATH_DATA[args.data],
                               tfrecord_dir=PATH_TFRECORD)
    recorder.create()
    # if args.split:
    #     recorder.create(validation_split=0.2)
    # else:
    #     recorder.create()
