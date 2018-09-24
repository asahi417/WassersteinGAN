import os
import argparse
import numpy as np
import pandas as pd


def get_path(data, model):
    tfrecord = os.getenv('TFRECORD', './datasets/tfrecords/%s.tfrecord' % data)
    ckpt = os.getenv('CHECKPOINT', './checkpoint/%s-%s' % (model, data))
    param = os.getenv('HYPERPARAMETER', './bin/hyperparameter/%s-%s.toml' % (model, data))
    return tfrecord, ckpt, param


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-m', '--model', help='Model.', required=True, type=str, **share_param)
    parser.add_argument('-v', '--version', help='Version.', required=True, type=int, **share_param)
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))
    _tfrecord, _ckpt, _ = get_path(args.data, args.model)

    stat = np.load('%s/v%i/meta.npz' % (_ckpt, args.version))
    print(pd.DataFrame(stat['loss'], columns=['G', 'C']))
