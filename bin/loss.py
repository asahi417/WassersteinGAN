import wgan
import toml
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

    stat = np.load('%s/meta.npz' % path_ckpt)
    print(pd.DataFrame(stat['loss'], columns=['G', 'C']))
