"""
check dataset image
"""

import os
import argparse
import numpy as np
from PIL import Image
from glob import glob

PATH_DATA = dict(celeba='./datasets/celeba/img/img_align_celeba', lsun='./datasets/lsun/image')
OUTPUT = os.getenv('OUTPUT', './scripts/check_image')

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT, exist_ok=True)


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-n', '--num', help='number.', default=10, type=int, **share_param)
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    return parser.parse_args()


def load_and_save(n, data_name):
    if data_name == 'celeba':
        image_filenames = sorted(glob('%s/*.png' % PATH_DATA[data_name]))
    else:
        image_filenames = sorted(glob('%s/*.jpg' % PATH_DATA[data_name]))

    for number, image_path in enumerate(image_filenames):
        # open as pillow instance
        image = Image.open(image_path)

        # pillow instance -> numpy array
        image = np.array(image)
        print(image.shape)

        # numpy array -> pillow instance
        image = Image.fromarray(image.astype('uint8'), 'RGB')

        # save it
        name = image_path.split('/')[-1]
        image.save('%s/%s-%s' % (OUTPUT, data_name, name))
        if number == n:
            break


if __name__ == '__main__':
    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))
    load_and_save(args.num, args.data)
