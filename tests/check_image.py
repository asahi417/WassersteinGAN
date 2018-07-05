"""
CelebA
"""


import numpy as np
from PIL import Image
from glob import glob

PATH_CELEBA = './datasets/celeba'
OUTPUT = './tests/celeba_example'


def load_and_save(n=10):
    image_filenames = sorted(glob('%s/img/img_align_celeba/*.png' % PATH_CELEBA))
    for number, image_path in enumerate(image_filenames):
        # open as pillow instance
        image = Image.open(image_path)

        # pillow instance -> numpy array
        image = np.array(image)
        print(image.shape, type(image))

        # numpy array -> pillow instance
        image = Image.fromarray(image.astype('uint8'), 'RGB')

        # save it
        name = image_path.split('/')[-1]
        image.save('%s/created_%s' % (OUTPUT, name))
        if number == n:
            break


if __name__ == '__main__':
    load_and_save(10)
