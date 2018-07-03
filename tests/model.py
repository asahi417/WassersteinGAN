import wgan


if __name__ == '__main__':
    config = dict(input=[64, 64, 3], n_z=128)
    wgan.WassersteinGAN(config=config)
