# Wasserstein GAN 
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/WassersteinGAN/blob/master/LICENSE)

Tensorflow implementation of [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf) with [gradient penalty](https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf).
Properties are summalized as below

- Tested by [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 
- Data is encoded as TFRecord format

There already exist several implementations of WGAN by tensorflow, but
few implementations use TFRecord, which is modern data feeding format optimized as used in tensorflow graph.
In this repository, TFRecord is used as data feeder for CelebA dataset.

# How to use it ?
Clone the repository and set up

```
git clone https://github.com/asahi417/WassersteinGAN
cd WassersteinGAN
pip install .
```
