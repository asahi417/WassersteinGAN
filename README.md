# Wasserstein GAN [WIP]
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/WassersteinGAN/blob/master/LICENSE)

This repository is about tensorflow implementation of [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf).
Properties are summalized as below

- Using TFRecord
- Tested by [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 

There already exist several implementations of WGAN by tensorflow, but
few implementations use TFRecord, which is modern data feeding format optimized as used in tensorflow graph.
In this repository, TFRecord is used as data feeder for CelebA dataset, recently introduced image dataset.

# How to use it ?
Clone the repository and set up

```
git clone https://github.com/asahi417/WassersteinGAN
cd WassersteinGAN
pip install .
```
