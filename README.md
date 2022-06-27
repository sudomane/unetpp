# Implementation of an L-sized U-Net++ using Keras and Tensorflow

The architecture for the model is based on [this paper](https://arxiv.org/abs/1807.10165)

## Overview

The algorithm that generates the L-sized U-Net++ was designed for the MICCAI BraTS challenge.
It has already been tested with the data provided by MICCAI.
It is recommended to create models of size L between 1 and 4, as anything above 4 may not be compatible with your computer's architecture.

Tested with Tensorflow 2.0.
