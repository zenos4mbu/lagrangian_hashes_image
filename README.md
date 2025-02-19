# Implementation of Lagrangian Hashing for Image Regression

This repository contains an implementation of the paper [Lagrangian Hashing for Compressed Neural Field Representations](https://arxiv.org/abs/2409.05334) applied to the image regression task. **Note:** This implementation heavily relies on the [kaolin-wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp) library.

## To-Do

- **Release status**
  - [x] Release the pytorch implementation
  - [ ] Release stable CUDA implementation
  - [ ] Provide more usage examples and demo scripts

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

Before using this code, ensure you have installed the `kaolin-wisp` library. Please follow the installation instructions provided in the [kaolin-wisp repository](https://github.com/NVIDIAGameWorks/kaolin-wisp).

## Usage

We provide several Bash scripts to quickly evaluate the performance of our model. The key training parameters are:

- **max-epochs**: Maximum number of epochs for model training.
- **pos_lr_weight**: Learning rate for parameters related to the positions of vectors in the codebook (relative to `grid_lr_weight`).
- **var_lr_weight**: Learning rate for parameters related to the standard deviation of vectors in the codebook (relative to `grid_lr_weight`).
- **grid_lr_weight**: Learning rate for all parameters within the codebook.
- **codebook_bitwidth**: Number of bitwidths used to represent the image (applied exponentially).

To run an evaluation, simply execute the desired Bash script. For example:

```bash
./train_image_pytorch.sh
