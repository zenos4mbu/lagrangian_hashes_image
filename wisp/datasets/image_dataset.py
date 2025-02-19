# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys

import glob

import numpy as np
import logging as log

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import imageio
import wisp.ops.image as img_ops
import wisp.ops.geometric as geo_ops
from PIL import Image

_img_suffix = ['png','jpg','jpeg','bmp','tif']

class ImageDataset(Dataset):
    """This is a single image dataset class.

    This class should be used for training tasks where the task is to fit a single
    image using a neural field. 
    """
    def __init__(self, 
        dataset_path       : str,
        num_pixels_per_image : int = 4096,
    ):
        self.root = os.path.abspath(os.path.expanduser(dataset_path))
        suffix = self.root.split('.')[-1]
        if suffix in _img_suffix:
            self.image = torch.from_numpy(np.array(Image.open(self.root)))
            self.image = self.image / 255.0
        elif suffix == 'exr':
            self.image = torch.from_numpy(imageio.imread(self.root)).float()
        else:
            raise Exception(f"Unsupported image format: {suffix}")
        
        self.num_pixels_per_image = num_pixels_per_image

        if self.image.shape[-1] not in [3, 4]:
            raise Exception("Image is not RGB or RGBA.")
        
        if self.image.shape[-1] == 4:
            # Handle the alpha channel if necessary. Here we simply drop it, but you might want to handle it differently.
            self.image = self.image[:, :, :3]

        # Convert image to grayscale if it's not already
        if self.image.shape[-1] == 3:
            self.image_gray = self.rgb_to_grayscale(self.image).squeeze()  # Assuming a batch dimension is required
        else:
            self.image_gray = self.image[:, :, 0]  # Assuming the image is already grayscale

        # Calculate the gradient magnitude
        self.gradient_magnitude = self.compute_gradient_magnitude(self.image_gray)

        self.h, self.w = self.image.shape[:2]
        self.coords = geo_ops.normalized_grid(self.h, self.w, use_aspect=False).reshape(-1, 2).cpu()
        self.pixels = self.image.reshape(-1, 3)
        self.gradients = self.gradient_magnitude.reshape(-1, 1)

    def rgb_to_grayscale(self, image):
        if image.dim() == 4 and image.shape[1] == 3:  # Check if image is in [B, C, H, W] format
            # Convert from [B, C, H, W] to [B, H, W, C] for computation
            image = image.permute(0, 2, 3, 1)
        if image.dim() == 3 and image.shape[2] == 3:  # Check if image is in [H, W, C] format
            r, g, b = image.split(1, dim=-1)
            return 0.2989 * r + 0.5870 * g + 0.1140 * b
        raise ValueError("Invalid image format for grayscale conversion")


    def compute_gradient_magnitude(self, image):
        # Define Sobel operators for x and y directions
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Add channel dimension to image if needed
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension
        elif len(image.shape) == 3:
            image = image.unsqueeze(1)  # Add channel dimension

        # Ensure sobel operators are on the same device as the image
        sobel_x = sobel_x.to(image.device)
        sobel_y = sobel_y.to(image.device)

        # Apply Sobel operators
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)

        # Calculate gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()  # Remove extra dimensions

        return grad_magnitude

    def get_image(self):
        return self.image

    def show_image(self):
        """Displays the image with dynamic range adjustment."""
        image_np = self.image.numpy() if isinstance(self.image, torch.Tensor) else self.image
        
        # Adjust the dynamic range to [0, 1] for visualization.
        image_min = image_np.min()
        image_max = image_np.max()
        adjusted_image = (image_np - image_min) / (image_max - image_min)
        
        plt.imshow(adjusted_image)
        plt.axis('off')  # Hide the axis.
        plt.show()
    
    def show_gradient_image(self):
        """Displays the image with dynamic range adjustment."""
        image_np = self.gradient_magnitude.numpy() if isinstance(self.image, torch.Tensor) else self.image
        
        # Adjust the dynamic range to [0, 1] for visualization.
        image_min = image_np.min()
        image_max = image_np.max()
        adjusted_image = (image_np - image_min) / (image_max - image_min)
        
        plt.imshow(adjusted_image)
        plt.axis('off')  # Hide the axis.
        plt.show()
    
    def __len__(self):
        return 100

    def __getitem__(self, idx : int):
        rand_idx = torch.randint(0, self.coords.shape[0], (self.num_pixels_per_image,))
        return self.coords[rand_idx], self.pixels[rand_idx], self.gradients[rand_idx]
