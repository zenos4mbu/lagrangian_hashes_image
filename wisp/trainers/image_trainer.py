# The MIT License (MIT)
#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
import wandb

import logging as log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from typing import Tuple

from wisp.config import configure, autoconfig, instantiate
from wisp.trainers import BaseTrainer, ConfigBaseTrainer
from wisp.ops.image.metrics import psnr, lpips, ssim

import git

class ImageTrainer(BaseTrainer):
    
    def pre_training(self):
        """
        Override this function to change the logic which runs before the first training iteration.
        This function runs once before training starts.
        """
        super().pre_training()
        self.tracker.metrics.define_metric('rgb_loss', aggregation_type=float)

    @torch.cuda.nvtx.range("ImageTrainer.step")
    def step(self, data):

        coords = data[0].to(self.device)
        rgb = data[1].to(self.device)
        grads = data[2].to(self.device)

        batch_size = coords.shape[0]
        assert(batch_size == 1)
        coords = coords[0]
        rgb = rgb[0]
        grads = grads[0]

        self.optimizer.zero_grad()
        loss = 0
        
        with torch.cuda.amp.autocast():
            rb = self.pipeline.nef.rgb(coords)
            rgb_pred = rb["rgb"]
            rgb_loss = F.mse_loss(rgb_pred, rgb)
            rgb_loss = rgb_loss.mean()
            loss += rgb_loss
        
        self.pipeline.nef.grid.codebook.update_stds()

        # Density loss
        # with torch.cuda.amp.autocast():
        #     gmms = rb["gmms"]
        #     kl_div = gmms * grads
        #     density_loss = kl_div.mean()
        #     loss += density_loss * 0.001



        # positions = self.pipeline.nef.grid.codebook.pos
        # variances = self.pipeline.nef.grid.codebook.var
        # begin_idxes = self.pipeline.nef.grid.codebook.begin_idxes_pos
        # resolutions = self.pipeline.nef.grid.codebook.resolutions
        # # Number of levels
        # num_levels = len(begin_idxes) - 1


        # # Position loss
        # with torch.cuda.amp.autocast():
        #     loss_pos = torch.mean(torch.relu(positions - 1.)) + torch.mean(torch.relu(-1. - positions))
        #     loss += loss_pos * 0.01

        # # Variance loss
        # loss_var = 0
        # with torch.cuda.amp.autocast():
        #     for level in range(num_levels):
        #         start_idx = begin_idxes[level]
        #         end_idx = begin_idxes[level + 1]
        #         level_var = variances[start_idx:end_idx]
        #         loss_var_step = torch.mean((torch.relu(torch.abs(level_var) - 4/resolutions[level].cuda())**2))

        #          # Replace NaNs with 0
        #         loss_var_step = torch.where(torch.isnan(loss_var_step), torch.zeros_like(loss_var_step), loss_var_step)

        #         loss_var += loss_var_step
        #     loss += (loss_var / num_levels * 0.01)

        self.tracker.metrics.total_loss += loss.item()
        self.tracker.metrics.rgb_loss += rgb_loss.item()
        
        
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        if self.cfg.scheduler:
            self.scheduler.step()
            
    def log_console(self):
        total_loss = self.tracker.metrics.average_metric('total_loss')
        rgb_loss = self.tracker.metrics.average_metric('rgb_loss')
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(total_loss)
        log_text += ' | rgb loss: {:>.3E}'.format(rgb_loss)
        codebook_np = self.pipeline.nef.grid.codebook.pos.detach().cpu().numpy()
        begin_idxes = self.pipeline.nef.grid.codebook.begin_idxes_pos.detach().cpu().numpy()
        splash_levels = self.pipeline.nef.grid.codebook.num_splashes_lod

        if self.epoch % 20 == 1:
            num_levels = len(begin_idxes) - 1
            train_image = self.train_dataset.image  # Assuming this is a numpy array
            img_height, img_width = self.train_dataset.image.shape[:2]

            # Filter levels with splashes > 0
            active_levels = [level for level, splashes in enumerate(splash_levels) if splashes > 0]
            
            # Initialize figure and axes for the active levels only
            fig, axes = plt.subplots(1, len(active_levels), figsize=(15, 6))
            fig.suptitle("Gaussian Positions in Codebook for Active Levels")
            
            # Ensure axes is an array even if there's only one plot
            if len(active_levels) == 1:
                axes = [axes]
            
            # Loop over active levels
            for plot_index, level in enumerate(active_levels):
                start_idx = begin_idxes[level]
                end_idx = begin_idxes[level + 1]
                ax = axes[plot_index]
                
                level_codebook = codebook_np[start_idx:end_idx, :]
                
                # Normalizing coordinates from [-1, 1] to [0, 1]
                gx_values = level_codebook[:, 0]
                gy_values = level_codebook[:, 1]
                
                # The extent argument specifies the (left, right, bottom, top) bounds of the image
                ax.imshow(train_image, extent=[-1, 1, -1, 1], aspect='auto', alpha=0.5)
                
                # Plotting with adjusted coordinates
                ax.scatter(gx_values, gy_values, c='blue', label='Gaussians', s=0.5, alpha=0.3)

                # Setting limits and labels as before, though they might be redundant with extent set
                ax.set_xlim([-1.2, 1.2])
                ax.set_ylim([-1.2, 1.2])
                ax.invert_yaxis()  # Invert the y-axis to match the image coordinate system

                ax.set_title(f"Level {level}")
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel("gx")
                ax.set_ylabel("gy")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)

            # Create the directory if it does not exist
            directory = f'plots/{self.cfg.exp_name}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Save the plot
            # plt.savefig(f'{directory}/gaussian_positions_overlay{self.epoch:05d}.png')
            # plt.close(fig)  # Make sure to close the specific figure instance

            # Log the last plot to wandb
            # wandb.log({"gaussian_positions_overlay": wandb.Image(f'{directory}/gaussian_positions_overlay{self.epoch:05d}.png')})
        # Continue with your logging as before
        # wandb.log({"total_loss": total_loss, "rgb_loss": rgb_loss})
        log.info(log_text)

    def validate(self):
        
        self.pipeline.nef.eval()
        
        record_dict = self.tracker.get_record_dict()
        if record_dict is None:
            log.info("app_config not supplied to Tracker, config won't be logged in the pandas log")
            record_dict = {}
        
        sha = None
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha[:8]
            record_dict["git_sha"] = sha
        except:
            log.info("Validation script not running within a git directory, so skipping SHA logging.")

        # Infer image
        coords = self.train_dataset.coords.cuda()
        img = []
        # if self.epoch == 500:
        #     positions = self.pipeline.nef.grid.codebook.pos
        #     # variances = self.pipeline.nef.grid.codebook.var
        #     # features = self.pipeline.nef.grid.codebook.feats
        #     # Define center and margin
        #     center = torch.tensor([0.5, 0.1], device=positions.device)
        #     margin = 0.25

        #     # Calculate distances from the center
        #     distances = torch.sqrt(((positions - center) ** 2).sum(dim=1))

        #     # Create a mask for positions within the specified margin
        #     close_to_center_mask = distances <= margin

        #     # Get a clone of the positions to avoid modifying the original tensor in-place
        #     new_positions = self.pipeline.nef.grid.codebook.pos.clone()

        #     # Set the positions close to the center to a high value (e.g., 1000000)
        #     new_positions[close_to_center_mask] = 1000000

        #     # Update the positions in the codebook with the modified positions
        #     self.pipeline.nef.grid.codebook.pos = torch.nn.Parameter(new_positions)

        for _coords in torch.split(coords, 1000000):
            with torch.no_grad():
                _img = self.pipeline.nef.rgb(_coords)['rgb'].detach()
            img.append(_img)
        img = torch.cat(img).reshape(self.train_dataset.h, self.train_dataset.w, 3)
        
        log_img = (img * 255).byte().cpu().permute(2,0,1).numpy()

        if coords.shape[0] < 1000000:
            self.tracker.log_image("rgb_output", log_img, self.epoch)

        # Begin validation; compute PSNR
        gts = self.train_dataset.pixels.reshape(self.train_dataset.h, self.train_dataset.w, 3).cuda()
        
        if 'lpips' in self.cfg.valid_metrics and not hasattr(self, 'lpips_model'):
            try:
                from lpips import LPIPS
            except:
                raise Exception(
                    "Module lpips not available. To install, run `pip install lpips`")
            self.lpips_model = LPIPS(net='vgg').cuda()
        
        metrics_dict = {}
        if 'psnr' in self.cfg.valid_metrics:
            if coords.shape[0] < 1000000:
                metrics_dict['psnr'] = psnr(img[...,:3], gts[...,:3])
            else:
                # Doesnt fit on GPUs...
                metrics_dict['psnr'] = psnr(img[...,:3].cpu(), gts[...,:3].cpu())
        if 'lpips' in self.cfg.valid_metrics:
            metrics_dict['lpips'] = lpips(img[...,:3], gts[...,:3], self.lpips_model)
        if 'ssim' in self.cfg.valid_metrics:
            metrics_dict['ssim'] = ssim(img[...,:3], gts[...,:3])
        
        # Save images
        # Image.fromarray((img*255).byte().cpu().numpy()).save(os.path.join(self.tracker.log_dir, 'img_pred{self.epoch}.png'))
        # Image.fromarray((gts*255).byte().cpu().numpy()).save(os.path.join(self.tracker.log_dir, 'img_gts{self.epoch}.png'))
        Image.fromarray((img*255).byte().cpu().numpy()).save(os.path.join(self.tracker.log_dir, f'img_pred{self.epoch}.png'))
        Image.fromarray((gts*255).byte().cpu().numpy()).save(os.path.join(self.tracker.log_dir, f'img_gts{self.epoch}.png'))

            
        for key in metrics_dict:
            if not key in self.return_dict:
                self.return_dict[key] = metrics_dict[key]
            else:
                self.return_dict[key] = max(self.return_dict[key], metrics_dict[key])
            log.info(f"{key}: {metrics_dict[key]:.2f}")
            self.tracker.log_metric(f"validation/{key}", metrics_dict[key], self.epoch)

        # Get image name and model path to write to database
        img_name = os.path.splitext(os.path.basename(self.train_dataset.root))[0]
        model_fname = os.path.abspath(os.path.join(self.tracker.log_dir, f'model.pth'))
        
        record_dict.update({"img_name" : img_name, "epoch": self.epoch, 
                            "log_fname" : self.tracker.log_fname, "model_fname": model_fname,
                            "git_sha" : sha if not sha is None else "", "width": self.train_dataset.w, "height": self.train_dataset.h, 
                            "num_pixels": self.train_dataset.w * self.train_dataset.h})

        record_dict.update(metrics_dict)

        # Save logs
        df = pd.DataFrame.from_records([record_dict])
        parent_log_dir = os.path.dirname(self.tracker.log_dir)
        fname = os.path.join(parent_log_dir, f"logs.parquet")
        if os.path.exists(fname):
            df_ = pd.read_parquet(fname)
            df = pd.concat([df_, df])
        df.to_parquet(fname, index=False)
        
        return
    
    def render_snapshot(self):
        # Snapshot rendering is done in validation for this trainer
        return
