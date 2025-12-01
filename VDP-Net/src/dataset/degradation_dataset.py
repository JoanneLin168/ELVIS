import torch
import torch.nn as nn
import os
import math
import numpy as np
import json

from torchvision import transforms
from PIL import Image
from .video_dataset import BaseVideoDataset
from .noise import reshape_noise_params, generate_noise
from .utils import get_random_coordinates, crop_image


class DegradationVideoDataset(BaseVideoDataset):
    def __init__(self, root_dir, num_frames=5, patch_size=256, video_list=[], transform=None, train=False, device='cuda'):
        super().__init__(root_dir, num_frames, video_list, transform, train, device)
        self.num_frames = num_frames
        self.patch_size = patch_size

        self.noise_types = ['exposure_value', 'shot_noise_log',
                                'read_noise', 'quant_noise', 'band_noise', 'band_noise_angle',
                                'blur_sigma1', 'blur_sigma2', 'blur_angle']


    def get_random_parameters(self, x):
        N, C, H, W = x.shape
        label = {
            'exposure_value': torch.rand(1).repeat(N).to(self.device),
            'shot_noise_log': torch.rand(1).repeat(N).to(self.device),
            'read_noise': torch.rand(1).repeat(N).to(self.device),
            'quant_noise': torch.rand(1).repeat(N).to(self.device),
            'band_noise': torch.rand(1).repeat(N).to(self.device),
            'band_noise_angle': torch.round(torch.rand(1)).float().repeat(N).to(self.device), # 0: horizontal, 1: vertical
            'blur_sigma1': torch.rand(N).to(self.device),
            'blur_sigma2': torch.rand(N).to(self.device),
            'blur_angle': torch.rand(N).to(self.device), # so if sig1>sig2 then | to \ kernels, otherwise - to / kernels
        }
        label = [label[key] for key in self.noise_types]
        return torch.stack(label, dim=1).to(x.device)

    def __getitem__(self, idx):
        video = self.videos[idx]
        video_path = os.path.join(self.root_dir, video)
        
        frames = sorted([
            f for f in os.listdir(video_path)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])

        # Randomly pick centre_idx
        if self.train:
            center_idx = torch.randint(self.num_frames // 2, len(frames) - self.num_frames // 2, (1,)).item()
        else:
            center_idx = self.num_frames // 2

        half_len = self.num_frames // 2
        if self.num_frames % 2 == 0:
            start = center_idx - half_len
            end = center_idx + half_len
        else:
            start = center_idx - half_len
            end = center_idx + half_len + 1

        clip_frame_paths = frames[start:end]

        clip = []
        crop_coords = None
        for frame_name in clip_frame_paths:
            frame_path = os.path.join(video_path, frame_name)
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            if self.train:
                # pad the image to the patch size if it is smaller
                if img.shape[-2] < self.patch_size or img.shape[-1] < self.patch_size:
                    pad_h = max(0, self.patch_size - img.shape[-2])
                    pad_w = max(0, self.patch_size - img.shape[-1])
                    img = transforms.functional.pad(img, (0, 0, pad_w, pad_h), fill=0, padding_mode='constant')
                # Randomly crop the image to the patch size
                if crop_coords is None:
                    crop_coords = get_random_coordinates(img, self.patch_size)
                img = crop_image(img, self.patch_size, coordinates=crop_coords)
            clip.append(img)

        gt = clip[half_len].to(self.device)  # Ground truth is the center frame
        clip = torch.stack(clip).to(self.device)  # [N, C, H, W]
        noisy = clip.clone()

        # Add degradation
        noise_params = self.get_random_parameters(clip) # repeated in function
        noise_dict = reshape_noise_params(noise_params.unsqueeze(0), num_frames=self.num_frames)
        
        noisy, x_dark = generate_noise(noisy.unsqueeze(0), noise_dict, num_frames=self.num_frames, return_dark=True)
        noisy = noisy.squeeze(0)
        x_dark = x_dark.squeeze(0)

        return {
            'noisy_frames': noisy,
            'clean_frames': clip,
            'clean_center': gt,
            'dark_frames': x_dark,
            'gt_labels': noise_params,
        }
