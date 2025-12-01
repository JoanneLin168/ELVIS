import torch
import torch.nn as nn
import os
import math

from torchvision import transforms
from PIL import Image


# Hacky solution: randomly selects indices to sample video clip from
class BaseVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_frames=5, video_list=[], transform=None, train=False, dataset_name=None, device='cuda'):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.train = train
        self.device = device

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform

        self.videos = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        # Filter videos if video_list is provided
        if len(video_list) > 0:
            self.videos = [video for video in self.videos if video in video_list]


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        video_path = os.path.join(self.root_dir, video)
        
        frames = sorted([
            f for f in os.listdir(video_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # Randomly pick centre_idx
        if self.train:
            center_idx = torch.randint(self.num_frames // 2, len(frames) - self.num_frames // 2, (1,)).item()
            half_len = self.num_frames // 2
            if self.num_frames % 2 == 0:
                start = center_idx - half_len
                end = center_idx + half_len
            else:
                start = center_idx - half_len
                end = center_idx + half_len + 1
        else:
            center_idx = self.num_frames // 2
            start = 0
            end = self.num_frames

        clip_frame_paths = frames[start:end]

        clip = []
        for frame_name in clip_frame_paths:
            frame_path = os.path.join(video_path, frame_name)
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            clip.append(img)

        return torch.stack(clip)  # [num_frames, C, H, W]