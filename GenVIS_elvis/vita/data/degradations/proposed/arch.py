
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as tf

from .blocks import ConvBlock, ConvBlock3d
from .noise import reshape_noise_params, generate_noise


class VDPNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, args=None, feat_layer='layer4', num_frames=16, num_classes=9):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_frames = num_frames
        self.args = args

        # Stems
        self.stem2d = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks
        # Create backbones for 2D and 3D feature extraction
        return_nodes = {f"layer{i}": f"layer{i}" for i in range(1, 5)}

        if "resnet50" in args.notes:
            backbone = resnet50(weights='DEFAULT').eval()
            layer_channels = {f"layer{i+1}": 2**(i+8) for i in range(4)} # for resnet50
            print("Using ResNet50 backbone on layer:", feat_layer)
        else:
            backbone = resnet18(weights='DEFAULT').eval()
            layer_channels = {f"layer{i+1}": 2**(i+6) for i in range(4)} # for resnet18
            print("Using ResNet18 backbone on layer:", feat_layer)
        self.backbone = create_feature_extractor(backbone, return_nodes)
        self.feat_layer = feat_layer

        self.num_classes = num_classes
        self.num_features = layer_channels[feat_layer]
        self.avgpool3d = nn.AdaptiveAvgPool3d(1)
        self.fusion3d = nn.Sequential(
            nn.Conv1d(self.num_frames, 1, kernel_size=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True)
        )
        self.head3d = nn.Linear(self.num_features, num_classes - 3) # removing 2 for motion blur in 2d backbone

        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
        self.head2d = nn.Linear(self.num_features, 3)


    def estimate_noise(self, noisy):
        B, N, C, H, W = noisy.size()
        noisy = noisy.view(-1, C, H, W)

        # Pass into backbone
        feats = self.backbone(noisy)[self.feat_layer]
        feats = self.avgpool2d(feats) # (B*N, C, 1, 1)
        feats = feats.view(B, N, self.num_features)

        # Get 2d feats and output
        feats_2d = feats.view(-1, self.num_features) # (B, N, C) -> (B*N, C)
        feats_2d = torch.flatten(feats_2d, 1)
        v_pred_2d = self.head2d(feats_2d)
        v_pred_2d = torch.sigmoid(v_pred_2d)
        v_pred_2d = v_pred_2d.view(B, N, -1) # (B*N, 2) -> (B, N, 2)

        # Get 3d feats and output
        feats_3d = self.fusion3d(feats).squeeze(1) # (B, N, C) -> (B, C)
        feats_3d = torch.flatten(feats_3d, 1)
        v_pred_3d = self.head3d(feats_3d)
        v_pred_3d = torch.sigmoid(v_pred_3d)
        v_pred_3d = v_pred_3d.unsqueeze(1).repeat(1, self.num_frames, 1)

        # Insert 2d motion blur predictions into noise params
        noise_params = torch.zeros((B, N, self.num_classes), device=noisy.device)
        noise_params[:, :, :-3] = v_pred_3d
        noise_params[:, :, -3:] = v_pred_2d
        return noise_params

    def forward(self, clean, noisy):
        B, N, C, H, W = noisy.size()
        noise_params = self.estimate_noise(noisy)

        # Use v_pred to synthesize noise onto clean frames
        noise_dict = reshape_noise_params(noise_params, num_frames=self.num_frames)
        x_synth = generate_noise(clean, noise_dict, num_frames=self.num_frames)
        x_synth = x_synth.view(B, N, C, H, W)

        return {'synth_noisy': x_synth, 'pred_labels': noise_params}

