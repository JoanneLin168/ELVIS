# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder

from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import SelfAttentionLayer, CrossAttentionLayer, PositionEmbeddingSine


@SEM_SEG_HEADS_REGISTRY.register()
class EnhancerV3MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # NOTE Modified by Sukjun Hwang: Issues with recent detectron2 versions.
        version = 2 # local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
        enhancer: nn.Module
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.enhancer = enhancer
        self.hidden_dim = 256
        self.enhancer_out_conv = nn.Conv2d(self.hidden_dim, 3, kernel_size=1, stride=1, padding=0)
        N_steps = self.hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # Fusion
        self.fusion_crossatt = nn.ModuleList()
        self.fusion_selfatt = nn.ModuleList()
        for _ in range(3):
            self.fusion_crossatt.append(CrossAttentionLayer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                ))
            self.fusion_selfatt.append(SelfAttentionLayer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                ))

        self.mask_fusion = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.clip_fusion = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        enhancer = MSDeformAttnPixelDecoder(
            input_shape = {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            mask_dim = cfg.MODEL.SEM_SEG_HEAD.ENHANCER_OUTPUT_DIM, # only change here
            norm = cfg.MODEL.SEM_SEG_HEAD.NORM,
            transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT,
            transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS,
            transformer_dim_feedforward = 1024,
            transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS,
            transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES,
            common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
        )

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "enhancer": enhancer,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        enhancer_out_feats, clip_enhancer_out_feats, enhancer_enc_out_feats, enhancer_ms_feats = self.enhancer.forward_features(features)
        mask_features, clip_mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)

        # Fuse features together
        # NOTE: for GenVIS "multi_scale_pixel_decoder" is used, not transformer_encoder. so use multi_scale_features
        for i in range(len(enhancer_ms_feats)):
            pos = self.pe_layer(multi_scale_features[i], None).flatten(2).permute(2,0,1)
            b, c, h, w = multi_scale_features[i].shape
            Q = multi_scale_features[i].flatten(2).permute(2, 0, 1)
            KV = enhancer_ms_feats[i].flatten(2).permute(2, 0, 1)
            fused_feat = self.fusion_crossatt[i](Q, KV, pos=pos)
            fused_feat = self.fusion_selfatt[i](fused_feat)
            multi_scale_features[i] = fused_feat.permute(1,2,0).view(b, c, h, w) + multi_scale_features[i] # skip connection

        concat_feats = torch.cat([enhancer_out_feats, mask_features], dim=1)
        mask_features = self.mask_fusion(concat_feats)
        concat_feats = torch.cat([clip_enhancer_out_feats, clip_mask_features], dim=1)
        clip_mask_features = self.clip_fusion(concat_feats)


        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, clip_mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)

        enhancer_out = self.enhancer_out_conv(enhancer_out_feats)
        predictions += (enhancer_out,)

        del enhancer_enc_out_feats, fused_feat, concat_feats
        return predictions
