from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model.REIN import NetVLAD


class REM_R50(nn.Module):
    def __init__(self, from_scratch=False):
        super(REM_R50, self).__init__()

        weights = None
        if not from_scratch:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)

        # Keep the first two ResNet stages, matching the ReResNet variants.
        self.encoder = nn.Sequential(OrderedDict([
            ("conv1", resnet.conv1),
            ("bn1", resnet.bn1),
            ("relu", resnet.relu),
            ("maxpool", resnet.maxpool),
            ("layer1", resnet.layer1),
            ("layer2", resnet.layer2),
        ]))
        # layer2 already outputs 512 channels. Keep all of them so this model
        # uses the same NetVLAD local/global dimensions as RE50REIN.
        self.projection = nn.Identity()

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x):
        height, width = x.shape[-2:]

        # KITTI images are loaded by OpenCV in BGR order.
        x = x[:, [2, 1, 0], :, :]
        x = (x - self.image_mean) / self.image_std
        features = self.projection(self.encoder(x))

        out1 = F.interpolate(
            features,
            size=(height // 4, width // 4),
            mode="bicubic",
            align_corners=True,
        )
        out1 = F.normalize(out1, dim=1)

        local_feats = F.interpolate(
            features,
            size=(height, width),
            mode="bicubic",
            align_corners=True,
        )
        local_feats = F.normalize(local_feats, dim=1)
        return out1, local_feats


class REIN(nn.Module):
    def __init__(self, num_clusters=64, from_scratch=False):
        super(REIN, self).__init__()

        self.rem = REM_R50(from_scratch=from_scratch)
        self.local_feat_dim = 512
        self.pooling = NetVLAD(
            num_clusters=num_clusters,
            dim=self.local_feat_dim,
        )
        self.global_feat_dim = self.local_feat_dim * num_clusters

    def forward(self, x):
        out1, local_feats = self.rem(x)
        global_desc = self.pooling(out1)
        return out1, local_feats, global_desc

    def backbone_state_dict(self):
        return self.rem.encoder.state_dict()
