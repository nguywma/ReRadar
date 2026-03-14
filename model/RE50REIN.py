import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import torchvision.models as models
from .reresnet_model.models.backbones.re_resnet import ReResNet 
from e2cnn.nn import GeometricTensor
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))


    def init_params(self, clsts, traindescs):

        clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :] # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

            

    def forward(self, x):
        N, C = x.shape[:2]
        x_flatten = x.view(N, C, -1)
        
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
    
class REM_R50(nn.Module):
    def __init__(self, from_scratch=False):
        super(REM_R50, self).__init__()        
        self.num_rotations = 8 
        
        # ResNet-50 configuration: 
        # Stage 2 uses Bottleneck blocks and outputs 512 total physical channels
        self.encoder = ReResNet(
            depth=50,
            num_stages=2,
            strides=(1, 2),
            dilations=(1, 1),
            out_indices=(1,), 
            frozen_stages=-1 if from_scratch else 1
        )
        
        # FIX: The physical output is 512, but after pooling 8 rotations, 
        # we have 512 / 8 = 64 channels.
        self.input_to_projection = 512 // self.num_rotations 
        
        # Project from 64 channels up to the 512 channels NetVLAD expects
        self.projection = nn.Conv2d(self.input_to_projection, 512, kernel_size=1)

    def forward(self, x):
        # 1. Forward through ReResNet (returns GeometricTensor)
        geo_out = self.encoder(x)
        out = geo_out.tensor # Shape: [B, 512, H, W]
        
        B, C_phys, H, W = out.shape
        
        # 2. Reshape to [B, 64, 8, H, W] and Max Pool
        # This collapses the 8 rotations into 64 invariant features
        out = out.view(B, -1, self.num_rotations, H, W)
        equ_features, _ = torch.max(out, dim=2) # Shape: [B, 64, H, W]

        # 3. Project 64 -> 512
        features = self.projection(equ_features) # Now matches [512, 64, 1, 1]

        # 4. Spatial Interpolation
        out1 = F.interpolate(features, size=(x.size(2)//4, x.size(3)//4), 
                             mode='bicubic', align_corners=True)
        out1 = F.normalize(out1, dim=1)
        
        out2 = F.interpolate(features, size=(x.size(2), x.size(3)), 
                             mode='bicubic', align_corners=True)
        out2 = F.normalize(out2, dim=1)

        return out1, out2

    def forward(self, x):
        # 1. Forward through the equivariant backbone
        # Returns a GeometricTensor
        geo_out = self.encoder(x)
        out = geo_out.tensor # Shape: [B, 512, H, W]
        
        B, C_total, H, W = out.shape
        
        # 2. Reshape and Group Pool to achieve Invariance
        # ReResNet logic: physical channels = (base_channels / rotations) * rotations
        # For ResNet50 Layer 2, base_channels is 512.
        out = out.view(B, -1, self.num_rotations, H, W)
        equ_features, _ = torch.max(out, dim=2) # Shape: [B, 64, H, W] 
        # (Note: 512 total / 8 rotations = 64 base channels)

        # 3. Project to the 128 channels NetVLAD expects
        # Since the 'slim' ReResNet only gave us 64 base features, 
        # we project up to 128.
        features = self.projection(equ_features)

        # 4. Spatial Interpolation for NetVLAD and Keypoints
        out1 = F.interpolate(features, size=(x.size(2)//4, x.size(3)//4), 
                             mode='bicubic', align_corners=True)
        out1 = F.normalize(out1, dim=1)
        
        out2 = F.interpolate(features, size=(x.size(2), x.size(3)), 
                             mode='bicubic', align_corners=True)
        out2 = F.normalize(out2, dim=1)

        return out1, out2

class REIN(nn.Module):
    def __init__(self, num_clusters=64):
        super(REIN, self).__init__()
        self.rem = REM_R50()
        self.pooling = NetVLAD(num_clusters=num_clusters, dim=512)
        self.global_feat_dim = 512
    
    def forward(self, x):
        out1, local_feats = self.rem(x)
        global_desc = self.pooling(out1)
        return out1, local_feats, global_desc