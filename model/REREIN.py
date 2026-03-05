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

    def __init__(self, num_clusters=128, dim=128):
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
    
class REM_Equivariant(nn.Module):
    def __init__(self, from_scratch=False):
        super(REM_Equivariant, self).__init__()
        
        # The group order (rotations)
        self.num_rotations = 8 
        
        # FIX: Multiply your target base channels by num_rotations.
        # NetVLAD wants 128. 128 * 8 = 1024.
        # We set base_channels=512 so that Layer 2 (base*2) becomes 1024.
        # The internal logic (1024 / 8) will then result in 128 base channels.
        self.encoder = ReResNet(
            depth=34,
            stem_channels=64 * self.num_rotations, 
            base_channels=64 * self.num_rotations, 
            num_stages=2,
            strides=(1, 2),
            dilations=(1, 1),
            out_indices=(1,), 
            frozen_stages=-1 if from_scratch else 1
        )

    def forward(self, x):
        geo_out = self.encoder(x)
        out = geo_out.tensor # Physical shape: [B, 128 * 8, H, W]
        
        B, C_phys, H, W = out.shape
        
        # Reshape to [B, 128, 8, H, W]
        out = out.view(B, -1, self.num_rotations, H, W)
        
        # Max pool along rotation dim to get [B, 128, H, W]
        equ_features, _ = torch.max(out, dim=2) 

        # NetVLAD branch
        out1 = F.interpolate(equ_features, size=(x.size(2)//4, x.size(3)//4), 
                             mode='bicubic', align_corners=True)
        out1 = F.normalize(out1, dim=1)
        
        # Keypoint branch
        out2 = F.interpolate(equ_features, size=(x.size(2), x.size(3)), 
                             mode='bicubic', align_corners=True)
        out2 = F.normalize(out2, dim=1)

        return out1, out2
    
class REIN(nn.Module):
    def __init__(self, num_clusters=64):
        super(REIN, self).__init__()
        self.rem = REM_Equivariant()
        # The output of ResNet34 layer2 is 128 channels.
        self.local_feat_dim = 128 
        self.pooling = NetVLAD(num_clusters=num_clusters, dim=self.local_feat_dim)
        self.global_feat_dim = self.local_feat_dim * num_clusters
    
    def forward(self, x):
        out1, local_feats = self.rem(x)
        global_desc = self.pooling(out1)
        return out1, local_feats, global_desc