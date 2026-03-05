import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from escnn import gspaces
from escnn import nn as enn

import torchvision.models as models

#Resnet34 with full escnn

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


class EquivariantResBlock(enn.EquivariantModule):
    def __init__(self, in_type, out_type, stride=1):
        super().__init__()
        self.in_type, self.out_type = in_type, out_type
        
        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu = enn.ReLU(out_type, inplace=True)
        self.conv2 = enn.R2Conv(out_type, out_type, kernel_size=3, padding=1, bias=False, frequencies_cutoff=lambda r: 3 * r)
        self.bn2 = enn.InnerBatchNorm(out_type)
        
        # Shortcut connection to match dimensions
        if stride != 1 or in_type != out_type:
            self.shortcut = enn.SequentialModule(
                enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, bias=False),
                enn.InnerBatchNorm(out_type)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.relu(out + identity)

    # FIX 1: Implement abstract method to allow SequentialModule instantiation
    def evaluate_output_shape(self, input_shape):
        return self.conv1.evaluate_output_shape(input_shape)

class REM(nn.Module):
    def __init__(self, rotations=8):
        super(REM, self).__init__()
        
        # Use Rot2dOnR2 (case-sensitive fix)
        self.r2_space = gspaces.rot2dOnR2(N=rotations)
        
        # FIX 2: Define types so that c_out is the channel count AFTER GroupPooling.
        # To get 128 channels for NetVLAD, we need 128 regular fields.
        def get_type(c_out): 
            return enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * c_out)
        
        in_type = enn.FieldType(self.r2_space, 3 * [self.r2_space.trivial_repr])
        t64 = get_type(64)   # Physical channels: 64 * rotations
        t128 = get_type(128) # Physical channels: 128 * rotations (Matches NetVLAD)

        # 1. Equivariant Stem
        self.equivariant_stem = enn.SequentialModule(
            enn.R2Conv(in_type, t64, kernel_size=7, stride=2, padding=3, bias=False),
            enn.InnerBatchNorm(t64),
            enn.ReLU(t64, inplace=True)
        )
        self.pool = enn.PointwiseMaxPool(t64, kernel_size=3, stride=2, padding=1)

        # 2. Layer 1: 3 Residual Blocks (ResNet-34 Spec)
        self.layer1 = enn.SequentialModule(
            EquivariantResBlock(t64, t64),
            EquivariantResBlock(t64, t64),
            EquivariantResBlock(t64, t64)
        )
        
        # 3. Layer 2: 4 Residual Blocks (ResNet-34 Spec)
        self.layer2 = enn.SequentialModule(
            EquivariantResBlock(t64, t128, stride=2),
            EquivariantResBlock(t128, t128),
            EquivariantResBlock(t128, t128),
            EquivariantResBlock(t128, t128)
        )

        # 4. Final Projection: Collapses 'rotations' dimension to 1
        # Resulting tensor will have exactly 128 channels
        self.invariant_map = enn.GroupPooling(t128)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Lifting: Standard Tensor -> Geometric Tensor
        x_geo = enn.GeometricTensor(x, self.equivariant_stem.in_type)
        
        # Backbone processing
        x_geo = self.pool(self.equivariant_stem(x_geo))
        x_geo = self.layer1(x_geo)
        features_geo = self.layer2(x_geo)
        
        # Invariant mapping: Geometric Tensor -> Standard Tensor [B, 128, H/8, W/8]
        features = self.invariant_map(features_geo).tensor 
        
        # Upsampling logic for NetVLAD and local features
        out1 = F.interpolate(features, size=(H // 4, W // 4), mode='bicubic', align_corners=True)
        out1 = F.normalize(out1, dim=1)
        
        out2 = F.interpolate(features, size=(H, W), mode='bicubic', align_corners=True)
        out2 = F.normalize(out2, dim=1)
        
        return out1, out2
    
class REIN(nn.Module):
    def __init__(self):
        super(REIN, self).__init__()
        self.rem = REM()
        self.pooling = NetVLAD()

        self.local_feat_dim = 128
        self.global_feat_dim = self.local_feat_dim*64
    
    def forward(self, x):

        out1, local_feats = self.rem(x)

        global_desc = self.pooling(out1)

        return out1, local_feats, global_desc


