import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from escnn import gspaces
from escnn import nn as enn

import torchvision.models as models

#Resnet34 with first block escnn

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


class REM(nn.Module):
    def __init__(self, rotations=8, from_scratch=False):
        super(REM, self).__init__()
        
        # 1. ESCNN Stem (Equivariant to Rotations)
        self.r2_space = gspaces.rot2dOnR2(N=rotations)
        in_type = enn.FieldType(self.r2_space, 3 * [self.r2_space.trivial_repr])
        out_type = enn.FieldType(self.r2_space, 64 * [self.r2_space.regular_repr])
        
        self.equivariant_stem = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=7, stride=2, padding=3, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True),
            enn.GroupPooling(out_type) # Makes features invariant
        )
        
        # 2. Match ResNet resolution (ResNet stem ends with MaxPool)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 3. Standard ResNet Backbone (Remaining Blocks)
        full_resnet = models.resnet34(pretrained=not from_scratch)
        self.resnet_backbone = nn.Sequential(
            full_resnet.layer1,
            full_resnet.layer2
        )

    def forward(self, x):
        # ESCNN requires geometric tensor input
        x_geo = enn.GeometricTensor(x, self.equivariant_stem.in_type)
        x_invariant = self.equivariant_stem(x_geo).tensor # Back to torch tensor
        
        x_pooled = self.pool(x_invariant)
        features = self.resnet_backbone(x_pooled)
        
        B, C, H, W = x.size()
        # Your original upsampling logic
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


