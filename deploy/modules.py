import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# Conv + bn + relu
class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, mode='conv', use_bn=True):
        super(ConvBlock, self).__init__()
        
        self.use_bn = use_bn
        
        if padding is None:
            padding = (kernel_size - 1) // 2
            pass
        if mode == 'conv':
            self.conv = nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
        elif mode == 'deconv':
            self.conv = nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
        elif mode == 'conv_3d':
            self.conv = nn.Conv3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
        elif mode == 'deconv_3d':
            self.conv = nn.ConvTranspose3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
        else:
            pass
        if self.use_bn:
            if '3d' not in mode:
                self.bn = nn.BatchNorm2d(out_planes)
            else:
                self.bn = nn.BatchNorm3d(out_planes)
                pass
            pass
        self.relu = nn.LeakyReLU(inplace=True)
        self._init_weight()
        return
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, inp):
        # return self.relu(self.conv(inp))
        if self.use_bn:
            return self.relu(self.bn(self.conv(inp)))
        else:
            return self.relu(self.conv(inp))


## The pyramid module from pyramid scene parsing
class PyramidModule(nn.Module):
    def __init__(self, options, in_planes, middle_planes, scales=[64, 32, 16, 8]):
        super(PyramidModule, self).__init__()
        
        self.pool_1 = torch.nn.AvgPool2d((scales[0] * options.height // options.width, scales[0]))
        self.pool_2 = torch.nn.AvgPool2d((scales[1] * options.height // options.width, scales[1]))
        self.pool_3 = torch.nn.AvgPool2d((scales[2] * options.height // options.width, scales[2]))
        self.pool_4 = torch.nn.AvgPool2d((scales[3] * options.height // options.width, scales[3]))
        self.conv_1 = ConvBlock(in_planes, middle_planes, kernel_size=1, use_bn=False)
        self.conv_2 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_3 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_4 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.upsample = torch.nn.Upsample(
            size=(scales[0] * options.height // options.width, scales[0]), mode='bilinear'
        )
        return
    
    def forward(self, inp):
        x_1 = self.upsample(self.conv_1(self.pool_1(inp)))
        print(f"x1shape:{x_1.shape}")
        
        x_2 = self.upsample(self.conv_2(self.pool_2(inp)))
        print(f"x2shape:{x_2.shape}")
        x_3 = self.upsample(self.conv_3(self.pool_3(inp)))
        print(f"x3shape:{x_3.shape}")
        x_4 = self.upsample(self.conv_4(self.pool_4(inp)))
        print(f"x4shape:{x_4.shape}")
        out = torch.cat([inp, x_1, x_2, x_3, x_4], dim=1)
        return out


# The pyramid module from pyramid scene parsing 传统的卷积金字塔
class PyramidModule2(nn.Module):
    # def __init__(self, options, in_planes, middle_planes, scales=[32, 16, 8, 4]):
    def __init__(self, options, in_planes, middle_planes, scales=None):
        super(PyramidModule, self).__init__()
        if scales is None:
            scales = [64, 32, 16, 8]
        
        self.scales = scales
        self.pool_1 = torch.nn.AdaptiveAvgPool2d(scales[0])
        self.pool_2 = torch.nn.AdaptiveAvgPool2d(scales[1])
        self.pool_3 = torch.nn.AdaptiveAvgPool2d(scales[2])
        self.pool_4 = torch.nn.AdaptiveAvgPool2d(scales[3])
        
        self.conv_1 = ConvBlock(in_planes, middle_planes, kernel_size=1, use_bn=False)
        self.conv_2 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_3 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_4 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.upsample = torch.nn.Upsample(
            size=(scales[0], scales[0]), mode='bilinear'
        )
        return
    
    def forward(self, input):
        x_1 = self.upsample(self.conv_1(self.pool_1(input)))
        x_2 = self.upsample(self.conv_2(self.pool_2(input)))
        x_3 = self.upsample(self.conv_3(self.pool_3(input)))
        x_4 = self.upsample(self.conv_4(self.pool_4(input)))
        out = torch.cat([input, x_1, x_2, x_3, x_4], dim=1)
        return out


class FPN(nn.Module):
    def __init__(self, in_planes=512, middle_planes=128, scales=None):
        super(FPN, self).__init__()
        if scales is None:
            scales = [64, 32, 16, 8]
        
        self.pool_1 = torch.nn.AdaptiveMaxPool2d(scales[0])
        self.pool_2 = torch.nn.AdaptiveMaxPool2d(scales[1])
        self.pool_3 = torch.nn.AdaptiveMaxPool2d(scales[2])
        self.pool_4 = torch.nn.AdaptiveMaxPool2d(scales[3])
        
        self.conv = ConvBlock(middle_planes, middle_planes, kernel_size=3, stride=1, padding=1)
        self.conv_nbn = ConvBlock(in_planes, middle_planes, kernel_size=3, stride=1, padding=1, use_bn=False)
        self.conv_1_1 = ConvBlock(middle_planes, middle_planes, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upSampleFinal = nn.Upsample(size=(scales[0], scales[0]), mode='bilinear')
        
        self.attention = AttentionBlock(middle_planes, middle_planes)
    
    def forward(self, backbone_out):
        _, _, H, W = backbone_out.shape
        top = self.pool_1(backbone_out)  # 先将其resize 512 64 64
        
        # 从大到小
        c1 = self.pool_1(self.attention(self.conv_nbn(top)))  # 128 64 64
        
        c2 = self.pool_2(self.attention(self.conv(c1)))  # 128 128 32 32
        
        c3 = self.pool_3(self.attention(self.conv(c2)))  # 128 16 16
        
        c4 = self.pool_4(self.attention(self.conv(c3)))  # torch.Size([1, 128, 8, 8])
        
        p4 = self.conv_1_1(c4)
        p3 = self.upsample(p4) + self.conv_1_1(c3)
        p2 = self.upsample(p3) + self.conv_1_1(c2)
        p1 = self.upsample(p2) + self.conv_1_1(c1)
        
        resized_p4 = self.upSampleFinal(self.conv(p4))
        
        resized_p3 = self.upSampleFinal(self.conv(p3))
        resized_p2 = self.upSampleFinal(self.conv(p2))
        resized_p1 = self.upSampleFinal(self.conv(p1))
        out = torch.cat([top, resized_p1, resized_p2, resized_p3, resized_p4], dim=1)
        
        return out


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


# The module to compute plane depths from plane parameters
def calcPlaneDepthsModule(width, height, planes, metadata, return_ranges=False):
    urange = (torch.arange(width, dtype=torch.float32).cuda().view((1, -1)).repeat(height, 1) / (float(width) + 1) * (
        metadata[4] + 1) - metadata[2]) / metadata[0]
    vrange = (torch.arange(height, dtype=torch.float32).cuda().view((-1, 1)).repeat(1, width) / (float(height) + 1) * (
        metadata[5] + 1) - metadata[3]) / metadata[1]
    ranges = torch.stack([urange, torch.ones(urange.shape).cuda(), -vrange], dim=-1)
    
    planeOffsets = torch.norm(planes, dim=-1, keepdim=True)
    planeNormals = planes / torch.clamp(planeOffsets, min=1e-4)
    
    normalXYZ = torch.sum(ranges.unsqueeze(-2) * planeNormals.unsqueeze(-3).unsqueeze(-3), dim=-1)
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1).unsqueeze(-2).unsqueeze(-2) / normalXYZ
    planeDepths = torch.clamp(planeDepths, min=0, max=MAX_DEPTH)
    if return_ranges:
        return planeDepths, ranges
    return planeDepths


## The module to compute depth from plane information
def calcDepthModule(width, height, planes, segmentation, non_plane_depth, metadata):
    planeDepths = calcPlaneDepthsModule(width, height, planes, metadata)
    allDepths = torch.cat([planeDepths.transpose(-1, -2).transpose(-2, -3), non_plane_depth], dim=1)
    return torch.sum(allDepths * segmentation, dim=1)


## Compute matching with the auction-based approximation algorithm
def assignmentModule(W):
    O = calcAssignment(W.detach().cpu().numpy())
    return torch.from_numpy(O).cuda()


def calcAssignment(W):
    numOwners = int(W.shape[0])
    numGoods = int(W.shape[1])
    P = np.zeros(numGoods)
    O = np.full(shape=(numGoods,), fill_value=-1)
    delta = 1.0 / (numGoods + 1)
    queue = list(range(numOwners))
    while len(queue) > 0:
        ownerIndex = queue[0]
        queue = queue[1:]
        weights = W[ownerIndex]
        goodIndex = (weights - P).argmax()
        if weights[goodIndex] >= P[goodIndex]:
            if O[goodIndex] >= 0:
                queue.append(O[goodIndex])
                pass
            O[goodIndex] = ownerIndex
            P[goodIndex] += delta
            pass
        continue
    return O


## Get one-hot tensor
def oneHotModule(inp, depth):
    inpShape = [int(size) for size in inp.shape]
    inp = inp.view(-1)
    out = torch.zeros(int(inp.shape[0]), depth).cuda()
    out.scatter_(1, inp.unsqueeze(-1), 1)
    out = out.view(inpShape + [depth])
    return out


# Warp image
def warpImages(options, planes, images, transformations, metadata):
    planeDepths, ranges = calcPlaneDepthsModule(options.width, options.height, planes, metadata, return_ranges=True)
    print(planeDepths.shape, ranges.shape, transformations.shape)
    exit(1)
    XYZ = planeDepths.unsqueeze(-1) * ranges.unsqueeze(-2)
    XYZ = torch.cat([XYZ, torch.ones([int(size) for size in XYZ.shape[:-1]] + [1]).cuda()], dim=-1)
    XYZ = torch.matmul(XYZ.unsqueeze(-3), transformations.unsqueeze(-4).unsqueeze(-4))
    UVs = XYZ[:, :, :, :, :, :2] / XYZ[:, :, :, :, :, 2:3]
    UVs = (UVs * metadata[:2] + metadata[2:4]) / metadata[4:6] * 2 - 1
    warpedImages = []
    for imageIndex in range(options.numNeighborImages):
        warpedImage = []
        image = images[:, imageIndex]
        for planeIndex in range(options.numOutputPlanes):
            warpedImage.append(F.grid_sample(image, UVs[:, :, :, imageIndex, planeIndex]))
            continue
        warpedImages.append(torch.stack(warpedImage, 1))
        continue
    warpedImages = torch.stack(warpedImages, 2)
    return warpedImages


if __name__ == '__main__':
    tensor2 = torch.rand(1, 512, 64, 64)
    
    
    class Options:
        def __init__(self, height, width):
            self.height = height
            self.width = width
            pass
    
    
    pyramid = PyramidModule(Options(512, 512), 512, 128)
    pyramid(tensor2)
    # fpn = FPN()
    # print(fpn(tensor2).shape)
    # print(pyramid(tensor2).shape)
    # [1, 1024, 64, 64]
    # ap = torch.nn.AvgPool2d((64, 64))
    # aap = nn.AdaptiveAvgPool2d(64)
    # print(aap(tensor2).shape)
