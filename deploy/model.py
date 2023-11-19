import torch
from torch import nn
from drn import drn_d_54
# from models.drn import drn_d_105
from modules import ConvBlock, PyramidModule, FPN
from utils import NUM_CORNERS, NUM_ROOMS, NUM_ICONS


class Model(nn.Module):
    def __init__(self, options):
        
        super(Model, self).__init__()
        
        self.options = options
        # backbone
        self.drn = drn_d_54(pretrained=True, out_map=64, num_classes=-1, out_middle=False)
        # FPN
        self.pyramid = PyramidModule(options, 512, 128)
        self.feature_conv = ConvBlock(1024, 512)
        self.segmentation_pred = torch.nn.Conv2d(512, NUM_CORNERS, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(options.height, options.width), mode='bilinear')
        return
    
    def forward(self, inp):
        features = self.drn(inp)
        features = self.pyramid(features)
        features = self.feature_conv(features)
        
        features = self.segmentation_pred(features)
        segmentation = self.upsample(features)
        segmentation = segmentation.transpose(1, 2).transpose(2, 3).contiguous()
        if self.options.method_type == 1:
            return torch.sigmoid(segmentation),
        else:
            return (
                torch.sigmoid(segmentation[:, :, :, :NUM_CORNERS]),
                segmentation[:, :, :, NUM_CORNERS:NUM_CORNERS + NUM_ICONS + 2],
                segmentation[:, :, :, -(NUM_ROOMS + 2):]
            )


class ModleV2(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
        # backbone
        self.drn = drn_d_54(pretrained=True, out_map=64, num_classes=-1, out_middle=False)
        self.fpn = FPN(512, 128)
        self.feature_conv = ConvBlock(1024, 512)
        self.segmentation_pred = torch.nn.Conv2d(512, NUM_CORNERS, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(options.height, options.width), mode='bilinear')
        return
    
    def forward(self, input_img):
        features = self.drn(input_img)
        features = self.pyramid(features)
        features = self.feature_conv(features)
        
        features = self.segmentation_pred(features)
        segmentation = self.upsample(features)
        segmentation = segmentation.transpose(1, 2).transpose(2, 3).contiguous()
        out = torch.sigmoid(segmentation),
        return out

# if __name__ == "main":
#     drn = drn_d_54(pretrained=True, out_map=64, num_classes=-1, out_middle=False)
