from models.drn import drn_d_54
#from models.drn import drn_d_105
from torch import nn
from models.modules import *
from models.utils.utils import *


class Model(nn.Module):
    def __init__(self, options):
        super(Model, self).__init__()

        self.options = options
        self.drn = drn_d_54(pretrained=True, out_map=64, num_classes=-1, out_middle=False)
        self.pyramid = PyramidModule(options, 512, 128)
        self.feature_conv = ConvBlock(1024, 512)
        self.segmentation_pred = nn.Conv2d(512, NUM_CORNERS, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(options.height, options.width), mode='bilinear')
        return

    def forward(self, inp):
        features = self.drn(inp)
        features = self.pyramid(features)
        features = self.feature_conv(features)
        segmentation = self.upsample(self.segmentation_pred(features))
        segmentation = segmentation.transpose(1, 2).transpose(2, 3).contiguous()
        if self.options.method_type == 1:
            return torch.sigmoid(segmentation),
        else:
            return torch.sigmoid(segmentation[:, :, :, :NUM_CORNERS]), segmentation[:, :, :, NUM_CORNERS:NUM_CORNERS + NUM_ICONS + 2], segmentation[:, :, :, -(NUM_ROOMS + 2):]
