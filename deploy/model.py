import torch
from torch import nn
from drn import drn_d_54
# from models.drn import drn_d_105
from modules import ConvBlock, PyramidModule, FPN
from utils import NUM_CORNERS, NUM_ROOMS, NUM_ICONS
from options import parse_args
import os


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
            # 这个看起来是将wall和opening角点给二分类了,分成3个,第一部分是关键点,第二部分是icon,第三部分是room的语义分割
            return (
                torch.sigmoid(segmentation[:, :, :, :NUM_CORNERS]),
                segmentation[:, :, :, NUM_CORNERS:NUM_CORNERS + NUM_ICONS + 2],
                segmentation[:, :, :, -(NUM_ROOMS + 2):]
            )


if __name__ == "__main__":
    args = parse_args()
    
    args.keyname = 'floorplan'
    args.restore = 1  # restore是0表示从头开始训练，1表示从上一次训练的模型继续训练
    cur_folder_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_folder_path = os.path.join(cur_folder_path, "checkpoint")
    args.checkpoint_dir = 'checkpoint/'
    args.checkpoint_dir = checkpoint_folder_path
    args.test_dir = 'test/' + args.keyname
    args.model_type = 1
    args.batchSize = 4
    args.outputWidth = 512
    args.outputHeight = 512
    args.restore = 0
    args.numEpochs = 500
    args.logDir = 'log/'
    args.showGraphInTensorboard = 1
    
    inp = torch.randn(1, 3, 512, 512)
    drn = drn_d_54(pretrained=True, out_map=64, num_classes=-1, out_middle=False)
    o = drn(inp)
    pyramid = PyramidModule(args, 512, 128)
    print(pyramid(o).shape)
