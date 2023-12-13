import timm
from pprint import pprint
import torch
import torch.nn as nn
from modules import ConvBlock, AttentionBlock
import torch.nn.functional as F

from options import parse_args
from utils import NUM_CORNERS, NUM_ICONS, NUM_ROOMS
import os


class MyModelV3(nn.Module):
    def __init__(self, options, backbone='convnext_base.fb_in1k', pretrained_cfg_overlay=None):
        super().__init__()
        
        self.options = options
        self.ConvNeXt = timm.create_model(
            backbone,
            pretrained=True,
            pretrained_cfg_overlay=dict(
                file='./checkpoint/convnext_base_1k_224_ema.pth' if pretrained_cfg_overlay is None else pretrained_cfg_overlay
            )
        )
        self.feature_conv = ConvBlock(1024, 512)
        self.segmentation_pred = torch.nn.Conv2d(512, NUM_CORNERS, kernel_size=1)
        self.extractor = SE(1024)
        self.fpn = FPNv3()
        self.upsample = torch.nn.Upsample(size=(options.height, options.width), mode='bilinear')
    
    def forward(self, x):
        _, _, H, W = x.shape
        feature = self.ConvNeXt.forward_features(x)
        extract_feature = self.extractor(feature)
        channel_resize_out = self.feature_conv(extract_feature)  # 512 16 16
        
        fpn_out = self.fpn(channel_resize_out)
        channel_resize_out = self.feature_conv(fpn_out)  # 继续降维
        segmentation_out = self.segmentation_pred(channel_resize_out)
        
        heatmap = F.interpolate(segmentation_out, size=(H, W), mode="bilinear")  # 上采样到正常图片大小
        segmentation = heatmap.transpose(1, 2).transpose(2, 3).contiguous()
        return torch.sigmoid(segmentation),


class SE(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))  # 期望生成1*1的图片
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        #   首先是 Squeeze 操作，从空间维度来进行特征压缩，将h*w*c的特征变成一个1*1*c的特征，得到向量某种程度上具有全域性的感受野，
        #   并且输出的通道数和输入的特征通道数相匹配，它表示在特征通道上响应的全域性分布。算法很简单，就是一个全局平均池化。
        out = self.squeeze(x)
        
        # 其次是 Excitation 操作，通过引入 w 参数来为每个特征通道生成权重，其中 w 就是一个多层感知器，是可学习的，中间经过一个降维，减少参数量。并通过一个 Sigmoid 函数获得 0~1 之间归一化的权重，完成显式地建模特征通道间的相关性。
        out = self.compress(out)
        out = F.leaky_relu(out)
        out = self.excitation(out)
        coefficient = torch.sigmoid(out)
        return x * coefficient


class FPNv3(nn.Module):
    def __init__(self, in_planes=512, middle_planes=256, scales=None):
        super().__init__()
        
        self.conv_1 = ConvBlock(in_planes, middle_planes, kernel_size=3, padding=1)
        self.conv_up = ConvBlock(middle_planes, in_planes, 1)
        self.conv_mid = ConvBlock(middle_planes, middle_planes, kernel_size=3, stride=1, padding=1)
        self.conv_1_1 = ConvBlock(middle_planes, middle_planes, kernel_size=1)
        self.convtop_1_1 = ConvBlock(in_planes, in_planes, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.attention = AttentionBlock(middle_planes, middle_planes)
    
    def forward(self, backbone_out):
        _, _, H, W = backbone_out.shape
        top = backbone_out
        
        # 从大到小
        c1 = self.pool(self.attention(self.conv_1(top)))  # 256 8 8
        
        c2 = self.pool(self.attention(self.conv_mid(c1)))  # 256 4 4
        
        p2 = self.conv_1_1(c2)  # 256 4 4
        
        _, _, H_c1, W_c1 = c1.shape
        p1 = F.interpolate(p2, size=(H_c1, W_c1), mode='bilinear') + self.conv_1_1(c1)  # 256 8 8
        
        up_p1 = self.conv_up(p1)
        p0 = F.interpolate(up_p1, size=(H, W), mode='bilinear') + self.convtop_1_1(top)
        
        resized_p2 = F.interpolate(self.conv_1_1(p2), size=(H, W), mode="bilinear")
        resized_p1 = F.interpolate(self.conv_1_1(p1), size=(H, W), mode="bilinear")
        resized_p0 = F.interpolate(self.convtop_1_1(p0), size=(H, W), mode="bilinear")
        out = torch.cat([resized_p0, resized_p1, resized_p2], dim=1)  # 1,1024,64,64
        
        return out


class MyModelv4(nn.Module):
    def __init__(self, options, backbone='convnext_base.fb_in22k', pretrained_cfg_overlay_path=None):
        super().__init__()
        
        self.options = options
        self.ConvNeXt = timm.create_model(
            backbone,
            pretrained=True,
            pretrained_cfg_overlay=dict(
                file='./checkpoint/convnext_base_22k_224.pth' if pretrained_cfg_overlay_path is None else pretrained_cfg_overlay_path
            ),
            features_only=True
        )
        self.feature_conv = ConvBlock(1024, 512)
        self.segmentation_pred = nn.Conv2d(512, NUM_CORNERS, kernel_size=1)
        
        self.latlayer_1 = nn.Conv2d(256, 256, kernel_size=1)
        self.latlayer_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.latlayer_3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.smooth_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.se_1 = SE(256)
        self.se_2 = SE(512)
        self.se_3 = SE(1024)
        
        self._init_weights()
    
    def _init_weights(self):
        # nn.init.kaiming_normal_(self.feature_conv.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.segmentation_pred.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.smooth_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.smooth_2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.smooth_3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.latlayer_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.latlayer_2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.latlayer_3.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        _, _, H, W = x.shape
        # backbone
        features = self.ConvNeXt(x)
        
        # FPN
        features = features[1:]
        feature1 = features[0]
        # print(feature1.shape)
        feature2 = features[1]
        # print(feature2.shape)
        feature3 = features[2]
        # print(feature3.shape)
        # p3 = self.smooth(self.latlayer_3(self.se_3(feature3)))
        # p2 = self.latlayer_2(self.se_2(feature2)) + F.interpolate(
        #     p3, size=(feature2.shape[2], feature2.shape[3]), mode='bilinear'
        # )
        # p2 = self.smooth(p2)
        # p1 = self.latlayer_1(self.se_1(feature1)) + F.interpolate(
        #     p2, size=(feature1.shape[2], feature1.shape[3]), mode='bilinear'
        # )
        # p1 = self.smooth(p1)
        # FPN缺陷
        p3 = self.smooth_3(self.latlayer_3(self.se_3(feature3)))
        p2 = self.latlayer_2(self.se_2(feature2)) + F.interpolate(
            p3, size=(feature2.shape[2], feature2.shape[3]), mode='bilinear'
        )
        p2 = self.smooth_2(p2)
        p1 = self.latlayer_1(self.se_1(feature1)) + F.interpolate(
            p2, size=(feature1.shape[2], feature1.shape[3]), mode='bilinear'
        )
        p1 = self.smooth_1(p1)
        
        rp3 = F.interpolate(p3, size=(feature1.shape[2], feature1.shape[3]), mode='bilinear')
        rp2 = F.interpolate(p2, size=(feature1.shape[2], feature1.shape[3]), mode='bilinear')
        
        fpn_out = torch.cat([feature1, p1, rp2, rp3], dim=1)
        
        # feature conv and head
        feat = self.feature_conv(fpn_out)
        segmentation_out = self.segmentation_pred(feat)
        heatmap = F.interpolate(segmentation_out, size=(H, W), mode="bilinear")  # 上采样到正常图片大小
        segmentation = heatmap.transpose(1, 2).transpose(2, 3).contiguous()
        if self.options.method_type == 1:
            return torch.sigmoid(segmentation),
        
        else:
            return (
                torch.sigmoid(segmentation[:, :, :, :NUM_CORNERS]),
                segmentation[:, :, :, NUM_CORNERS:NUM_CORNERS + NUM_ICONS + 2],
                segmentation[:, :, :, -(NUM_ROOMS + 2):]
            )
        # extract_feature = self.extractor(feature)
        # channel_resize_out = self.feature_conv(extract_feature)  # 512 16 16
        #
        # fpn_out = self.fpn(channel_resize_out)
        # channel_resize_out = self.feature_conv(fpn_out)  # 继续降维
        # segmentation_out = self.segmentation_pred(channel_resize_out)
        #
        # heatmap = F.interpolate(segmentation_out, size=(H, W), mode="bilinear")  # 上采样到正常图片大小
        # segmentation = heatmap.transpose(1, 2).transpose(2, 3).contiguous()
        # return torch.sigmoid(segmentation),


class MyModelv5(nn.Module):
    def __init__(self, options, backbone='convnext_base.fb_in22k', pretrained_cfg_overlay_path=None):
        super().__init__()
        
        self.options = options
        self.ConvNeXt = timm.create_model(
            backbone,
            pretrained=True,
            pretrained_cfg_overlay=dict(
                file='./checkpoint/convnext_base_22k_224.pth' if pretrained_cfg_overlay_path is None else pretrained_cfg_overlay_path
            ),
            features_only=True
        )
        self.feature_conv = ConvBlock(1024, 512)
        self.segmentation_pred = nn.Conv2d(512, NUM_CORNERS, kernel_size=1)
        
        self.latlayer_1 = nn.Conv2d(256, 256, kernel_size=1)
        self.latlayer_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.latlayer_3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.smooth_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.se_1 = SE(256)
        self.se_2 = SE(512)
        self.se_3 = SE(1024)
        
        self._init_weights()
    
    def _init_weights(self):
        # nn.init.kaiming_normal_(self.feature_conv.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.segmentation_pred.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.smooth_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.smooth_2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.smooth_3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.latlayer_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.latlayer_2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.latlayer_3.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        _, _, H, W = x.shape
        # backbone
        features = self.ConvNeXt(x)
        
        # FPN
        features = features[1:]
        feature1 = features[0]
        # print(feature1.shape)
        feature2 = features[1]
        # print(feature2.shape)
        feature3 = features[2]
        # print(feature3.shape)
        # p3 = self.smooth(self.latlayer_3(self.se_3(feature3)))
        # p2 = self.latlayer_2(self.se_2(feature2)) + F.interpolate(
        #     p3, size=(feature2.shape[2], feature2.shape[3]), mode='bilinear'
        # )
        # p2 = self.smooth(p2)
        # p1 = self.latlayer_1(self.se_1(feature1)) + F.interpolate(
        #     p2, size=(feature1.shape[2], feature1.shape[3]), mode='bilinear'
        # )
        # p1 = self.smooth(p1)
        
        p3 = self.latlayer_3(self.se_3(feature3))
        
        p2 = self.latlayer_2(self.se_2(feature2)) + F.interpolate(
            p3, size=(feature2.shape[2], feature2.shape[3]), mode='bilinear'
        )
        
        p1 = self.latlayer_1(self.se_1(feature1)) + F.interpolate(
            p2, size=(feature1.shape[2], feature1.shape[3]), mode='bilinear'
        )
        # 除上采样产生的混叠效应(aliasing effect)
        p3_out = self.smooth_3(p3)
        p2_out = self.smooth_2(p2)
        p1_out = self.smooth_1(p1)
        
        resized_p3_out = F.interpolate(p3_out, size=(feature1.shape[2], feature1.shape[3]), mode='bilinear')
        resized_p2_out = F.interpolate(p2_out, size=(feature1.shape[2], feature1.shape[3]), mode='bilinear')
        
        fpn_out = torch.cat([feature1, p1_out, resized_p2_out, resized_p3_out], dim=1)
        
        # feature conv and head
        feat = self.feature_conv(fpn_out)
        segmentation_out = self.segmentation_pred(feat)
        heatmap = F.interpolate(segmentation_out, size=(H, W), mode="bilinear")  # 上采样到正常图片大小
        segmentation = heatmap.transpose(1, 2).transpose(2, 3).contiguous()
        if self.options.method_type == 1:
            return torch.sigmoid(segmentation),
        
        else:
            return (
                torch.sigmoid(segmentation[:, :, :, :NUM_CORNERS]),
                segmentation[:, :, :, NUM_CORNERS:NUM_CORNERS + NUM_ICONS + 2],
                segmentation[:, :, :, -(NUM_ROOMS + 2):]
            )
        # extract_feature = self.extractor(feature)
        # channel_resize_out = self.feature_conv(extract_feature)  # 512 16 16
        #
        # fpn_out = self.fpn(channel_resize_out)
        # channel_resize_out = self.feature_conv(fpn_out)  # 继续降维
        # segmentation_out = self.segmentation_pred(channel_resize_out)
        #
        # heatmap = F.interpolate(segmentation_out, size=(H, W), mode="bilinear")  # 上采样到正常图片大小
        # segmentation = heatmap.transpose(1, 2).transpose(2, 3).contiguous()
        # return torch.sigmoid(segmentation),


if __name__ == "__main__":
    # model_names = timm.list_models('convnext*', pretrained=True)
    # pprint(model_names)
    
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
    args.numEpochs = 200
    
    # test
    # ms = timm.list_models('convnext_base*', pretrained=True)
    # pprint(ms)
    # convnext = timm.create_model('convnext_base.fb_in22k')
    # pprint(convnext.default_cfg)
    x = torch.randn(1, 3, 512, 512)
    
    v4 = MyModelv4(
        args,
    )
    print(v4(x)[0].shape)
    
    # t2 = torch.rand(1, 1024, 16, 16)
    # mm = MyModelV3(args)
    # # fe = SE(1024)
    # print(mm(x)[0].shape)  # torch.Size([1, 512, 16, 16])
    
    # to_fpn = torch.rand(1, 512, 16, 16)
    # nfpn = FPN()
    # out = nfpn(to_fpn)
    # print(out.shape)
    
    # model = timm.create_model(
    #     'convnext_base.fb_in1k',
    #     pretrained=True,
    #     features_only=True,
    #     pretrained_cfg_overlay=dict(file='./checkpoint/convnext_base_1k_224_ema.pth')
    # )
    # # pprint(model.default_cfg)
    # model.eval()
    # pprint(f'Feature channels: {model.feature_info.channels()}')
    # pprint(f'Feature channels: {model.feature_info}')
    #
    # features = model(x)
    # for x in features:
    #     print(x.shape)
    #     """
    #     torch.Size([1, 128, 128, 128])
    #     torch.Size([1, 256, 64, 64]
    #     torch.Size([1, 512, 32, 32]
    #     torch.Size([1, 1024, 16, 16])
    #     """
