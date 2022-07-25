import torch
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

'''
__all__ = ['FANet', 'fanet_lite_b0', 'fanet_lite_b1', 'fanet_lite_b2', 'fanet_b0', 'fanet_b1', 'fanet_b2']

model_urls = {
    'fanet_b0': 'https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/models/mtvision/fanet_b0.pth',
    'fanet_b1': 'https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/models/mtvision/fanet_b1.pth',
    'fanet_b2': 'https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/models/mtvision/fanet_b2.pth',
    'fanet_lite_b0': 'https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/models/mtvision/fanet_lite_b0.pth',
    'fanet_lite_b1': 'https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/models/mtvision/fanet_lite_b1.pth',
    'fanet_lite_b2': 'https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/models/mtvision/fanet_lite_b2.pth',
}
'''

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class HardSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp((x + 1) / 2, min=0, max=1)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.clamp((x + 1) / 2, min=0, max=1)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

class Act(nn.Module):
    def __init__(self, out_planes=None, act_type="relu"):
        super(Act, self).__init__()

        self.act = None
        if act_type == "relu":
            self.act = nn.ReLU(inplace=False)
        elif act_type == "prelu":
            self.act = nn.PReLU(out_planes)
        elif act_type == "swish":
            self.act = Swish()
        elif act_type == "hardswish":
            self.act = HardSwish()
        elif act_type == "mish":
            self.act = Mish()

    def forward(self, x):
        if self.act is not None:
            x = self.act(x)
        return x


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1, act_type="relu", ibn=False):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2, bias=False)
        if ibn:
            self.norm = IBN(out_planes)
        else:
            self.norm = nn.BatchNorm2d(out_planes)

        self.act = Act(out_planes, act_type)

    def forward(self, x):
        out = self.norm(self.conv(x))
        out = self.act(out)
        return out
        

class SqueezeExcitation(nn.Module):
    def __init__(self, inplanes, outplanes, reduce=0):
        super(SqueezeExcitation, self).__init__()
        if reduce == 0:
            self.context = nn.AdaptiveAvgPool2d(1)
            self.fusion = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )
        elif reduce > 0:
            self.context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=inplanes, out_channels=inplanes//reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(inplanes//reduce),
                nn.ReLU(inplace=False)
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(in_channels=inplanes//reduce, out_channels=outplanes, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.context(x)
        out = self.fusion(out)
        out = self.sigmoid(out)
        return out


def init_params(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if 'first' in name:
                nn.init.normal_(m.weight, 0, 0.01)
            else:
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if name.endswith("conv_out.bn.weight"):
                nn.init.constant_(m.weight, 0)
            else:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0001)
            nn.init.constant_(m.running_mean, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0001)
            nn.init.constant_(m.running_mean, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, stride=1, att=False, avd=False, avd_first=False, avd_dw=False, act_type="relu"):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.avd = avd and (self.stride > 1)
        self.avd_first = avd_first

        if self.avd:
            if avd_dw:
                self.avd_layer = nn.Sequential(
                    nn.Conv2d(in_channels=out_planes*groups//4, out_channels=out_planes*groups//4, kernel_size=5, groups=out_planes*groups//4, stride=2, padding=2, bias=False),
                    nn.BatchNorm2d(out_planes*groups//4)
                )
            else:
                self.avd_layer = nn.AvgPool2d(3, self.stride, padding=1)
            self.stride = 1

        self.conv_in = ConvX(in_planes, out_planes*groups//4, groups=1, kernel_size=1, stride=1, act_type="relu")
        self.conv = ConvX(out_planes*groups//4, out_planes*groups//4, groups=groups, kernel_size=3, stride=self.stride, act_type="relu")
        self.conv_out = ConvX(out_planes*groups//4, out_planes, groups=1, kernel_size=1, stride=1, act_type=None)
        self.act = Act(out_planes, act_type)

        self.att = None
        if att == "se":
            self.att = SqueezeExcitation(out_planes, out_planes, reduce=16)
        elif att == "slse":
            self.att = SqueezeExcitation(out_planes*groups//4, out_planes*groups//4, reduce=0)

        self.skip = None
        if stride == 1 and in_planes != out_planes:
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )

        if stride == 2 and in_planes != out_planes:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, groups=in_planes, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        skip = x

        out = self.conv_in(x)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        
        out = self.conv(out)
        
        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        if self.att is not None:
            mul = self.att(out)
        if self.att is not None:
            out = out * mul

        out = self.conv_out(out)

        if self.skip is not None:
            skip = self.skip(x)
        out += skip
        out = self.act(out)
        return out


@BACKBONES.register_module()
class FANet(BaseBackbone):
    """FANet Variants
    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, base_width, layers, stem, att_type=None, avd=True, avd_first=True, avd_dw=True, groups=2, act_type="relu", init_cfg=None, pretrained=None,):
        super(FANet, self).__init__(init_cfg)
        self.att_type = att_type
        self.avd = avd
        self.avd_first = avd_first
        self.avd_dw = avd_dw
        self.act_type = act_type

        if stem == "norm":
            self.first_conv = nn.Sequential(
                ConvX( 3, 32, 1, 3, 2),
                ConvX(32, 32, 1, 3, 1),
                ConvX(32, 64, 1, 3, 1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        elif stem == "lite":
            self.first_conv = nn.Sequential(
                ConvX( 3, 32, 1, 3, 2),
                ConvX(32, 32, 1, 3, 1),
                ConvX(32, 64, 1, 3, 2)
            )
        elif stem == "deep":
            self.first_conv = nn.Sequential(
                ConvX( 3, 32, 1, 3, 2),
                ConvX(32, 32, 1, 3, 1),
                ConvX(32, 64, 1, 3, 2),
                ConvX(64, 64, 1, 3, 1)
            )
        self.layer1 = self._make_layers(64, base_width*4, groups, layers[0], stride=2)
        self.layer2 = self._make_layers(base_width*4, base_width*8, groups, layers[1], stride=2)
        self.layer3 = self._make_layers(base_width*8, base_width*16, groups, layers[2], stride=2)

        # self.conv_last = ConvX(base_width*16, base_width*16, 1, 1)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        
        # self.fc = nn.Linear(base_width*16, base_width*16, bias=False)
        # self.bn = nn.BatchNorm1d(base_width*16)
        # self.act = Act(base_width*16, act_type=act_type)
        # self.drop = nn.Dropout(p=dropout)
        # self.classifier = nn.Linear(base_width*16, num_classes, bias=False)
        
    def init_weights(self):
        super(FANet, self).init_weights()
        
        if (isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained'):
            return

        init_params(self)


    def _make_layers(self, inputs, outputs, groups, num_block, stride):
        layers = [Bottleneck(inputs, outputs, groups, stride, self.att_type, self.avd, self.avd_first, self.avd_dw, self.act_type)]

        for _ in range(1, num_block):
            layers.append(Bottleneck(outputs, outputs, groups, 1, self.att_type, self.avd, self.avd_first, self.avd_dw, self.act_type))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        x = self.first_conv(x)
        # print(x.shape)
        outs.append(x)
        x = self.layer1(x)
        # print(x.shape)
        outs.append(x)
        x = self.layer2(x)
        # print(x.shape)
        outs.append(x)
        x = self.layer3(x)
        # print(x.shape)
        outs.append(x)
        # exit()
        # x = self.conv_last(x)
        # x = self.gap(x).flatten(1)
        # x = self.act(self.bn(self.fc(x)))
        # x = self.drop(x)
        # x = self.classifier(x)
        return tuple(outs) 


'''
def _fanet(arch, num_classes, base_width, layers, pretrained, progress, stem="lite", att_type=None, avd=True, avd_first=True, avd_dw=True, groups=2, act_type="relu", dropout=0.20):
    model = FANet(num_classes, base_width, layers, stem, att_type, avd, avd_first, avd_dw, groups, act_type, dropout)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def fanet_lite_b0(num_classes=1000, pretrained=False, progress=True):
    return _fanet('fanet_lite_b0', num_classes, 64, [3, 3, 2], pretrained, progress)

def fanet_lite_b1(num_classes=1000, pretrained=False, progress=True):
    return _fanet('fanet_lite_b1', num_classes, 96, [3, 6, 3], pretrained, progress)

def fanet_lite_b2(num_classes=1000, pretrained=False, progress=True):
    return _fanet('fanet_lite_b2', num_classes, 112, [8, 13, 3], pretrained, progress)

def fanet_b0(num_classes=1000, pretrained=False, progress=True):
    return _fanet('fanet_b0', num_classes, 64, [3, 3, 2], pretrained, progress, stem="deep", att_type="slse", avd_first=False)

def fanet_b1(num_classes=1000, pretrained=False, progress=True):
    return _fanet('fanet_b1', num_classes, 96, [3, 6, 3], pretrained, progress, stem="deep", att_type="slse", avd_first=False)

def fanet_b2(pretrained=False, progress=True):
    return _fanet('fanet_b2', 1000, 112, [8, 13, 3], pretrained, progress, stem="deep", att_type="slse", avd_first=False)
'''
