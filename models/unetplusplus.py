from typing import Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['UNet', 'NestedUNet']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, strides=(2, 2, 2, 2), dilations=(1, 1, 1, 1)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])

        self.inplanes = 1024

        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, 1000)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, dilation=1)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        # state_dict.pop('fc.weight')
        # state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
        print("model pretrained initialized")
    return model

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes
        if inplanes != planes:
            self.conv = nn.Conv2d(inplanes, planes, 1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.inplanes != self.planes:
            identity = self.conv(identity)
        out += identity
        out = self.relu(out)

        return out

class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

class MHCLS(nn.Module):
    def __init__(self, args) -> None:
        super(MHCLS, self).__init__()
        self.args = args
        self.num_heads = args.net_nheads
        self.num_classes = args.net_num_classes
        self.certainty = args.net_certainty
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.mh_classifiers = nn.ModuleList([nn.Sequential(
                                                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(16),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(in_channels=16, out_channels= self.num_classes, kernel_size=1)) for i in range(self.num_heads)])
        self.mh_vor_classifiers = nn.ModuleList([nn.Sequential(
                                                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(16),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(in_channels=16, out_channels= self.num_classes, kernel_size=1)) for i in range(self.num_heads)])
        self.mh_certainty = nn.ModuleList([nn.Sequential(
                                                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(16),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(in_channels=16, out_channels = 1, kernel_size=1)) for i in range(self.num_heads)])
        self.mh_regresser = nn.ModuleList([nn.Sequential(
                                                nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(16),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(in_channels=16, out_channels = 1, kernel_size=1)) for i in range(self.num_heads)])
        if self.args.degree_version in ['v4']:
            self.mh_degree = nn.ModuleList([nn.Sequential(
                                                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                                    nn.BatchNorm2d(128),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                                                    nn.BatchNorm2d(256),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv2d(in_channels=256, out_channels = (4*self.args.net_N+3), kernel_size=1)) for i in range(self.num_heads)])
        elif self.args.degree_version in ['v9', 'v10']:
            self.mh_degree = nn.ModuleList([nn.Sequential(
                                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                                    nn.BatchNorm2d(64),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                                    nn.BatchNorm2d(64),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv2d(in_channels=64, out_channels = 8, kernel_size=1)) for i in range(self.num_heads)])
        else:
            self.mh_degree = nn.ModuleList([nn.Sequential(
                                                    nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
                                                    nn.BatchNorm2d(16),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv2d(in_channels=16, out_channels = 1, kernel_size=1)) for i in range(self.num_heads)])
            
        self.mh_counter = nn.ModuleList([nn.Sequential(
                                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(16),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=16, out_channels = 1, kernel_size=1)) for i in range(self.num_heads)])
        
    def forward(self, x, deg_map):
        y = []
        vor = []
        cert = []
        heat = []
        deg = []
        count = []
        x = self.upsample(x)
        h,w = deg_map.shape[-2:]
        x = F.interpolate(x, (h,w), mode='bilinear')
        # x_deg = torch.concat((x, deg_map), dim=1)
            
        for ii in range(self.num_heads):
            y.append(self.mh_classifiers[ii](x))
            vor.append(self.mh_vor_classifiers[ii](x))
            cert.append(self.mh_certainty[ii](x))
            heat.append(self.mh_regresser[ii](x))
            deg.append(self.mh_degree[ii](x))
            count.append(self.mh_counter[ii](x))
        return y, vor, cert, heat, deg, count

class NestedUNet(nn.Module):
    def __init__(self, args):
        super(NestedUNet, self).__init__()
        self.args = args
        self.num_classes = args.net_num_classes
        self.deep_supervision = False
        nb_filter = [64, 256, 512, 1024, 2048]

        self.resnet50 = resnet50(pretrained=True, strides=(2,2,2,1), dilations=(1,1,2,2))

        self.conv0_0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1,self.resnet50.relu, self.resnet50.maxpool)
        self.conv1_0 = nn.Sequential(self.resnet50.layer1)
        self.conv2_0 = nn.Sequential(self.resnet50.layer2)
        self.conv3_0 = nn.Sequential(self.resnet50.layer3)
        self.conv4_0 = nn.Sequential(self.resnet50.layer4)

        # self.conv0_0 = BasicBlock(3, nb_filter[0])
        # self.conv1_0 = BasicBlock(nb_filter[0], nb_filter[1])
        # self.conv2_0 = BasicBlock(nb_filter[1], nb_filter[2])
        # self.conv3_0 = BasicBlock(nb_filter[2], nb_filter[3])
        # self.conv4_0 = BasicBlock(nb_filter[3], nb_filter[4])
        
        # self.conv1_0 = self._make_layer(BasicBlock, 2, nb_filter[0], nb_filter[1])
        # self.conv2_0 = self._make_layer(BasicBlock, 2, nb_filter[1], nb_filter[2])
        # self.conv3_0 = self._make_layer(BasicBlock, 2, nb_filter[2], nb_filter[3])
        # self.conv4_0 = self._make_layer(BasicBlock, 2, nb_filter[3], nb_filter[4])

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv0_1 = BasicBlock(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = BasicBlock(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = BasicBlock(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = BasicBlock(nb_filter[3]+nb_filter[4], nb_filter[3])
        
        # self.conv0_1 = self._make_layer(BasicBlock, 2, nb_filter[0]+nb_filter[1], nb_filter[0])
        # self.conv1_1 = self._make_layer(BasicBlock, 2, nb_filter[1]+nb_filter[2], nb_filter[1])
        # self.conv2_1 = self._make_layer(BasicBlock, 2, nb_filter[2]+nb_filter[3], nb_filter[2])
        # self.conv3_1 = self._make_layer(BasicBlock, 2, nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = BasicBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = BasicBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = BasicBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2])
        
        # self.conv0_2 = self._make_layer(BasicBlock, 2, nb_filter[0]*2+nb_filter[1], nb_filter[0])
        # self.conv1_2 = self._make_layer(BasicBlock, 2, nb_filter[1]*2+nb_filter[2], nb_filter[1])
        # self.conv2_2 = self._make_layer(BasicBlock, 2, nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = BasicBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = BasicBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1])
        
        # self.conv0_3 = self._make_layer(BasicBlock, 2, nb_filter[0]*3+nb_filter[1], nb_filter[0])
        # self.conv1_3 = self._make_layer(BasicBlock, 2, nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = BasicBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0])
        
        # self.conv0_4 = self._make_layer(BasicBlock, 2, nb_filter[0]*4+nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = MHCLS(self.args)
            self.final2 = MHCLS(self.args)
            self.final3 = MHCLS(self.args)
            self.final4 = MHCLS(self.args)

            self.pretrained = nn.ModuleList([self.conv0_0, self.conv1_0, self.conv2_0, self.conv3_0, self.conv4_0])
            self.new_added = nn.ModuleList([self.conv0_1, self.conv1_1, self.conv2_1, self.conv3_1,
                                            self.conv0_2, self.conv1_2, self.conv2_2,
                                            self.conv0_3, self.conv1_3,
                                            self.conv0_4,
                                            self.final1, self.final2, self.final3, self.final4])

        else:
            self.final = MHCLS(self.args)
    
            self.pretrained = nn.ModuleList([self.conv0_0, self.conv1_0, self.conv2_0, self.conv3_0, self.conv4_0])
            self.new_added = nn.ModuleList([self.conv0_1, self.conv1_1, self.conv2_1, self.conv3_1,
                                            self.conv0_2, self.conv1_2, self.conv2_2,
                                            self.conv0_3, self.conv1_3,
                                            self.conv0_4,
                                            self.final])

    def _make_layer(self, block, num_block, inplane, plane):
        layer_list = []
        for i in range(num_block):
            if i == 0:
                layer = block(inplane, plane)
            else:
                layer = block(plane, plane)
            layer_list.append(layer)
        layers = nn.Sequential(*layer_list)
        return layers

    def forward(self, input):
        deg_map = input[:,3:,:,:]
        input = input[:,:3,:,:]
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            y1, vor1, cert1, heat1, deg1, count1 = self.final1(x0_1,deg_map)
            y2, vor2, cert2, heat2, deg2, count2 = self.final2(x0_2,deg_map)
            y3, vor3, cert3, heat3, deg3, count3 = self.final3(x0_3,deg_map)
            y4, vor4, cert4, heat4, deg4, count4 = self.final4(x0_4,deg_map)
            y = y1 + y2 + y3 + y4
            vor = vor1 + vor2 + vor3 + vor4
            cert = cert1 + cert2 + cert3 + cert4
            heat = heat1 + heat2 + heat3 + heat4
            deg = deg1 + deg2 + deg3 + deg4
            count = count1 + count2 + count3 + count4
            return y, vor, cert, heat, deg, count4

        else:
            y, vor, cert, heat, deg, count = self.final(x0_4,deg_map)
            return y, vor, cert, heat, deg, count

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser("CellSeg training argument parser.")
    parser.add_argument('--net_stride', default=16, type=int)
    parser.add_argument('--net_num_classes', default=3, type=int)
    args = parser.parse_args()
    model = NestedUNet(args)
    x = torch.zeros([4,3,64,64])
    
    y = model(x)
    print(y[0].shape)