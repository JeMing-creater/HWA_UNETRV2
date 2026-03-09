from torch import nn
import torch
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class SENetDilatedBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(SENetDilatedBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=2,
            dilation=2,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.se = SELayer3D(planes * self.expansion, reduction=16)
        self.stride = stride

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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SENet3D(nn.Module):

    def __init__(self, block, layers, shortcut_type='B', cardinality=32, num_classes=2):
        self.inplanes = 64
        super(SENet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, cardinality, stride=(1, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, cardinality, stride=(1, 2, 2))
        self.layer4 = self._make_layer(SENetDilatedBottleneck, 256, layers[3], shortcut_type, cardinality, stride=1)
        # self.layer4 = self._make_layer(ResNeXtDilatedBottleneck, 512, layers[3], shortcut_type, cardinality, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
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

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.LeakyReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid())
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class SENetBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None, reduction=16):
        super(SENetBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.se = SELayer3D(planes * self.expansion, reduction)
        self.stride = stride

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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BackBone3D(nn.Module):
    def __init__(self):
        super(BackBone3D, self).__init__()
        net = SENet3D(SENetBottleneck, [3, 4, 6, 3], num_classes=2)
        # resnext3d-101 is [3, 4, 23, 3]
        # we use the resnet3d-50 with [3, 4, 6, 3] blocks
        # and if we use the resnet3d-101, change the block list with [3, 4, 23, 3]
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        # the layer0 contains the first convolution, bn and relu
        self.layer1 = nn.Sequential(*net[3:5])
        # the layer1 contains the first pooling and the first 3 bottle blocks
        self.layer2 = net[5]
        # the layer2 contains the second 4 bottle blocks
        self.layer3 = net[6]
        # the layer3 contains the media bottle blocks
        # with 6 in 50-layers and 23 in 101-layers
        self.layer4 = net[7]
        # the layer4 contains the final 3 bottle blocks
        # according the backbone the next is avg-pooling and dense with num classes uints
        # but we don't use the final two layers in backbone networks

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4



def add_conv3D(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv3d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('group_norm', nn.GroupNorm(16, out_ch))
    stage.add_module('leaky', nn.PReLU())
    return stage

class ASA3D(nn.Module):
    def __init__(self, level, vis=False):
        super(ASA3D, self).__init__()
        self.level = level
        compress_c = 16
        self.attention = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.pool = add_conv3D(64, 64, 3, (1, 2, 2))
        self.weight_level_0 = add_conv3D(64, compress_c, 1, 1)
        self.weight_level_1 = add_conv3D(64, compress_c, 1, 1)

        self.weight_levels = nn.Conv3d(compress_c*2, 2, kernel_size=1, stride=1, padding=0)
        self.vis = vis


    def forward(self, inputs0, inputs1):
        if self.level == 0:
            level_f = inputs1
        elif self.level == 1:
            level_f = self.pool(inputs1)
        elif self.level == 2:
            level_f0 = self.pool(inputs1)
            level_f = self.pool(level_f0)
        elif self.level == 3:
            level_f0 = self.pool(inputs1)
            level_f = self.pool(level_f0)

        level_0_weight_v = self.weight_level_0(inputs0)
        level_1_weight_v = self.weight_level_1(level_f)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        adaptive_attention = self.attention(inputs0) * inputs0 * levels_weight[:, 0:1, :, :, :] + \
                            self.attention(level_f) * level_f * levels_weight[:, 1:, :, :, :]

        # attention = self.attention(fused_out_reduced)
        out = self.refine(torch.cat((inputs0, adaptive_attention*level_f), 1))
        if self.vis:
            return out, levels_weight, adaptive_attention.sum(dim=1)
        else:
            return out
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class VAblock(nn.Module):
    """
    created by zyt
    """
    def __init__(self, num_channels, level, cardinality):
        super(VAblock, self).__init__()
        self.level = level
        if level == 1:
            self.l1_va_branch0 = nn.Sequential(nn.Conv3d(num_channels,num_channels,kernel_size=1),
                                               nn.Conv3d(num_channels,num_channels,kernel_size=3, stride=1, padding=1,dilation=1), nn.Sigmoid())
            self.l1_va_branch1 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3,padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1,padding=3, dilation=3), nn.Sigmoid())
            self.l1_va_branch2 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3,padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1,padding=5,dilation=5), nn.Sigmoid())
            self.l1_va_branch3 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=(5, 7, 7),dilation=(5, 7, 7)), nn.Sigmoid())
        elif level == 2:
            self.l2_va_branch0 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=1,dilation=1), nn.Sigmoid())
            self.l2_va_branch1 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=3,dilation=3), nn.Sigmoid())
            self.l2_va_branch2 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=5, dilation=5), nn.Sigmoid())
        elif level == 3:
            self.l3_va_branch0 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, dilation=1), nn.Sigmoid())
            self.l3_va_branch1 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3,padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=3, dilation=3), nn.Sigmoid())
        elif level == 4:
            self.l4_va_branch0 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, dilation=1), nn.Sigmoid())
            self.l4_va_branch1 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=3, dilation=3), nn.Sigmoid())
        self.fusion1 = nn.Sequential(
            nn.Conv3d(256, num_channels, kernel_size=1)
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Conv3d(192, num_channels, kernel_size=1)
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.fusion3 = nn.Sequential(
            nn.Conv3d(128, num_channels, kernel_size=1)
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.fusion4 = nn.Sequential(
            nn.Conv3d(128, num_channels, kernel_size=1)
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        # self.gn = nn.GroupNorm(32, 64)
        self.PReLU = nn.PReLU()

    def forward(self, x):
        if self.level == 1:
            x_l1_branch0 = self.l1_va_branch0(x) * x
            x_l1_branch1 = self.l1_va_branch1(x) * x
            x_l1_branch2 = self.l1_va_branch2(x) * x
            x_l1_branch3 = self.l1_va_branch3(x) * x
            x_out1 = self.fusion1(torch.cat((x_l1_branch0,x_l1_branch1,x_l1_branch2,x_l1_branch3),1))
            out = self.PReLU(x_out1 + x)

        elif self.level == 2:
            x_l2_branch0 = self.l2_va_branch0(x) * x
            x_l2_branch1 = self.l2_va_branch1(x) * x
            x_l2_branch2 = self.l2_va_branch2(x) * x
            x_out2 = self.fusion2(torch.cat((x_l2_branch0, x_l2_branch1, x_l2_branch2), 1))
            out = self.PReLU(x_out2 + x)

        elif self.level == 3:
            x_l3_branch0 = self.l3_va_branch0(x) * x
            x_l3_branch1 = self.l3_va_branch1(x) * x
            x_out3 = self.fusion3(torch.cat((x_l3_branch0, x_l3_branch1), 1))
            out = self.PReLU(x_out3 + x)

        elif self.level == 4:
            x_l4_branch0 = self.l4_va_branch0(x) * x
            x_l4_branch1 = self.l4_va_branch1(x) * x
            # print(x.size(),x_l4_branch0.size(),x_l4_branch1.size())
            x_out4 = self.fusion4(torch.cat((x_l4_branch0, x_l4_branch1), 1))
            out = self.PReLU(x_out4 + x)

        return out

def add_conv3D(in_ch, out_ch, ksize, stride):
        """
        Add a conv2d / batchnorm / leaky ReLU block.
        Args:
            in_ch (int): number of input channels of the convolution layer.
            out_ch (int): number of output channels of the convolution layer.
            ksize (int): kernel size of the convolution layer.
            stride (int): stride of the convolution layer.
        Returns:
            stage (Sequential) : Sequential layers composing a convolution block.
        """
        stage = nn.Sequential()
        pad = (ksize - 1) // 2
        stage.add_module('conv', nn.Conv3d(in_channels=in_ch,
                                           out_channels=out_ch, kernel_size=ksize, stride=stride,
                                           padding=pad, bias=False))
        stage.add_module('group_norm', nn.GroupNorm(16, out_ch))
        stage.add_module('leaky', nn.PReLU())
        return stage
class BiVA(nn.Module):
    """
    created by zyt
    """
    def __init__(self,num_channels, epsilon=1e-4, first_time=False, onnx_export=False, attention=True):
        super(BiVA,self).__init__()
        self.epsilon = epsilon
        self.first_time = first_time
        #conv layers
        self.conv_l1_down_channel = nn.Sequential(
            nn.Conv3d(128, num_channels, kernel_size=1),
            nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.conv_l2_down_channel = nn.Sequential(
            nn.Conv3d(256, num_channels, kernel_size=1),
            nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.conv_l3_down_channel = nn.Sequential(
            nn.Conv3d(512, num_channels, kernel_size=1),
            nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.conv_l4_down_channel = nn.Sequential(
            nn.Conv3d(512, num_channels, kernel_size=1),
            nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.VAblock1 = VAblock(num_channels, level=1, cardinality=32)
        self.VAblock2 = VAblock(num_channels, level=2, cardinality=32)
        self.VAblock3 = VAblock(num_channels, level=3, cardinality=32)
        self.VAblock4 = VAblock(num_channels, level=4, cardinality=32)

        self.pool1 = add_conv3D(num_channels, num_channels, 1, (1, 2, 2))
        self.pool2 = add_conv3D(num_channels, num_channels, 1, (1, 2, 2))

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # self.swish1 = MemoryEfficientSwish() if not onnx_export else Swish()
        # self.swish2 = MemoryEfficientSwish() if not onnx_export else Swish()
        # self.swish3 = MemoryEfficientSwish() if not onnx_export else Swish()
        # self.swish4 = MemoryEfficientSwish() if not onnx_export else Swish()

        # Weight
        self.F3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.F3_w1_relu = nn.PReLU()
        self.F2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.F2_w1_relu = nn.PReLU()
        self.F1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.F1_w1_relu = nn.PReLU()

        self.F1_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.F1_w2_relu = nn.PReLU()
        self.F2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.F2_w2_relu = nn.PReLU()
        self.F3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.F3_w2_relu = nn.PReLU()
        self.F4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.F4_w2_relu = nn.PReLU()


        self.attention = attention

    def forward(self, inputs1, inputs2, inputs3, inputs4):
        """
        illustration of a minimal bifpn unit
            F4_0 ------------>F4_1-------------> F4_2 -------->
               |-------------|                ↑
                             ↓                |
            F3_0 ---------> F3_1 ---------> F3_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            F2_0 ---------> F2_1 ---------> F2_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            F1_0 ---------> F1_1 ---------> F1_2 -------->
        """
        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            F1_out, F2_out, F3_out, F4_out = self._forward_fast_attention(inputs1, inputs2, inputs3, inputs4)
        else:
            F1_out, F2_out, F3_out, F4_out = self._forward(inputs1, inputs2, inputs3, inputs4)

        return F1_out, F2_out, F3_out, F4_out

    def _forward_fast_attention(self, inputs1, inputs2, inputs3, inputs4):

        if self.first_time:
            # Down channel
            inputs1 = self.conv_l1_down_channel(inputs1)
            inputs2 = self.conv_l2_down_channel(inputs2)
            inputs3 = self.conv_l3_down_channel(inputs3)
            inputs4 = self.conv_l4_down_channel(inputs4)
        else:
            inputs1=inputs1
            inputs2=inputs2
            inputs3=inputs3
            inputs4=inputs4

        # F4_0 to F4_1
        F4_1 = self.VAblock4(inputs4)

        # Weights for F3_0 and F4_1 to F3_1
        F3_w1 = self.F3_w1_relu(self.F3_w1)
        weight = F3_w1 / (torch.sum(F3_w1, dim=0) + self.epsilon)
        # Connections for F3_0 and F4_1 to F3_1 respectively
        F3_1 = self.VAblock3(self.swish(weight[0] * inputs3 + weight[1] * F4_1))

        # Weights for F2_0 and F3_1 to F2_1
        F2_w1 = self.F2_w1_relu(self.F2_w1)
        weight = F2_w1 / (torch.sum(F2_w1, dim=0) + self.epsilon)
        # Connections for F2_0 and F3_1 to F2_1 respectively
        F3_1_1 = F.upsample(F3_1, size=inputs2.size()[2:], mode='trilinear')
        F2_1 = self.VAblock2(self.swish(weight[0] * inputs2 + weight[1] * F3_1_1))

        # Weights for F1_0 and F2_1 to F1_1
        F1_w1 = self.F1_w1_relu(self.F1_w1)
        weight = F1_w1 / (torch.sum(F1_w1, dim=0) + self.epsilon)
        # Connections for F1_0 and F2_1 to F1_1 respectively
        F2_1_1 = F.upsample(F2_1, size=inputs1.size()[2:], mode='trilinear')
        F1_1 = self.VAblock1(self.swish(weight[0] * inputs1 + weight[1] * F2_1_1))


        # Weights for F1_0, F1_1 to F1_2
        F1_w2 = self.F1_w2_relu(self.F1_w2)
        weight = F1_w2 / (torch.sum(F1_w2, dim=0) + self.epsilon)
        # Connections for F1_0 and F1_1 to F1_2 respectively
        F1_2 = self.swish(weight[0] * inputs1 + weight[1] * F1_1)

        # Weights for F2_0, F2_1 and F1_2 to F2_2
        F2_w2 = self.F2_w2_relu(self.F2_w2)
        weight = F2_w2 / (torch.sum(F2_w2, dim=0) + self.epsilon)
        # Connections for F2_0, F2_1 and F1_2 to F2_2 respectively
        F1_2_1 = self.pool1(F1_2)
        F2_2 = self.swish(weight[0] * inputs2 + weight[1] * F2_1 + weight[2] * F1_2_1)

        # Weights for F3_0, F3_1 and F2_2 to F3_2
        F3_w2 = self.F3_w2_relu(self.F3_w2)
        weight = F3_w2 / (torch.sum(F3_w2, dim=0) + self.epsilon)
        # Connections for F3_0, F3_1 and F2_2 to F3_2 respectively
        F2_2_1 = self.pool2(F2_2)
        F3_2 = self.swish(weight[0] * inputs3 + weight[1] * F3_1 + weight[2] * F2_2_1)

        # Weights for F4_0 , F4_1 and F3_2 to F4_2
        F4_w2 = self.F4_w2_relu(self.F4_w2)
        weight = F4_w2 / (torch.sum(F4_w2, dim=0) + self.epsilon)
        # Connections for F4_0, F4_1 and F3_2 to F4_2
        F4_2 = self.swish(weight[0] * inputs4 + weight[1] * F4_1 + weight[2] * F3_2)

        return F1_2, F2_2, F3_2, F4_2

    def _forward(self, inputs1, inputs2, inputs3, inputs4):
        if self.first_time:
            # Down channel
            inputs1 = self.conv_l1_down_channel(inputs1)
            inputs2 = self.conv_l2_down_channel(inputs2)
            inputs3 = self.conv_l3_down_channel(inputs3)
            inputs4 = self.conv_l4_down_channel(inputs4)
        else:
            inputs1 = inputs1
            inputs2 = inputs2
            inputs3 = inputs3
            inputs4 = inputs4

        F4_1 = self.VAblock4(inputs4)

        # Connections for F3_0 and F4_1 to F3_1 respectively
        F3_1 = self.VAblock3(self.swish(inputs3 + F4_1))

        # Connections for F2_0 and F3_1 to F2_1 respectively
        F3_1_1 = F.upsample(F3_1, size=inputs2.size()[2:], mode='trilinear')
        F2_1 = self.VAblock2(self.swish(inputs2 + F3_1_1))

        # Connections for F1_0 and F2_1 to F1_1 respectively
        F2_1_1 = F.upsample(F2_1, size=inputs1.size()[2:], mode='trilinear')
        F1_1 = self.VAblock1(self.swish(inputs1 + F2_1_1))

        # Connections for F1_0 and F1_1 to F1_2 respectively
        F1_2 = self.swish(inputs1 + F1_1)

        # Connections for F2_0, F2_1 and F1_2 to F2_2 respectively
        F1_2_1 = self.pool1(F1_2)
        F2_2 = self.swish(inputs2 + F2_1 + F1_2_1)

        # Connections for F3_0, F3_1 and F2_2 to F3_2 respectively
        F2_2_1 = self.pool2(F2_2)
        F3_2 = self.swish(inputs3 + F3_1 + F2_2_1)

        # Connections for F4_0, F4_1 and F3_2 to F4_2
        F4_2 = self.swish(inputs4 + F4_1 + F3_2)

        return F1_2, F2_2, F3_2, F4_2


class MTLN3D(nn.Module):
    def __init__(self, vis=False):
        super(MTLN3D, self).__init__()
        # self.training = train
        self.backbone = BackBone3D()

        self.bivablock1 = BiVA(num_channels=64, first_time=True)
        # self.bivablock2 = BiVA(num_channels=64, first_time=False)

        self.ASA0 = ASA3D(level=0, vis=vis)
        self.ASA1 = ASA3D(level=1, vis=vis)
        self.ASA2 = ASA3D(level=2, vis=vis)
        self.ASA3 = ASA3D(level=3, vis=vis)

        self.fusion0 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )

        self.fusion1 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )

        ### segmentaion branch
        self.attention0 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=1), nn.Sigmoid()
        )
        self.conv0 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )

        self.attention1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=1), nn.Sigmoid()
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )

        self.predict_fuse0 = nn.Conv3d(64, 2, kernel_size=1)
        self.predict_fuse1 = nn.Conv3d(64, 2, kernel_size=1)

        self.predict = nn.Conv3d(64, 2, kernel_size=1)

        ### classification branch

        self.pool0 = add_conv3D(64, 64, 3, (1, 2, 2))
        self.attention2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=1), nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.pool1 = add_conv3D(64, 64, 3, (1, 2, 2))
        self.attention3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=1), nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, 2)
        # self.fc2 = nn.Linear(4096, 64)
        # self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        layer0 = self.backbone.layer0(x)
        layer1 = self.backbone.layer1(layer0)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)

        Scale1V, Scale2V, Scale3V, Scale4V = self.bivablock1(layer1, layer2,layer3,layer4)

        F20_upsample = F.upsample(Scale2V, size=layer1.size()[2:], mode='trilinear')
        F30_upsample = F.upsample(Scale3V, size=layer1.size()[2:], mode='trilinear')
        F40_upsample = F.upsample(Scale4V, size=layer1.size()[2:], mode='trilinear')

        fuse0 = self.fusion0(torch.cat((F40_upsample, F30_upsample, F20_upsample, Scale1V), 1))
        #
        Scale1A = self.ASA0(Scale1V, fuse0)
        Scale2A = self.ASA1(Scale2V, fuse0)
        Scale3A = self.ASA2(Scale3V, fuse0)
        Scale4A = self.ASA3(Scale4V, fuse0)

        F2_upsample = F.upsample(Scale2A, size=layer1.size()[2:], mode='trilinear')
        F3_upsample = F.upsample(Scale3A, size=layer1.size()[2:], mode='trilinear')
        F4_upsample = F.upsample(Scale4A, size=layer1.size()[2:], mode='trilinear')

        fuse1 = self.fusion1(torch.cat((F4_upsample, F3_upsample, F2_upsample, Scale1A), 1))

        ### segmentation branch
        out_F3_0 = torch.cat((Scale4A, Scale3A), 1)
        out_F3_1 = F.upsample(out_F3_0, size=Scale2A.size()[2:], mode='trilinear')

        out_F2_0 = torch.cat((out_F3_1, Scale2A), 1)
        out_F2_1 = self.conv0(self.attention0(out_F2_0) * Scale2A)
        out_F2_2 = F.upsample(out_F2_1, size=Scale1A.size()[2:], mode='trilinear')

        out_F1_0 = torch.cat((out_F2_2, Scale1A), 1)
        out_F1_1 = self.conv1(self.attention1(out_F1_0) * Scale1A)
        out_F1_2 = F.upsample(out_F1_1, size=x.size()[2:], mode='trilinear')

        ### classificication branch
        out_F10 = self.pool0(Scale1A)

        out_F20 = torch.cat((out_F10, Scale2A), 1)
        out_F21 = self.conv2(self.attention2(out_F20) * Scale2A)
        out_F22 = self.pool1(out_F21)

        out_F30 = torch.cat((out_F22, Scale3A), 1)
        out_F31 = self.conv3(self.attention3(out_F30) * Scale3A)

        out_F40 = torch.cat((out_F31, Scale4A), 1)

        fuse0 = F.upsample(fuse0, size=x.size()[2:], mode='trilinear')
        fuse1 = F.upsample(fuse1, size=x.size()[2:], mode='trilinear')

        seg_fuse0 = self.predict_fuse0(fuse0)
        seg_fuse1 = self.predict_fuse1(fuse1)

        seg_predict = self.predict(out_F1_2)

        class_predict1 = self.pool(out_F40)
        class_predict1 = class_predict1.view(class_predict1.size(0), -1)
        class_predict = self.fc(class_predict1)

        return seg_fuse0, seg_fuse1, seg_predict, class_predict


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MTLN3D().to(device)
    model.eval()

    x = torch.randn(2, 3, 128, 128, 64, device=device)

    with torch.no_grad():
        seg_fuse0, seg_fuse1, seg_predict, _ = model(x)

    print("seg_pred.shape =", seg_predict.shape)