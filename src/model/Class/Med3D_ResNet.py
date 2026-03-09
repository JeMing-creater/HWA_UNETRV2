import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )

class BasicBlock(nn.Module):
    """
    ResNet-18 / ResNet-34 的基础模块
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    ResNet-50 / ResNet-101 / ResNet-152 的瓶颈模块
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=1):
        super(ResNet3D, self).__init__()
        self.inplanes = 64
        
        # 1. Stem Layer: 初始卷积层，快速下采样
        # 输入通道改为 in_channels (3)
        self.conv1 = nn.Conv3d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 2. Residual Layers (4 stages)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 3. Global Average Pooling & Classification Head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.sigmoid = nn.Sigmoid() # 二分类置信度激活

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, 3, W, H, Z)
        # 转换为 (Batch, 3, Z, H, W) 也就是 (B, C, D, H, W)
        x = x.permute(0, 1, 4, 3, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

def generate_model(model_depth=18, in_channels=3, num_classes=1):
    """
    工厂函数，用于生成不同深度的 ResNet
    Med3D 论文中主要使用了 ResNet-10, 18, 34, 50, 101, 152
    """
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet3D(BasicBlock, [1, 1, 1, 1], in_channels=in_channels, num_classes=num_classes)
    elif model_depth == 18:
        model = ResNet3D(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)
    elif model_depth == 34:
        model = ResNet3D(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)
    elif model_depth == 50:
        model = ResNet3D(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)
    elif model_depth == 101:
        model = ResNet3D(Bottleneck, [3, 4, 23, 3], in_channels=in_channels, num_classes=num_classes)
    
    return model

if __name__ == "__main__":
    # 配置
    BATCH_SIZE = 2
    IN_CHANNELS = 3
    W, H, Z = 128, 128, 64
    
    # 实例化模型 (默认使用 ResNet-18，这在 Med3D 中最常用)
    # 如果您想要更强的模型，可以将 model_depth 改为 50
    model = generate_model(model_depth=18, in_channels=IN_CHANNELS, num_classes=1)
    
    # 创建模拟输入
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, W, H, Z)
    
    print("-" * 30)
    print("Med3D (3D ResNet-18) Test Summary")
    print("-" * 30)
    
    # 前向传播测试
    try:
        output = model(input_tensor)
        print(f"Input Shape:  {input_tensor.shape}")
        print(f"Output Shape: {output.shape}")
        print(f"Output Values:\n {output.detach().numpy()}")
        
        # 验证输出
        assert output.shape == (BATCH_SIZE, 1), "Output shape mismatch!"
        # 验证数值范围 (Sigmoid 应该在 0-1 之间)
        assert output.min() >= 0 and output.max() <= 1, "Output values out of range [0, 1]"
        
        print("-" * 30)
        print("Test Passed: Med3D ResNet model is ready.")
        
    except Exception as e:
        print(f"Test Failed: {e}")