import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """ X3D 论文中使用的 Swish (SiLU) 激活函数 """
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation (SE) Attention Block """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class X3D_Bottleneck(nn.Module):
    """
    X3D 的核心构建块：使用深度可分离 3D 卷积
    结构: 1x1x1 Conv -> 3x3x3 Depthwise Conv -> SE -> 1x1x1 Conv
    """
    def __init__(self, in_dim, out_dim, stride=1, expansion=2.25):
        super(X3D_Bottleneck, self).__init__()
        
        mid_dim = int(in_dim * expansion)
        self.use_residual = (stride == 1 and in_dim == out_dim)

        # 1. Pointwise Conv (Expansion)
        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_dim)
        self.act1 = Swish()

        # 2. Depthwise Conv (Spatial-Temporal)
        # groups=mid_dim 实现了深度可分离卷积
        self.conv2 = nn.Conv3d(mid_dim, mid_dim, kernel_size=3, stride=stride, padding=1, groups=mid_dim, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_dim)
        self.act2 = Swish()
        
        # 3. Squeeze-and-Excitation
        self.se = SEBlock(mid_dim)

        # 4. Pointwise Conv (Projection)
        self.conv3 = nn.Conv3d(mid_dim, out_dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_dim)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        out = self.se(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.use_residual:
            out += residual
            
        return out

class X3D_Stem(nn.Module):
    """ Video Stem: 处理输入的初始卷积 """
    def __init__(self, in_channels, out_channels):
        super(X3D_Stem, self).__init__()
        # 空间 stride=2, 时间(深度) stride=1 保持深度信息
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = Swish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class X3D_Classifier(nn.Module):
    """
    X3D (基于 X3D-S 配置修改)
    适合 3D 医学图像分类
    """
    def __init__(self, in_channels=3, num_classes=1, dropout_rate=0.5):
        super(X3D_Classifier, self).__init__()
        
        # 配置: X3D-S 风格的通道数配置
        # 这些数字经过精细调节以平衡 FLOPs 和精度
        widths = [24, 48, 96, 192] 
        self.in_channels = in_channels

        # 1. Stem
        self.stem = X3D_Stem(in_channels, widths[0])
        
        # 2. Stages (堆叠 Bottlenecks)
        self.layer1 = self._make_layer(widths[0], widths[0], blocks=2, stride=1) # Stride 1 preserves spatial
        self.layer2 = self._make_layer(widths[0], widths[1], blocks=3, stride=2)
        self.layer3 = self._make_layer(widths[1], widths[2], blocks=5, stride=2)
        self.layer4 = self._make_layer(widths[2], widths[3], blocks=2, stride=2)
        
        # 3. Post-processing Conv (Conv5)
        self.conv5 = nn.Conv3d(widths[3], widths[3], kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm3d(widths[3])
        self.act5 = Swish()
        
        # 4. Classification Head
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(widths[3], num_classes)
        # self.sigmoid = nn.Sigmoid() # 输出置信度
        
        self._init_weights()

    def _make_layer(self, in_dim, out_dim, blocks, stride):
        layers = []
        # 第一个 block 处理 stride (下采样) 和通道变化
        layers.append(X3D_Bottleneck(in_dim, out_dim, stride=stride))
        # 后续 block 保持维度
        for _ in range(1, blocks):
            layers.append(X3D_Bottleneck(out_dim, out_dim, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (Batch, 3, W, H, Z)
        # Adapt to PyTorch 3D standard: (Batch, C, D, H, W)
        # 这里的 Z (深度/切片) 对应 PyTorch 的 D 维度
        # x = x.permute(0, 1, 4, 3, 2) # (B, 3, Z, H, W)
        
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.act5(self.bn5(self.conv5(x)))
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        # x = self.sigmoid(x)
        
        return x

if __name__ == "__main__":
    # 模拟输入参数
    BATCH_SIZE = 2
    IN_CHANNELS = 3
    # X3D 非常灵活，但建议 H, W 为 32 的倍数
    W, H, Z = 64, 64, 64 
    
    model = X3D_Classifier(in_channels=IN_CHANNELS, num_classes=1)
    
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, W, H, Z)
    
    print("-" * 30)
    print("X3D (Efficient 3D ConvNet) Test Summary")
    print("-" * 30)
    
    try:
        # 计算参数量
        params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {params / 1e6:.2f}M (Very Efficient!)")
        
        output = model(input_tensor)
        print(f"Input Shape:  {input_tensor.shape}")
        print(f"Output Shape: {output.shape}")
        print(f"Output Values:\n {output.detach().numpy()}")
        
        assert output.shape == (BATCH_SIZE, 1), "Output shape mismatch!"
        assert output.min() >= 0 and output.max() <= 1, "Output values out of range [0, 1]"
        
        print("-" * 30)
        print("Test Passed: X3D model is ready.")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()