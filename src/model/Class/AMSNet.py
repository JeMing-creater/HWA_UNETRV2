import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    """
    3D Channel Attention Module (SE-Block style)
    通过学习每个通道的重要性权重，强化关键特征，抑制噪声。
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
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

class MSIB(nn.Module):
    """
    Multi-Scale Integration Block (MSIB)
    同时使用不同尺寸的卷积核提取特征，捕捉多尺度信息。
    """
    def __init__(self, in_channels, out_channels):
        super(MSIB, self).__init__()
        
        # 分支通道数分配 (为了保持输出通道数一致，这里做简单的分配)
        inter_channels = out_channels // 4
        
        # Branch 1: 1x1x1 Convolution
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3x3 Convolution
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 5x5x5 Convolution (用两个3x3代替，减少参数量且增加非线性，效果等同于5x5)
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合后的 1x1 卷积，调整回目标通道数
        # 3个分支 concat 后通道数为 inter_channels * 3
        self.fusion = nn.Sequential(
            nn.Conv3d(inter_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        out = torch.cat((x1, x2, x3), dim=1) # Channel-wise concatenation
        out = self.fusion(out)
        return out

class AMSNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(AMSNet, self).__init__()
        
        # 1. 初始特征提取
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # 2. 堆叠 MSIB + Attention 模块
        # Stage 1
        self.msib1 = MSIB(32, 64)
        self.att1 = ChannelAttention3D(64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Stage 2
        self.msib2 = MSIB(64, 128)
        self.att2 = ChannelAttention3D(128)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Stage 3
        self.msib3 = MSIB(128, 256)
        self.att3 = ChannelAttention3D(256)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 3. 分类头
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
            nn.Sigmoid() # 输出置信度
        )
        
        self._init_weights()

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
        # x shape: (B, 3, W, H, Z)
        # Permute to (B, C, Z, H, W) -> 这里假设 Z 是深度 D
        x = x.permute(0, 1, 4, 3, 2)
        
        # Layer 1
        x = self.conv1(x)
        
        # Block 1
        x = self.msib1(x)
        x = self.att1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.msib2(x)
        x = self.att2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.msib3(x)
        x = self.att3(x)
        x = self.pool3(x)
        
        # Classification
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x

if __name__ == "__main__":
    # 配置参数
    BATCH_SIZE = 2
    IN_CHANNELS = 3
    W, H, Z = 128, 128, 64
    
    # 实例化模型
    model = AMSNet(in_channels=IN_CHANNELS, num_classes=1)
    
    # 模拟输入数据 (B, 3, W, H, Z)
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, W, H, Z)
    
    print("-" * 30)
    print("AMSNet (Attention Multi-Scale Network) Test")
    print("-" * 30)
    
    try:
        # 前向传播
        output = model(input_tensor)
        
        print(f"Input Shape:  {input_tensor.shape}")
        print(f"Output Shape: {output.shape}")
        print(f"Output Values:\n {output.detach().numpy()}")
        
        # 验证
        assert output.shape == (BATCH_SIZE, 1), "Output shape mismatch!"
        assert output.min() >= 0 and output.max() <= 1, "Output values out of range [0, 1]"
        
        print("-" * 30)
        print("Test Passed: AMSNet model is ready.")
        
    except Exception as e:
        print(f"Test Failed: {e}")