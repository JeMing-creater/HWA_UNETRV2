import torch
import torch.nn as nn
import math

class Conv3DBlock(nn.Module):
    """
    基础 3D 卷积块：Conv3d + BatchNorm + ReLU + MaxPool
    用于逐步提取特征并降低空间维度。
    """
    def __init__(self, in_channels, out_channels, pool=True):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

class Hybrid_CNN_Transformer(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        img_size=(128, 128, 64), 
        num_classes=1,
        cnn_channels=[32, 64, 128], # CNN 各层通道数
        embed_dim=256,              # Transformer 嵌入维度
        trans_depth=4,              # Transformer 层数
        num_heads=8,                # 注意力头数
        mlp_ratio=4.,
        dropout=0.1
    ):
        super(Hybrid_CNN_Transformer, self).__init__()
        
        # 1. 3D CNN Encoder (Feature Extractor)
        # ---------------------------------------------------
        self.cnn_encoder = nn.Sequential()
        curr_dim = in_channels
        for i, out_dim in enumerate(cnn_channels):
            self.cnn_encoder.add_module(
                f'block_{i}', 
                Conv3DBlock(curr_dim, out_dim, pool=True)
            )
            curr_dim = out_dim
            
        # 计算 CNN 输出后的特征图尺寸
        # 假设经过了 3 次 MaxPool (2x2x2)，尺寸缩小为原来的 1/8
        self.D_feat = img_size[2] // (2 ** len(cnn_channels))
        self.H_feat = img_size[1] // (2 ** len(cnn_channels))
        self.W_feat = img_size[0] // (2 ** len(cnn_channels))
        
        # 展平后的序列长度 (Sequence Length)
        self.seq_len = self.D_feat * self.H_feat * self.W_feat
        cnn_out_dim = cnn_channels[-1]
        
        # 2. Bridge: CNN to Transformer Projection
        # ---------------------------------------------------
        # 将 CNN 的特征通道映射到 Transformer 的 embed_dim
        self.proj = nn.Linear(cnn_out_dim, embed_dim)
        
        # 位置编码 (Positional Embedding) - Learnable
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder (Global Context Modeler)
        # ---------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-Norm 结构更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_depth)

        # 4. Classification Head
        # ---------------------------------------------------
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid() # 输出置信度
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input Check: (B, 3, W, H, Z)
        # Adapt to PyTorch Conv3d: (B, C, D, H, W) -> Assuming Z is Depth
        x = x.permute(0, 1, 4, 3, 2) # (B, 3, Z, H, W)
        
        B = x.shape[0]

        # 1. Local Feature Extraction (CNN)
        x = self.cnn_encoder(x) # Output: (B, 128, D', H', W')
        
        # 2. Sequence Flattening
        # (B, C, D', H', W') -> (B, C, N) -> (B, N, C)
        x = x.flatten(2).transpose(1, 2) 
        
        # 3. Projection & Embedding
        x = self.proj(x) # (B, Seq_Len, Embed_Dim)
        
        # Add CLS token (可选，这里使用 GAP，但保留 CLS 结构以备扩展)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embedding
        # 注意：这里需要处理 pad 后的 pos_embed 对齐，简单起见我们只取对应的部分
        # (Seq_Len + 1 for CLS token)
        if x.shape[1] <= self.pos_embed.shape[1] + 1:
             x = x + torch.cat((torch.zeros(1, 1, self.pos_embed.shape[2], device=x.device), self.pos_embed), dim=1)[:, :x.shape[1], :]
        
        x = self.dropout(x)

        # 4. Global Modeling (Transformer)
        x = self.transformer(x)

        # 5. Classification
        # 使用 CLS token 的输出，或者全局平均池化 (GAP)
        # 这里使用 GAP (Global Average Pooling) 融合所有 token 的信息，通常对分类更稳健
        x = x[:, 1:, :].mean(dim=1) # Exclude CLS token for averaging, or integrate it
        
        x = self.norm(x)
        x = self.classifier(x)
        
        return x

if __name__ == "__main__":
    # 配置
    BATCH_SIZE = 2
    IN_CHANNELS = 3
    W, H, Z = 128, 128, 64
    
    # 实例化模型
    # 注意：img_size 必须准确，用于计算 Transformer 的序列长度
    model = Hybrid_CNN_Transformer(
        in_channels=IN_CHANNELS, 
        img_size=(W, H, Z), 
        num_classes=1,
        embed_dim=256
    )
    
    # 模拟输入
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, W, H, Z)
    
    print("-" * 30)
    print("Hybrid 3D CNN-Transformer Model Test")
    print("-" * 30)
    
    try:
        output = model(input_tensor)
        
        print(f"Input Shape:  {input_tensor.shape}")
        print(f"Output Shape: {output.shape}")
        print(f"Output Values:\n {output.detach().numpy()}")
        
        # 验证
        assert output.shape == (BATCH_SIZE, 1), "Output shape mismatch!"
        assert output.min() >= 0 and output.max() <= 1, "Output values out of range [0, 1]"
        
        print("-" * 30)
        print("Test Passed: Hybrid model is ready.")
        
    except Exception as e:
        print(f"Test Failed: {e}")