import torch
import torch.nn as nn
import math

class Conv3DBlock(nn.Module):
    """
    基础 3D 卷积块
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
        in_channels=1,              # 默认为 1
        img_size=(64, 64, 64),      # 默认为 64
        num_classes=1,
        cnn_channels=[32, 64, 128], # CNN 通道配置
        embed_dim=256,
        trans_depth=4,
        num_heads=8,
        mlp_ratio=4.,
        dropout=0.1
    ):
        super(Hybrid_CNN_Transformer, self).__init__()
        
        # 1. CNN Encoder
        self.cnn_encoder = nn.Sequential()
        curr_dim = in_channels
        for i, out_dim in enumerate(cnn_channels):
            self.cnn_encoder.add_module(
                f'block_{i}', 
                Conv3DBlock(curr_dim, out_dim, pool=True)
            )
            curr_dim = out_dim
            
        # 计算经过 CNN 下采样后的尺寸 (假设每层都做 2x2x2 Pooling)
        downsample_factor = 2 ** len(cnn_channels)
        self.D_feat = img_size[2] // downsample_factor
        self.H_feat = img_size[1] // downsample_factor
        self.W_feat = img_size[0] // downsample_factor
        
        # 序列长度
        self.seq_len = self.D_feat * self.H_feat * self.W_feat
        cnn_out_dim = cnn_channels[-1]
        
        # 2. Projection
        self.proj = nn.Linear(cnn_out_dim, embed_dim)
        
        # 位置编码 (包含 CLS token 的位置)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # 3. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_depth)

        # 4. Classifier Head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            # 【重要修改】删除了这里的 Sigmoid()
            # 因为训练用的 BCEWithLogitsLoss 自带 Sigmoid
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
        # Input: (B, C, D, H, W)
        
        # 【重要修改】删除了 permute，保持原始顺序
        
        B = x.shape[0]

        # 1. CNN
        x = self.cnn_encoder(x) 
        # Output shape: (B, 128, D', H', W')
        
        # 2. Flatten & Transpose
        # (B, C, D', H', W') -> (B, C, N) -> (B, N, C)
        x = x.flatten(2).transpose(1, 2) 
        
        # 3. Projection
        x = self.proj(x) 
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embedding
        # 简单对齐，防止尺寸不匹配报错
        if x.shape[1] <= self.pos_embed.shape[1]:
             x = x + self.pos_embed[:, :x.shape[1], :]
        else:
            # 如果输入尺寸比预想的大，就插值 (简单处理则截断，严谨处理应用插值)
            x = x + self.pos_embed[:, :x.shape[1], :]

        x = self.dropout(x)

        # 4. Transformer
        x = self.transformer(x)

        # 5. Classification (Use CLS token)
        # 取第 0 个 token (CLS) 进行分类，或者用 Global Average
        x = x[:, 0, :] 
        
        x = self.norm(x)
        x = self.classifier(x) # 输出 Logits
        
        return x