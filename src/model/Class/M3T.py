import torch
import torch.nn as nn
import math

class Conv3DStem(nn.Module):
    """
    3D CNN Stem: 用于提取浅层3D特征并进行下采样。
    这一步将高维的3D图像压缩为较小的特征体 (Feature Volume)。
    """
    def __init__(self, in_channels=3, out_channels=64):
        super(Conv3DStem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2), # 下采样 1/2
            
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2), # 下采样 1/4
        )

    def forward(self, x):
        return self.stem(x)

class M3T(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        img_size=(128, 128, 64), 
        num_classes=1, 
        embed_dim=768, 
        depth=6, 
        num_heads=8, 
        mlp_ratio=4.0, 
        dropout=0.1
    ):
        """
        M3T: Multi-Plane and Multi-Slice Transformer implementation.
        
        Args:
            in_channels (int): 输入模态数 (默认 3)
            img_size (tuple): 输入图像尺寸 (W, H, Z)
            num_classes (int): 输出类别数 (默认 1)
            embed_dim (int): Transformer 的嵌入维度
        """
        super(M3T, self).__init__()
        
        # 1. 3D CNN 浅层特征提取
        cnn_out_channels = 64
        self.cnn_stem = Conv3DStem(in_channels, cnn_out_channels)
        
        # 计算经过 CNN Stem 后的特征图尺寸 (假设下采样了 4 倍)
        # 注意：输入顺序是 (W, H, Z)，但 PyTorch Conv3d 通常处理 (D, H, W)
        # 这里我们统一内部处理为 (D, H, W) = (Z, H, W)
        self.D_feat = img_size[2] // 4
        self.H_feat = img_size[1] // 4
        self.W_feat = img_size[0] // 4
        
        # 计算每个切片展平后的维度: channels * height * width (针对某一视图)
        # Axial Slice (D-axis): shape (C, H, W) -> flattened
        self.feat_dim_axial = cnn_out_channels * self.H_feat * self.W_feat
        # Coronal Slice (H-axis): shape (C, D, W) -> flattened
        self.feat_dim_coronal = cnn_out_channels * self.D_feat * self.W_feat
        # Sagittal Slice (W-axis): shape (C, D, H) -> flattened
        self.feat_dim_sagittal = cnn_out_channels * self.D_feat * self.H_feat

        # 2. 投影层 (Projection): 将不同视角的 2D 切片映射到 Transformer 的 embed_dim
        self.proj_axial = nn.Linear(self.feat_dim_axial, embed_dim)
        self.proj_coronal = nn.Linear(self.feat_dim_coronal, embed_dim)
        self.proj_sagittal = nn.Linear(self.feat_dim_sagittal, embed_dim)

        # 3. Class Token 和 Position Embeddings
        # 序列总长度 = D切片数 + H切片数 + W切片数 + 1(CLS token)
        total_seq_len = self.D_feat + self.H_feat + self.W_feat + 1
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, total_seq_len, embed_dim))
        
        # 平面编码 (Plane Embeddings): 区分该 Token 属于哪个视角 (Axial, Coronal, Sagittal)
        self.plane_embed_axial = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.plane_embed_coronal = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.plane_embed_sagittal = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_drop = nn.Dropout(p=dropout)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 5. MLP Head (Classification)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        # self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.plane_embed_axial, std=.02)
        nn.init.trunc_normal_(self.plane_embed_coronal, std=.02)
        nn.init.trunc_normal_(self.plane_embed_sagittal, std=.02)
        self.apply(self._init_weights_layer)

    def _init_weights_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x shape: (B, 3, W, H, Z)
        # 调整维度以适配 Conv3d (B, C, D, H, W) -> 这里我们将 Z 视为 D
        # x = x.permute(0, 1, 4, 3, 2) # (B, 3, Z, H, W)
        
        B = x.shape[0]
        
        # 1. 3D CNN Feature Extraction
        features = self.cnn_stem(x) # Output: (B, C=64, D', H', W')
        
        # 2. Multi-Plane Slicing & Projection
        # -----------------------------------------------------------
        # View 1: Axial (沿 D 轴切片) -> 序列长度为 D'
        # shape: (B, C, D, H, W) -> permute to (B, D, C, H, W) -> flatten -> (B, D, C*H*W)
        x_axial = features.permute(0, 2, 1, 3, 4).flatten(2)
        x_axial = self.proj_axial(x_axial) # (B, D', embed_dim)
        x_axial = x_axial + self.plane_embed_axial # Add Plane Embedding

        # View 2: Coronal (沿 H 轴切片) -> 序列长度为 H'
        # shape: (B, C, D, H, W) -> permute to (B, H, C, D, W) -> flatten -> (B, H, C*D*W)
        x_coronal = features.permute(0, 3, 1, 2, 4).flatten(2)
        x_coronal = self.proj_coronal(x_coronal) # (B, H', embed_dim)
        x_coronal = x_coronal + self.plane_embed_coronal # Add Plane Embedding

        # View 3: Sagittal (沿 W 轴切片) -> 序列长度为 W'
        # shape: (B, C, D, H, W) -> permute to (B, W, C, D, H) -> flatten -> (B, W, C*D*H)
        x_sagittal = features.permute(0, 4, 1, 2, 3).flatten(2)
        x_sagittal = self.proj_sagittal(x_sagittal) # (B, W', embed_dim)
        x_sagittal = x_sagittal + self.plane_embed_sagittal # Add Plane Embedding

        # 3. Concatenate all slices + CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x_seq = torch.cat((cls_tokens, x_axial, x_coronal, x_sagittal), dim=1) # (B, Total_Seq, embed_dim)

        # 4. Add Positional Embeddings
        # 注意：如果输入尺寸变化，这里需要插值调整 pos_embed 的大小，这里简化为固定尺寸匹配
        if x_seq.shape[1] == self.pos_embed.shape[1]:
            x_seq = x_seq + self.pos_embed
        else:
            # 简单的尺寸容错处理 (在实际应用中通常使用插值)
            x_seq = x_seq + self.pos_embed[:, :x_seq.shape[1], :]
            
        x_seq = self.pos_drop(x_seq)

        # 5. Transformer Encoder
        x_seq = self.transformer_encoder(x_seq)

        # 6. Classification Head (使用 CLS token 的输出)
        cls_out = x_seq[:, 0]
        cls_out = self.norm(cls_out)
        out = self.head(cls_out) # (B, 1)
        
        # 7. 最终激活 (Sigmoid 用于输出置信度)
        # out = self.sigmoid(out)
        
        return out

if __name__ == "__main__":
    # 模拟您的输入要求
    # 尺寸: (Batch Size, 3(模态数), W, H, Z)
    batch_size = 2
    in_channels = 3
    W, H, Z = 64, 64, 64
    
    input_tensor = torch.randn(batch_size, in_channels, W, H, Z)
    
    # 实例化模型
    # 注意：img_size 必须与输入数据的 (W, H, Z) 一致，以便计算 Linear 层的输入维度
    model = M3T(in_channels=in_channels, img_size=(W, H, Z), num_classes=1)
    
    # 打印参数量 (可选)
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # 前向传播
    output = model(input_tensor)
    
    print("-" * 30)
    print("M3T Model Test Summary")
    print("-" * 30)
    print(f"Input Shape:  {input_tensor.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Output Values (First Batch):\n {output[0].detach().numpy()}")
    print("-" * 30)
    
    # 验证输出尺寸
    assert output.shape == (batch_size, 1), f"Expected output shape {(batch_size, 1)}, but got {output.shape}"
    print("Test Passed: Output shape is correct.")