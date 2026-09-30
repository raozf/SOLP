import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
class FourierFeatureEncoding(nn.Module):
    """
    将低维坐标映射到高维傅里叶特征，解决 MLP 的 Spectral Bias 问题。
    参考: "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    """
    def __init__(self, input_dim=3, mapping_size=64, scale=10):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        # 随机高斯矩阵 B，不可学习，作为 Buffer 存储
        self.register_buffer('B', torch.randn((input_dim, mapping_size)) * scale)

    def forward(self, x):
        # x: [..., input_dim]
        # projection: [..., mapping_size]
        x_proj = (2. * np.pi * x) @ self.B
        # 输出 sin 和 cos，维度翻倍: [..., 2 * mapping_size]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class LatentFFTFilter(nn.Module):
    def __init__(self, in_channels=4, hidden=128):
        """
        in_channels: Latent 的通道数 (例如 VAE 通常是 4)
        hidden: 网络隐藏层通道数
        """
        super().__init__()
        # ===== 1️⃣ 坐标编码 =====
        # 将 (t, h, w) 3维坐标映射到 128 维 (64*2)
        self.pos_enc = FourierFeatureEncoding(input_dim=3, mapping_size=64, scale=10)
        pos_dim = 64 * 2

        # ===== 2️⃣ 频域卷积网络 (替代原先的 MLP) =====
        # 输入通道 = Latent通道数(幅度值) + 位置编码通道数
        input_dim = in_channels + pos_dim

        self.freq_conv = nn.Sequential(
            # 第一层：融合 幅度特征 和 位置特征
            # 使用 1x1 卷积先降维/融合，或者直接 3x3 开始
            nn.Conv3d(input_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden), # GroupNorm 对 BatchSize 不敏感，适合视频任务
            nn.GELU(),

            # 中间层：增加深度，扩大感受野，建立频率间的交互
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            
            # 再次交互
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),

            # 输出层：输出对 Log-Amplitude 的残差调整量
            # 输出通道数必须等于 Latent 通道数
            nn.Conv3d(hidden, in_channels, kernel_size=1)
        )
        # ===== 3️⃣ 可学习残差强度 =====
        # 初始化为 0，确保网络从 Identity (不做任何修改) 开始学习
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
        # 这是一个额外的技巧：将最后一层卷积的权重初始化得非常小
        # 进一步保证初始输出接近 0
        nn.init.uniform_(self.freq_conv[-1].weight, -1e-5, 1e-5)
        nn.init.zeros_(self.freq_conv[-1].bias)

    def forward(self, z, coord=None):
        """
        z: [B, C, T, H, W] (Latent Code)
        coord: [T, H, W, 3] (归一化坐标, 范围通常是 0-1 或 -1-1)
        """
        if coord is None:
            raise ValueError("LatentFFTFilter requires coord for freq conditioning")

        orig_dtype = z.dtype
        B, C, T, H, W = z.shape

        z_fp32 = z.float()
        Z = torch.fft.fftn(z_fp32, dim=(2, 3, 4))
        # 分离 幅度(Amplitude) 和 相位(Phase)
        A = torch.abs(Z)
        phase = Z / (A + 1e-6)
        # 取对数幅度，数值分布更适合神经网络处理
        A_log = torch.log(A + 1e-6) # [B, C, T, H, W]

        # ------------------------------------------------------------------
        # 2. 准备网络输入
        # ------------------------------------------------------------------
        # 处理坐标编码
        # coord: [T, H, W, 3] -> pos_feat: [T, H, W, 128]
        pos_feat = self.pos_enc(coord.to(z.device).float())
        
        # 调整维度以匹配 Batch: [1, 128, T, H, W] -> [B, 128, T, H, W]
        pos_feat = pos_feat.permute(3, 0, 1, 2).unsqueeze(0) # [1, 128, T, H, W]
        pos_feat = pos_feat.expand(B, -1, -1, -1, -1)

        # 拼接 幅度特征 和 位置特征
        # x: [B, C+128, T, H, W]
        x = torch.cat([A_log, pos_feat], dim=1)
        # ------------------------------------------------------------------
        # 3. 频域卷积处理 (学习残差 dA)
        # ------------------------------------------------------------------
        dA = self.freq_conv(x) # [B, C, T, H, W]
        # ------------------------------------------------------------------
        # 4. 逆变换与残差连接
        # ------------------------------------------------------------------
        # 应用调整：A_new = exp(log(A) + dA)
        A_hat = torch.exp(A_log + dA)
        # 重组复数形式 (保持原相位)
        Z_hat = A_hat * phase
        # iFFT
        z_hat = torch.fft.ifftn(Z_hat, dim=(2, 3, 4)).real
        # 转换回原始精度
        z_hat = z_hat.to(orig_dtype)
        # 最终输出：原图 + alpha * (去噪图 - 原图)
        # 随着训练进行，alpha 会自动调整，网络会学会只剔除噪声频率
        return z + self.alpha * (z_hat - z)