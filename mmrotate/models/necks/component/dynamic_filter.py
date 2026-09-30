import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

class DynamicFilter(BaseModule):
    def __init__(self,
                 input_dim=256,
                 filter_size=256,
                 num_filters=4,
                 init_cfg=None):
        super(DynamicFilter, self).__init__(init_cfg)

        self.num_filters = num_filters  # 滤波器的数量
        self.filter_size = filter_size  # 滤波器的尺寸

        # 定义复数权重参数
        self.complex_weights = nn.Parameter(
            torch.randn(filter_size, filter_size, num_filters, 2, dtype=torch.float32) * 0.02
        )
        
        # 定义 MLP，用于生成 routeing 权重
        hidden_dim = int(0.25 * input_dim)  # hidden layer 的大小，可以调整
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_filters),
        )

    def fourier_transform(self, x):
        """对输入特征图进行傅里叶变换"""
        fft_x = torch.fft.fft2(x, dim=(-2, -1))
        fft_x = torch.fft.fftshift(fft_x, dim=(-2, -1))
        return fft_x

    def frequency_processing(self, fft_x, routeing):
        """在频域中处理数据，应用动态滤波器"""
        B, C, H, W = fft_x.shape  # 获取傅里叶变换后的张量形状
        
        # 获取复数权重，将权重扩展到与 fft_x 的空间维度一致
        complex_weights = torch.view_as_complex(self.complex_weights)
        
        # 使用 routeing 权重生成滤波器
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        
        # 需要调整 weight 的尺寸以匹配 fft_x 的空间维度
        weight = weight.view(B, H, W, -1).permute(0, 3, 1, 2)  # 调整维度顺序为 (B, C, H, W)
        
        # 执行权重与频域特征的逐元素相乘
        filtered_fft_x = fft_x * weight
        
        return filtered_fft_x

    def inverse_fourier_transform(self, fft_x):
        """执行逆傅里叶变换，将数据转换回空间域"""
        fft_x = torch.fft.ifftshift(fft_x, dim=(-2, -1))
        ifft_x = torch.fft.ifft2(fft_x, dim=(-2, -1))
        return ifft_x.real

    def forward(self, x):
        # 1. 执行傅里叶变换
        fft_x = self.fourier_transform(x)
        
        # 2. 计算 routeing 权重
        B, C, H, W = x.shape
        # 对输入特征进行全局平均池化 (B, C, H, W) -> (B, C)
        pooled_x = x.mean(dim=(2, 3))  # 在空间维度上进行平均
        # 通过 MLP 生成 routeing 权重
        routeing = self.mlp(pooled_x)  # (B, num_filters)
        routeing = routeing.view(B, self.num_filters, -1).softmax(dim=1)  # 使用 softmax 保证权重总和为 1

        # 3. 在频域应用动态滤波器
        processed_fft_x = self.frequency_processing(fft_x, routeing)
        
        # 4. 执行逆傅里叶变换，将频域特征转换回空间域
        ifft_x = self.inverse_fourier_transform(processed_fft_x)
        
        # 5. 将逆变换后的特征图与原始特征图相加
        combined_out = x + ifft_x
        
        return combined_out
    
    def laplacian_transform(self, x):
        """应用拉普拉斯变换，使用卷积操作模拟拉普拉斯算子"""
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
        laplacian_kernel = laplacian_kernel.repeat(x.size(1), 1, 1, 1)  # 扩展到输入通道数
        return F.conv2d(x, laplacian_kernel, padding=1, groups=x.size(1))
    
    def fourier_transform_abs(self, x):
        """应用傅里叶变换，返回变换后的幅度谱"""
        B, C, H, W = x.shape
        # 进行2D傅里叶变换，只对空间维度（H, W）进行
        fft_x = torch.fft.fft2(x, dim=(-2, -1))  # 在H和W维度上执行傅里叶变换
        # 转换为频谱幅度
        magnitude_spectrum = torch.abs(torch.fft.fftshift(fft_x, dim=(-2, -1)))  # fftshift也只在H和W维度上进行
        return magnitude_spectrum