import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossRegionGroupAttentionModule(nn.Module):
    def __init__(self, in_channels, N, height=7, width=7, G1=4, G2=2, num_heads=4):
        super(CrossRegionGroupAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.N = N
        self.height = height
        self.width = width
        self.G1 = G1
        self.G2 = G2
        
        # Self-Attention 模块，用于计算每组之间的交互
        self.attn1 = nn.MultiheadAttention(embed_dim=in_channels * height * width, num_heads=num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=in_channels * height * width, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x shape: (N, C, H, W)
        N, C, H, W = x.shape

        # Step 1: 展平空间维度，得到 (N, C*H*W)
        x_flat = x.flatten(2).transpose(1, 2)  # (N, H*W, C)

        # Step 2: 分组后，进行第一个阶段的注意力计算
        # 将输入分成 G1 组，每组大小为 N // G1
        x_grouped1 = x_flat.reshape(self.G1, N // self.G1, C * H * W)  # (G1, N//G1, C*H*W)
        
        # 使用自注意力计算每个组内的交互
        attn_output1, _ = self.attn1(x_grouped1, x_grouped1, x_grouped1)  # (G1, N//G1, C*H*W)

        # Step 3: 将第一个阶段的输出恢复形状
        attn_output1 = attn_output1.reshape(N, C * H * W)  # 恢复为 (N, C*H*W)

        attn_output_grouped2 = attn_output1.reshape(self.G2, N // self.G2, C * H * W)
        
        # 使用自注意力计算第二阶段的交互
        attn_output2, _ = self.attn2(attn_output_grouped2, attn_output_grouped2, attn_output_grouped2)

        # Step 5: 恢复最终输出
        attn_output2 = attn_output2.reshape(N, H*W, C)  # 恢复为 (N, H*W, C)
        attn_output2 = attn_output2.transpose(1, 2).reshape(N, C, H, W)  # 恢复为 (N, C, H, W)

        return attn_output2

if __name__ == '__main__':
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    # 示例输入：假设 N=512, C=256, H=7, W=7, G1=4, G2=2
    x = torch.randn(512, 256, 7, 7)  # 输入的特征图

    model = CrossRegionGroupAttentionModule(in_channels=256, N=512, height=7, width=7, G1=256, G2=256)
    output = model(x)

    print(output.shape)  # 应该输出 (512, 256, 7, 7)

    # summary(model, (256, 7, 7))

    # 使用 FlopCountAnalysis 来计算 FLOPs
    flops = FlopCountAnalysis(model, x)

    # 获取 FLOPs 总数
    total_flops = flops.total()

    # 将 FLOPs 转换为 GFLOPs（除以 10^9）
    gflops = total_flops / 1e9

    print(f'Total GFLOPs: {gflops:.4f}')