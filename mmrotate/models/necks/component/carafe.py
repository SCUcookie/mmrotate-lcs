import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

class CARAFE(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, up_kernel_size=3, scale_factor=2):
        super(CARAFE, self).__init__()
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size

        # Encoder to generate reassembly kernel weights
        self.encoder = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)  # 1x1 Conv to reduce channels
        self.conv_w = nn.Conv2d(in_channels // 4, up_kernel_size**2 * scale_factor**2, kernel_size=1)  # generate weights

        # Unfold layer used to get local patches
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=1, padding=kernel_size // 2)

        # Upsample to achieve final resolution
        self.upsample = nn.PixelShuffle(scale_factor)

        # Pointwise convolution for dimension matching (optional)
        self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        N, C, H, W = x.shape

        # Step 1: Generate kernel weights for each pixel
        w = self.encoder(x)  # Shape: [N, C//4, H, W]
        w = self.conv_w(w)  # Shape: [N, up_kernel_size**2 * scale_factor**2, H, W]
        # Reshape weight tensor to match patches for reassembly
        w = w.view(N, self.up_kernel_size ** 2, self.scale_factor ** 2, H, W)  # Shape: [N, K*K, scale_factor*scale_factor, H, W]

        # Step 2: Extract patches from input feature maps
        x_unfolded = self.unfold(x)  # Shape: [N, C*K*K, H*W]
        x_unfolded = x_unfolded.view(N, C, self.kernel_size ** 2, H, W)  # Reshape: [N, C, K*K, H, W]

        # Step 3: Re-assemble features using learned weights
        # Correct broadcasting operation to match shapes
        out = torch.einsum('nckhw,nkfhw->ncfhw', x_unfolded, w)  # Apply kernel weights to each patch

        # Step 4: Upsample using PixelShuffle-like operation
        out = out.view(N, C * self.scale_factor**2, H, W)  # Reshape to [N, C*scale_factor*scale_factor, H, W]
        out = self.upsample(out)  # Upsample: [N, C, H*scale_factor, W*scale_factor]

        # Match the output channels if necessary
        if self.match_channels:
            out = self.match_channels(out)

        return out

# Testing CARAFE
if __name__ == "__main__":
    carafe_layer = CARAFE(in_channels=64, out_channels=64, kernel_size=3, up_kernel_size=3, scale_factor=2)
    
    # Example input tensors for different scales
    input_tensors = [
        torch.randn(2, 64, 256, 256),  # Scale 1
        torch.randn(2, 64, 128, 128),  # Scale 2
        torch.randn(2, 64, 64, 64),    # Scale 3
        torch.randn(2, 64, 32, 32)     # Scale 4
    ]

    # Calculate the number of parameters
    total_params = sum(p.numel() for p in carafe_layer.parameters())
    print(f"Total number of parameters: {total_params}")

    # Calculate the computational cost (FLOPs) for each input tensor
    from thop import profile
    for i, input_tensor in enumerate(input_tensors, 1):
        flops, params = profile(carafe_layer, inputs=(input_tensor,))
        print(f"Scale {i} - GFLOPs: {flops / 1e9}, Parameters: {params}")

        output_tensor = carafe_layer(input_tensor)
        print(f"\033[91mScale {i} - Output shape: {output_tensor.shape}\033[0m")  # Expected shape: (2, 64, H*scale_factor, W*scale_factor)
