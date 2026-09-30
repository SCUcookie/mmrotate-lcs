import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule


class IntensiveCompute(BaseModule):
    def __init__(self, out_channels):
        super(IntensiveCompute, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool_mlp = nn.Linear(out_channels, out_channels)
        self.avg_pool_mlp = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        max_pool_x = self.max_pool(x)
        avg_pool_x = self.avg_pool(x)
        max_pool_c = torch.mean(max_pool_x, dim=(2, 3), keepdim=False)
        max_pool_c = self.max_pool_mlp(max_pool_c).sigmoid().unsqueeze(-1).unsqueeze(-1)
        avg_pool_c = torch.mean(avg_pool_x, dim=(2, 3), keepdim=False)
        avg_pool_c = self.avg_pool_mlp(avg_pool_c).sigmoid().unsqueeze(-1).unsqueeze(-1)
        x = max_pool_c * max_pool_x + avg_pool_c * avg_pool_x
        return x

class ChannelAttention(BaseModule):
    def __init__(self, in_planes, ratio = 4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # nn.init.xavier_uniform_(self.conv1.weight)
        # nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(BaseModule):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)
    

class SelectiveFeatureFusion(BaseModule):
    ''' Selective Feature Fusion(SFF) '''
    def __init__(self, out_channels, scale_factor):
        super(SelectiveFeatureFusion, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3,
                                             stride=2, padding=1, output_padding=1)
        if scale_factor == 1:
            self.upsample = nn.Identity()
        else:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                                        align_corners=True)
        
        self.channel_attention = ChannelAttention(out_channels)
    
    def forward(self, f_high, f_low):
        # Transposed convolution for f_high
        f_att = self.trans_conv(f_high)
        
        # Bilinear interpolation for f_low
        f_att = self.upsample(f_att)
        
        # Channel attention fusion
        fusion_feature = f_low * self.channel_attention(f_att)
        
        # Feature add
        f_out = fusion_feature + f_att
        return f_out

class FeatureSelectionModule(BaseModule):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()

        self.ca = ChannelAttention(in_chan)
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1)

    def forward(self, x):
        atten = self.ca(x)
        atten_x = x * atten
        x = x + atten_x
        return self.conv(x)
    
class FeatureInteractionModule(BaseModule):
    def __init__(self, chan):
        super(FeatureInteractionModule, self).__init__()

        self.sa = SpatialAttention(kernel_size=7)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        # self.conv = nn.Conv2d(chan, chan, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x_low, x_high):
        x_low = self.downsample(x_low)
        x = x_high + x_low
        atten = self.sa(x_high)
        x = x * atten + x
        return x
    
    # def __init__(self, in_channels):
    #     super(FeatureInteractionModule, self).__init__()
    #     self.spatial_interaction = SpatialInteraction(in_channels)
    #     self.channel_interaction = ChannelInteraction(in_channels)
    #     self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    # def forward(self, input, index):
    #     short_cut = input[index]
    #     x = input[index]
    #     pre_shape = x.shape
    #     for f, idx in enumerate(input):
    #         if idx != index:
    #            f = nn.functional.interpolate(f, size=pre_shape[2:], mode='bilinear')
    #         x = x + f
    #     spatial_output = self.spatial_interaction(x)
    #     channel_output = self.channel_interaction(x)
    #     interaction = spatial_output + channel_output + short_cut
    #     output = self.conv1x1(interaction)
    #     return output
    
class ChannelInteraction(BaseModule):
    def __init__(self, in_channels):
        super(ChannelInteraction, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_fc1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 使用1x1卷积来合并全局信息
        self.fc2 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        avg_pool = self.shared_fc1(self.global_avg_pool(x)).relu()
        max_pool = self.shared_fc1(self.global_max_pool(x)).relu()
        cat_features = torch.cat([avg_pool, max_pool], dim=1)
        atten = self.fc2(cat_features).sigmoid()
        x = x * atten + x
        return x
    
class SpatialInteraction(BaseModule):
    def __init__(self, in_channels):
        super(SpatialInteraction, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dilated_conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels)

    def forward(self, x):
        local_interaction = self.conv3x3(x)
        non_local_interaction = self.dilated_conv3x3(x)
        output = torch.relu(local_interaction + non_local_interaction)
        return output