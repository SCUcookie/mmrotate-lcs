import torch
import unittest
from mmrotate.registry import MODELS
from torch.nn.modules.batchnorm import _BatchNorm
from thop import profile

# 假设 MFPN 模型已经注册到 MODELS 中
class TestMFPN(unittest.TestCase):

    def setUp(self):
        # 设置输入通道数和输出通道数
        self.in_channels = [64, 128, 320, 512]
        self.out_channels = 256
        self.num_outs = 5
        self.img_size = 256  # 假设特征图大小为 256x256

        # 初始化 MFPN 模型
        self.mfpn = MODELS.build(dict(
            type='MFPN',
            img_size=self.img_size*4,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_outs=self.num_outs
        ))

    def test_mfpn_output_shape(self):
        """测试 MFPN 的输出特征图形状是否正确"""
        # 构造输入特征图
        inputs = [
            torch.rand(1, c, self.img_size // (2 ** i), self.img_size // (2 ** i)).cuda()
            for i, c in enumerate(self.in_channels)
        ]
        
        # 将模型移动到 GPU
        self.mfpn.cuda()
        
        # 执行前向传播
        outputs = self.mfpn(inputs)

        # 打印输出形状
        print("\033[91mOutputs:\033[0m")
        for i, output in enumerate(outputs):
            print(f"\033[91mOutput {i + 1} shape: {output.shape}\033[0m")

        # 计算参数数量和FLOPs
        flops, params = profile(self.mfpn, inputs=(inputs,))
        print(f"\033[91mTotal GFLOPs: {flops / 1e9}, Parameters: {params / 1e6}M\033[0m")
        print("\033[92mFor FPN in 1024x1024\nTotal GFLOPs: 53.418655744, Parameters: 2.623488M\033[0m")

        # # 打印详细的参数信息
        # print("\nDetailed parameter information:")
        # for name, param in self.mfpn.named_parameters():
        #     print(f"{name}: {param.numel()} parameters")

        # 检查输出特征图数量是否正确
        self.assertEqual(len(outputs), self.num_outs)

        # 检查每个输出的形状是否正确
        for i in range(self.num_outs):
            out_shape = outputs[i].shape
            expected_shape = (1, self.out_channels, self.img_size // (2 ** (i)), self.img_size // (2 ** (i)))
            self.assertEqual(out_shape, expected_shape, f"输出 {i} 的形状不匹配: {out_shape} != {expected_shape}")

if __name__ == "__main__":
    unittest.main()
