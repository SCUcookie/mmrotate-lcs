import unittest
import torch
from mmdet.models.necks import FPN  # 从mmdet中导入FPN模块
from thop import profile

class TestFPN(unittest.TestCase):

    def setUp(self):
        """在每个测试用例之前初始化FPN模型"""
        # 设置输入通道数和输出通道数
        self.in_channels = [64, 128, 320, 512]
        self.out_channels = 256
        self.num_outs = 5  # 输出特征图数量

        # 初始化 FPN 模型
        self.fpn = FPN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_outs=self.num_outs
        )

    def test_fpn_output_shape(self):
        """测试FPN的输出特征图形状是否正确"""
        img_size = 256  # 输入图像大小 (256x256)

        # 构造输入特征图，每个输入特征图的空间尺寸依次减半
        inputs = [
            torch.rand(1, c, img_size // (2 ** i), img_size // (2 ** i)).cuda()
            for i, c in enumerate(self.in_channels)
        ]
        
        # 将 FPN 模型移动到 GPU
        self.fpn.cuda()

        # 执行前向传播
        outputs = self.fpn(inputs)


        # 打印输出形状
        print("\033[91mOutputs:\033[0m")
        for i, output in enumerate(outputs):
            print(f"\033[91mOutput {i + 1} shape: {output.shape}\033[0m")

        # 计算参数数量和FLOPs
        flops, params = profile(self.fpn, inputs=(inputs,))
        print(f"\033[91mTotal GFLOPs: {flops / 1e9}, Parameters: {params / 1e6}M\033[0m")

        # 打印详细的参数信息
        print("\nDetailed parameter information:")
        for name, param in self.fpn.named_parameters():
            print(f"{name}: {param.numel()} parameters")


        # 检查输出特征图数量是否正确
        self.assertEqual(len(outputs), self.num_outs)

        # 检查每个输出特征图的形状是否正确
        for i in range(self.num_outs):
            out_shape = outputs[i].shape
            expected_shape = (1, self.out_channels, img_size // (2 ** i), img_size // (2 ** i))
            self.assertEqual(out_shape, expected_shape, f"输出 {i} 的形状不匹配: {out_shape} != {expected_shape}")

if __name__ == "__main__":
    unittest.main()
