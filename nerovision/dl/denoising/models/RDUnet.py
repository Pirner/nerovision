import torch
from torch import nn


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        neural network block for downsampling within RDUnet
        :param in_channels: number of channels (or features) that go into the block
        :param out_channels: number of channels (or features) that go out of the block
        """
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        return self.actv(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            cat_channels: int,
            out_channels: int,
            kernel_size=3,
    ):
        """
        neural network block for upsampling within RDUnet
        :param in_channels: number of channels (or features) that go into the block
        :param cat_channels: how many features are being concatenated
        :param out_channels: number of features that go out
        :param kernel_size: the size of the kernel for the convolution in 2d
        """
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.actv = nn.PReLU(out_channels)
        self.actv_t = nn.PReLU(in_channels)

    def forward(self, x):
        upsample, concat = x
        upsample = self.actv_t(self.conv_t(upsample))
        return self.actv(self.conv(torch.cat([concat, upsample], 1)))


class InputBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
    ):
        """
        neural network block for the first input into the neural network RDUnet.
        the very first block, grayscale images, 1 channel in, rgb images 3 channels in
        :param in_channels: number of channels that go into the network
        :param out_channels: number of channels that go ot of the network
        :param kernel_size: the size of the kernel for the convolution
        """
        super(InputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')

        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class OutputBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
    ):
        """
        the output block of the RDUnet neural network
        :param in_channels: number of input features that are being forwarded
        :param out_channels: output channels (gray, rgb, 1, 3)
        :param kernel_size: the size of the kernel for the convolution
        """
        super(OutputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same')
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')

        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class DenoisingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            inner_channels: int,
            out_channels: int,
            kernel_size=3,
    ):
        """
        denoising block in the neural network
        :param in_channels: number of input features for the denoising block
        :param inner_channels: number of features that are there within the block
        :param out_channels: number of features that go out of the denoising block
        :param kernel_size: the size of the kernel for the convolution layers
        """
        super(DenoisingBlock, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, kernel_size, padding='same')
        self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, kernel_size, padding='same')
        self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, kernel_size, padding='same')
        self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, kernel_size, padding='same')

        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))

        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))

        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))

        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))

        return out_3 + x


class RDUNet(nn.Module):
    r"""
    Residual-Dense U-net for image denoising.
    """
    def __init__(self, **kwargs):
        super().__init__()

        channels = kwargs['channels']
        filters_0 = kwargs['base filters']
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0

        # Encoder:
        # Level 0:
        self.input_block = InputBlock(channels, filters_0)
        self.block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.down_0 = DownsampleBlock(filters_0, filters_1)

        # Level 1:
        self.block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.down_1 = DownsampleBlock(filters_1, filters_2)

        # Level 2:
        self.block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.down_2 = DownsampleBlock(filters_2, filters_3)

        # Level 3 (Bottleneck)
        self.block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)
        self.block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)

        # Decoder
        # Level 2:
        self.up_2 = UpsampleBlock(filters_3, filters_2, filters_2)
        self.block_2_2 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_3 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)

        # Level 1:
        self.up_1 = UpsampleBlock(filters_2, filters_1, filters_1)
        self.block_1_2 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_3 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)

        # Level 0:
        self.up_0 = UpsampleBlock(filters_1, filters_0, filters_0)
        self.block_0_2 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_3 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)

        self.output_block = OutputBlock(filters_0, channels)

    def forward(self, inputs):
        out_0 = self.input_block(inputs)    # Level 0
        out_0 = self.block_0_0(out_0)
        out_0 = self.block_0_1(out_0)

        out_1 = self.down_0(out_0)          # Level 1
        out_1 = self.block_1_0(out_1)
        out_1 = self.block_1_1(out_1)

        out_2 = self.down_1(out_1)          # Level 2
        out_2 = self.block_2_0(out_2)
        out_2 = self.block_2_1(out_2)

        out_3 = self.down_2(out_2)          # Level 3 (Bottleneck)
        out_3 = self.block_3_0(out_3)
        out_3 = self.block_3_1(out_3)

        out_4 = self.up_2([out_3, out_2])   # Level 2
        out_4 = self.block_2_2(out_4)
        out_4 = self.block_2_3(out_4)

        out_5 = self.up_1([out_4, out_1])   # Level 1
        out_5 = self.block_1_2(out_5)
        out_5 = self.block_1_3(out_5)

        out_6 = self.up_0([out_5, out_0])   # Level 0
        out_6 = self.block_0_2(out_6)
        out_6 = self.block_0_3(out_6)

        return self.output_block(out_6) + inputs
