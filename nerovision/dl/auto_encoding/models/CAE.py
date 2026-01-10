from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class Encoder(nn.Module):
    """
    convolutional encoder for an auto encoder
    """
    def __init__(
            self,
            features: List[int],
            latent_dim: int,
            strided_h: int,
            strided_w: int,
            kernel_size=3,
            in_channels=3
    ) -> None:
        """
        constructor for Encoder
        :param features: features that go in
        :param latent_dim: compressed latent dim representation
        :param strided_h: height of the last compressed representation
        :param strided_w: width of the last compressed representation
        :param kernel_size: size of the kernel for the convolutions
        :param in_channels: how many channels for the input
        """
        super(Encoder, self).__init__()

        self.features = features
        self.latent_dim = latent_dim
        self.strided_h = strided_h
        self.strided_w = strided_w
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        conv_layers = []

        for i, feat in enumerate(features):
            if i == 0:
                conv_layers.append(ResidualBlock(self.in_channels, feat, stride=2, kernel_size=kernel_size))
            else:
                conv_layers.append(ResidualBlock(features[i - 1], feat, stride=2, kernel_size=kernel_size))

        self.flatten = nn.Flatten()
        self.latent_linear = nn.Linear(features[-1] * strided_h * strided_w, latent_dim)

        conv_model = nn.Sequential(*conv_layers)
        self.conv_model = conv_model

    def forward(self, x: Tensor) -> Tensor:
        conv_out = self.conv_model(x)
        flattened = self.flatten(conv_out)
        latent_vector = self.latent_linear(flattened)
        return latent_vector

class Decoder(nn.Module):
    def __init__(
            self,
            features: List[int],
            strided_h: int,
            strided_w: int,
            latent_dim: int,
            kernel_size: int = 3,
            in_channels=3
    ):
        """
        constructor for Decoder
        :param features: features that go in
        :param latent_dim: compressed latent dim representation
        :param strided_h: height of the last compressed representation
        :param strided_w: width of the last compressed representation
        :param kernel_size: size of the kernel for the convolutions
        :param in_channels: how many channels for the input
        """
        super(Decoder, self).__init__()
        self.features = features
        self.latent_dim = latent_dim
        self.strided_h = strided_h
        self.strided_w = strided_w
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        self.reversed_features = features[::-1]

        self.latent_unfold = nn.Linear(self.latent_dim, self.reversed_features[0] * self.strided_h * self.strided_w)
        self.reshape = Reshape(self.reversed_features[0], self.strided_h, self.strided_w)

        # build convolutional decoder
        conv_layers = []

        for i, feat in enumerate(self.reversed_features):
            if i == len(self.reversed_features) - 1:
                conv_layers.append(nn.ConvTranspose2d(
                    self.reversed_features[i],
                    self.in_channels,
                    kernel_size=2,
                    stride=2)
                )
            else:
                conv_layers.append(nn.ConvTranspose2d(feat, self.reversed_features[i + 1], kernel_size=2, stride=2))
                conv_layers.append(ResidualBlock(
                    self.reversed_features[i + 1],
                    self.reversed_features[i + 1],
                    kernel_size=kernel_size,
                    stride=1
                ))
        conv_model = nn.Sequential(*conv_layers)
        self.conv_model = conv_model

    def forward(self, x):
        latent_unfold = self.latent_unfold(x)
        reshaped = self.reshape(latent_unfold)
        reconstruction = self.conv_model(reshaped)
        return reconstruction



class CAE(nn.Module):
    """
    convolutional autoencoder, central class for anomaly detection via reconstruction with auto encoders.
    """
    kernel_size: int
    n_layers: int


    def __init__(
            self,
            im_h: int,
            im_w: int,
            features: List[int],
            latent_dim: int,
            n_channels=3,
            kernel_size=3,
            return_latent=True,
    ) -> None:
        """
        constructor for CNNAutoencoder
        :param im_h: image height
        :param im_w: image width
        :param kernel_size: size of the kernel for the convolution operations
        :param features: features for the encoder
        :param n_channels: number of input channels
        :param latent_dim: size of the latent dimensional vector for the compression space
        """
        super(CAE, self).__init__()

        self.in_channels = n_channels
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.im_w = im_w
        self.im_h = im_h
        self.features = features
        self.return_latent = return_latent

        if im_h % 32 != 0:
            raise ValueError('Image Height should be a multiple of 32')
        if im_w % 32 != 0:
            raise ValueError('Image Width should be a multiple of 32')
        self.strided_w = self.im_w // pow(2, len(self.features))
        self.strided_h = self.im_h // pow(2, len(self.features))

        # build the encoder
        encoder = Encoder(features, latent_dim, self.strided_h, self.strided_w, self.kernel_size, self.in_channels)
        self._encoder = encoder
        # build the decoder
        decoder = Decoder(
            features,
            strided_h=self.strided_h,
            strided_w=self.strided_w,
            latent_dim=self.latent_dim,
            kernel_size=self.kernel_size,
            in_channels=self.in_channels
        )
        self._decoder = decoder

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self._encoder(x)  # latent dimensional vector representation
        decoded = self._decoder(encoded)
        if self.return_latent:
            return decoded, encoded
        else:
            return decoded
