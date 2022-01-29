"""
Discriminator and Generator model from DCGAN paper
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_size, n_features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input is N x n_channel x 64 x 64 
            nn.Conv2d(img_size, n_features, kernel_size=4, stride=2, padding=1),    #32x32
            nn.LeakyReLU(0.2),
            self._block(n_features, n_features * 2, kernel_size=4, stride=2, padding=1),    #16x16
            self._block(n_features * 2, n_features * 4, kernel_size=4, stride=2, padding=1),    #8x8
            self._block(n_features * 4, n_features * 8, kernel_size=4, stride=2, padding=1),    #4x4
            nn.Conv2d(n_features * 8, 1, kernel_size=4, stride=2, padding=0),   #1x1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, n_channels, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0),    #4x4
            self._block(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1),    #8x8
            self._block(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1),    #16x16
            self._block(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1),    #32x32
            nn.ConvTranspose2d(features_g * 2, n_channels, kernel_size=4, stride=2, padding=1),    #64x64
            nn.Tanh(),
        )



    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


