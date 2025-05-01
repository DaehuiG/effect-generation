
import torch
import torch.nn as nn

# Residual Block used in the Generator
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# GESGAN Generator: Encoder-ResBlock-Decoder
class GESGANGenerator(nn.Module):
    def __init__(self, style_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 7, 1, 3),  # 3 for image + 1 for style channel
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.resblocks = nn.Sequential(*[ResidualBlock(256) for _ in range(5)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )
        self.style_dim = style_dim

    def forward(self, x, c):
        B, _, H, W = x.size()
        style_map = c.view(B, self.style_dim, 1, 1)
        style_map = style_map.repeat(1, 1, H, W)
        x = torch.cat([x, style_map[:, :1]], dim=1)
        return self.decoder(self.resblocks(self.encoder(x)))

# GESGAN Discriminator with auxiliary classifier
class GESGANDiscriminator(nn.Module):
    def __init__(self, num_styles=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.adv_head = nn.Conv2d(512, 1, 4)  # real/fake
        self.cls_head = nn.Conv2d(512, num_styles, 4)  # style classification

    def forward(self, x):
        feat = self.features(x)
        adv = self.adv_head(feat).view(x.size(0), -1)
        cls = self.cls_head(feat).view(x.size(0), -1)
        return adv, cls
