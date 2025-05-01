# model.py -------------------------------------------------------------
import torch
import torch.nn as nn

# ── Residual block ──────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch)
        )
    def forward(self, x): return x + self.block(x)

# ── Generator (Encoder-ResBlocks-Decoder) ────────────────────────────
class GESGANGenerator(nn.Module):
    def __init__(self, style_dim: int):
        super().__init__()
        in_ch = 4 + style_dim
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256), nn.ReLU(True)
        )
        self.res = nn.Sequential(*[ResBlock(256) for _ in range(3)])
        self.dec = nn.Sequential( 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256,128,3,1,1,bias=False), 
            nn.InstanceNorm2d(128), nn.ReLU(True), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(128,64,3,1,1,bias=False), 
            nn.InstanceNorm2d(64), nn.ReLU(True), 
            nn.Conv2d(64,4,7,1,3), nn.Tanh()
        )
        self.style_dim = style_dim
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
            nn.init.normal_(m.weight,0,0.02)

    def forward(self, x, c):
        B,_,H,W = x.shape
        style = c.view(B,self.style_dim,1,1).expand(-1,-1,H,W)
        h = torch.cat([x, style], 1)
        return self.dec(self.res(self.enc(h)))

# ── Discriminator (PatchGAN + aux-classifier) ────────────────────────
class GESGANDiscriminator(nn.Module):
    def __init__(self, num_styles: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),  nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.InstanceNorm2d(128),      nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.InstanceNorm2d(256),      nn.LeakyReLU(0.2,True),
            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.InstanceNorm2d(512),      nn.LeakyReLU(0.2,True)
        )
        self.adv = nn.Conv2d(512,1,4)           # real/fake
        self.cls = nn.Conv2d(512,num_styles,4)  # style class
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            nn.init.normal_(m.weight,0,0.02)

    def forward(self, x):
        feat = self.features(x)
        return self.adv(feat).view(x.size(0),-1), self.cls(feat).view(x.size(0),-1)
