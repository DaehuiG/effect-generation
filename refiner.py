# Gray/alpha channel image refiner
import torch
import torch.nn as nn
import torchvision.transforms as T

# ----------------------------
# Text Encoder (간단한 임베딩)
# ----------------------------
class SimpleTextEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(1000, embed_dim)  # 가변 단어 사전 크기
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, tokens):
        x = self.embedding(tokens)  # [B, T, D]
        x = x.permute(0, 2, 1)      # [B, D, T]
        x = self.pool(x)            # [B, D, 1]
        return x.squeeze(-1)        # [B, D]

# ----------------------------
# U-Net 기반 Generator
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNetRefiner(nn.Module):
    def __init__(self, in_ch=2, text_dim=128):
        super().__init__()
        self.inc = DoubleConv(in_ch + 1, 64)  # +1 for text embedding as channel
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2), DoubleConv(256, 128))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2), DoubleConv(128, 64))
        self.outc = nn.Conv2d(64, 2, 1)  # Output: [gray, alpha]

    def forward(self, x, text_embed):
        B, _, H, W = x.size()
        text_map = text_embed.view(B, 1, 1, -1).repeat(1, 1, H, W)
        x = torch.cat([x, text_map], dim=1)  # [B, 2+1, H, W]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(torch.cat([x2, self.up1[0](x3)], dim=1))
        x = self.up2(torch.cat([x1, self.up2[0](x)], dim=1))
        out = self.outc(x)
        return out  # [B, 2, H, W]

# ----------------------------
# 예제 입력 및 추론
# ----------------------------
if __name__ == "__main__":
    B, H, W = 1, 128, 128
    gray_alpha = torch.randn(B, 2, H, W)  # Gray + Alpha
    tokens = torch.randint(0, 999, (B, 10))

    text_encoder = SimpleTextEncoder()
    generator = UNetRefiner()

    text_embed = text_encoder(tokens)  # [B, 128]
    refined = generator(gray_alpha, text_embed)  # [B, 2, H, W]

    print("Output shape:", refined.shape)  # [1, 2, 128, 128]
    # refined[0, 0] = gray, refined[0, 1] = alpha
