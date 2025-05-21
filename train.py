
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from model import GESGANGenerator, GESGANDiscriminator

# Config
IMG_SIZE = 128
NUM_STYLES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
LR = 2e-4
LAMBDA_L1 = 100

# Dummy dataset loader (replace with actual DataLoader)
def dummy_loader(batch_size=4):
    for _ in range(10):  # 10 batches
        x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)  # structure image
        y = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)  # target effect
        c = torch.randint(0, NUM_STYLES, (batch_size,))
        c_onehot = torch.nn.functional.one_hot(c, NUM_STYLES).float()
        yield x.to(DEVICE), y.to(DEVICE), c_onehot.to(DEVICE)

# Losses
bce = nn.BCEWithLogitsLoss()
ce = nn.CrossEntropyLoss()
l1 = nn.L1Loss()

# Models
G = GESGANGenerator(style_dim=NUM_STYLES).to(DEVICE)
D = GESGANDiscriminator(num_styles=NUM_STYLES).to(DEVICE)

# Optimizers
g_optim = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
d_optim = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# Training loop
for epoch in range(EPOCHS):
    for i, (x, y, c) in enumerate(dummy_loader()):
        B = x.size(0)

        # === Train Discriminator ===
        y_fake = G(x, c).detach()
        d_real, cls_real = D(y)
        d_fake, _ = D(y_fake)

        real_label = torch.ones_like(d_real).to(DEVICE)
        fake_label = torch.zeros_like(d_fake).to(DEVICE)

        d_adv_loss = bce(d_real, real_label) + bce(d_fake, fake_label)
        d_cls_loss = ce(cls_real, c.argmax(dim=1))
        d_loss = d_adv_loss + d_cls_loss

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        # === Train Generator ===
        y_fake = G(x, c)
        d_fake, cls_fake = D(y_fake)

        g_adv_loss = bce(d_fake, real_label)
        g_cls_loss = ce(cls_fake, c.argmax(dim=1))
        g_l1_loss = l1(y_fake, y) * LAMBDA_L1
        g_loss = g_adv_loss + g_cls_loss + g_l1_loss

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        if i % 5 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    # Save generated image sample
    with torch.no_grad():
        sample = G(x[:4], c[:4])
        save_image(sample * 0.5 + 0.5, f"result/sample_epoch_{epoch}.png")
