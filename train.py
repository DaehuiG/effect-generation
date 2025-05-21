# train.py ------------------------------------------------------------
import os, torch, torchvision
from torch import nn
from torch.optim import Adam
from dataset import build_loader
from model import GESGANGenerator, GESGANDiscriminator

# ── 하이퍼파라미터 ───────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 50
BATCH       = 8
LR_G, LR_D  = 3e-4, 1e-4
LAMBDA_L1   = 100

# ── Data & Model 준비 ───────────────────────────────────────────────
loader, NUM_STYLES = build_loader(batch=BATCH, root="dataset")
G = GESGANGenerator(style_dim=NUM_STYLES).to(DEVICE)
D = GESGANDiscriminator(num_styles=NUM_STYLES).to(DEVICE)

bce, ce, l1 = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.L1Loss()
g_opt = Adam(G.parameters(), lr=LR_G, betas=(0.5,0.999))
d_opt = Adam(D.parameters(), lr=LR_D, betas=(0.5,0.999))

# ── 결과 폴더 넘버링 ─────────────────────────────────────────────────
os.makedirs("result", exist_ok=True)
existing = sorted([int(d) for d in os.listdir("result") if d.isdigit()])
run_id   = existing[-1]+1 if existing else 0
save_dir = os.path.join("result", f"{run_id:02d}")
os.makedirs(save_dir, exist_ok=True)

# ── 학습 Loop ────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    for i,(x,y,c) in enumerate(loader):
        x,y,c = x.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
        real,fake = torch.ones,(torch.zeros)
        # ── Discriminator ──
        with torch.no_grad():
            y_fake = G(x,c)
        d_real, cls_real = D(y)
        d_fake, _        = D(y_fake.detach())
        real_lbl = real_like = torch.ones_like(d_real)
        fake_lbl = torch.zeros_like(d_fake)

        d_adv = bce(d_real, real_lbl) + bce(d_fake, fake_lbl)
        d_cls = ce(cls_real, c.argmax(1))
        d_loss = d_adv + d_cls
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()

        # ── Generator ──
        y_fake = G(x,c)
        d_fake, cls_fake = D(y_fake)
        g_adv = bce(d_fake, real_lbl)
        g_cls = ce(cls_fake, c.argmax(1))
        g_l1  = l1(y_fake, y)*LAMBDA_L1
        g_loss = g_adv*1.0 + g_cls*4.0 + g_l1*50
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        if i%20==0:
            print(f"[E{epoch}/{EPOCHS}] [B{i}] D:{d_loss.item():.2f} G:{g_loss.item():.2f}")

    # ── 샘플 저장 ──
    G.eval()
    with torch.no_grad():
        xs,_,cs = next(iter(loader))
        sample = G(xs.to(DEVICE)[:4], cs.to(DEVICE)[:4])
        out = (sample * 0.5 + 0.5).clamp(0,1)
        torchvision.utils.save_image(
            out,
            os.path.join(save_dir,f"epoch_{epoch:03d}.png"), nrow=4
        )
    G.train()