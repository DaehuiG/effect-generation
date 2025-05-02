# infer_seq.py  ──────────────────────────────────────────────────
import os, argparse, glob, imageio
import torch, torchvision
import torchvision.transforms as TV
from PIL import Image
from model import GESGAN_G          # 학습 때와 동일한 클래스
from dataset import IMG_SIZE, MAX_T # 동일 상수

# ──────── CLI ────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="구조 PNG 또는 구조 시퀀스 폴더")
parser.add_argument("--model", required=True, help="학습된 Generator state_dict(.pt)")
parser.add_argument("--styles", nargs="+", default=["arcane","fireball","slash1"],
                    help="학습 때 사용한 스타일 순서와 동일하게 지정")
parser.add_argument("--out", default="infer_out", help="출력 폴더")
parser.add_argument("--gif", action="store_true", help="스타일별 GIF도 저장")
args = parser.parse_args()

device     = "cuda" if torch.cuda.is_available() else "cpu"
style_dim  = len(args.styles)
os.makedirs(args.out, exist_ok=True)

# ──────── 모델 로드 ─────────────────────────────────────────────
G = GESGAN_G(style_dim).to(device)
G.load_state_dict(torch.load(args.model, map_location=device))
G.eval()

# ──────── 구조 시퀀스 불러오기 ─────────────────────────────────
def load_rgba(path):
    trans = TV.Compose([
        TV.Resize((IMG_SIZE, IMG_SIZE), interpolation=TV.InterpolationMode.BICUBIC),
        TV.ToTensor(), TV.Normalize([.5]*4, [.5]*4)
    ])
    return trans(Image.open(path).convert("RGBA"))

if os.path.isdir(args.input):
    frame_paths = sorted(glob.glob(os.path.join(args.input, "*.png")))
else:
    frame_paths = [args.input]

# 단일 PNG이면 MAX_T 길이로 반복
if len(frame_paths) == 1:
    frame_paths *= MAX_T

# 프레임 수가 MAX_T보다 짧으면 마지막 프레임 복사로 패딩
if len(frame_paths) < MAX_T:
    frame_paths += [frame_paths[-1]] * (MAX_T - len(frame_paths))

x_seq = torch.stack([load_rgba(p) for p in frame_paths[:MAX_T]]).unsqueeze(0).to(device)  # (1,T,4,H,W)
T_frames = x_seq.size(1)

# ──────── 스타일별 생성 ────────────────────────────────────────
for idx, sty in enumerate(args.styles):
    c = torch.zeros(1, style_dim, device=device); c[0, idx] = 1
    h_prev = None
    out_frames = []

    with torch.no_grad():
        for t in range(T_frames):
            h_prev = G(x_seq[:, t], c, h_prev)
            frame = (h_prev * 0.5 + 0.5).clamp(0, 1)        # (1,4,H,W)
            out_frames.append(frame.cpu())

            # 개별 PNG 저장
            torchvision.utils.save_image(frame,
                f"{args.out}/{sty}_{t:03d}.png")

    # 시퀀스 가로 strip 미리보기
    torchvision.utils.save_image(torch.cat(out_frames, 0),
        f"{args.out}/{sty}_strip.png", nrow=T_frames)

    # GIF 저장 (옵션)
    if args.gif:
        gif_frames = [(f[:3]*255).byte().permute(1,2,0).numpy()  # RGB
                      for f in out_frames]
        imageio.mimsave(f"{args.out}/{sty}.gif", gif_frames,
                        duration=0.04, loop=0)

print(f"✓ 완료! 결과 PNG/GIF가 '{args.out}/' 폴더에 저장됐습니다.")

