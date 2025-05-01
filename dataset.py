# dataset.py -----------------------------------------------------------
import os, glob, random
from PIL import Image
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

IMG_SIZE = 128          # 수정 가능
EXT = ("*.png", "*.jpg", "*.jpeg")

def _list_imgs(folder: str) -> List[str]:
    files = []
    for e in EXT:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)

class EffectDataset(Dataset):
    """dataset/<style_name>/*.png  구조를 자동 인식"""
    def __init__(self, root="dataset", img_size: int = IMG_SIZE):
        self.samples: List[Tuple[str, int]] = []
        self.styles: List[str] = sorted([d for d in os.listdir(root)
                                         if os.path.isdir(os.path.join(root, d))])
        for idx, style in enumerate(self.styles):
            for f in _list_imgs(os.path.join(root, style)):
                self.samples.append((f, idx))

        self.resize_norm = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5]*4, [0.5]*4)   # RGBA 4-채널
        ])

        self.num_styles = len(self.styles)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_idx = self.samples[idx]

        # 이미지 열고 RGBA 변환
        img = Image.open(path).convert("RGBA")

        # 가장 짧은 변 기준 중앙 정사각 크롭
        w, h = img.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s))

        # 리사이즈 + 정규화
        img = self.resize_norm(img)

        # one-hot style 벡터
        c = torch.zeros(self.num_styles)
        c[cls_idx] = 1.0
        return img, img, c

def build_loader(batch=8, shuffle=True, root="dataset"):
    ds = EffectDataset(root)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4,
                      pin_memory=True, drop_last=True), ds.num_styles