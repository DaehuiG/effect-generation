
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class EffectSequenceDataset(Dataset):
    def __init__(self, root, img_size=128, style_list=None):
        self.root = root
        self.img_size = img_size
        self.folders = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])
        self.samples = []
        for folder in self.folders:
            img_paths = sorted(glob.glob(os.path.join(root, folder, "*.png")))
            if len(img_paths) > 1:
                self.samples.append((img_paths[0], img_paths, folder))  # (x_path, y_paths, style_name)

        self.style_list = style_list if style_list else sorted(set(folder for _, _, folder in self.samples))
        self.style_to_idx = {style: i for i, style in enumerate(self.style_list)}

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_path, y_paths, style_name = self.samples[idx]
        x_img = self.transform(Image.open(x_path).convert("RGB"))
        y_imgs = torch.stack([
            self.transform(Image.open(p).convert("RGB")) for p in y_paths
        ])
        style_idx = self.style_to_idx[style_name]
        style_onehot = torch.zeros(len(self.style_list))
        style_onehot[style_idx] = 1.0
        return x_img, y_imgs, style_onehot
