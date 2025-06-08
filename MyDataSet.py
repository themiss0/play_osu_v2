import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import setting


class MyDataSet(Dataset):
    def __init__(self, map_path, window_size=16):
        heatmaps = np.load(f"{map_path}/heats.npy")
        pics = np.load(f"{map_path}/pics.npy")
        times = np.load(f"{map_path}/times.npy")
        clicks = np.load(f"{map_path}/clicks.npy")
        holds = np.load(f"{map_path}/holds.npy")

        self.heatmaps = torch.from_numpy(heatmaps).float()
        self.clicks = torch.from_numpy(clicks).float()
        self.holds = torch.from_numpy(holds).float()
        self.pics = torch.from_numpy(pics).float() / 255
        self.times = torch.from_numpy(times).float()

        self.window_size = window_size
        self.seq_len = len(self.heatmaps)

    def __len__(self):
        return self.seq_len - self.window_size

    def __getitem__(self, idx):
        pic = self.pics[idx]
        heatmap = self.heatmaps[idx]
        click = self.clicks[idx]
        hold = self.holds[idx]
        time = self.times[idx]

        # 加 channel 维度
        if pic.ndim == 2:
            pic = pic.unsqueeze(0)  # (1, H, W)

        if heatmap.ndim == 2:
            heatmap = heatmap.unsqueeze(0)  # (1, H, W)

        return [heatmap, pic, click, hold]
