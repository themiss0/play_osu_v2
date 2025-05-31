import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import setting


class MyDataSet(Dataset):
    def __init__(self, map_path, window_size=16):
        heatmaps = np.load(f"{setting.dataset_path}\\{map_path}\\heats.npy")
        pics = np.load(f"{setting.dataset_path}\\{map_path}\\pics.npy")
        times = np.load(f"{setting.dataset_path}\\{map_path}\\times.npy")
        clicks = np.load(f"{setting.dataset_path}\\{map_path}\\clicks.npy")

        self.heatmaps = torch.from_numpy(heatmaps).float()
        self.clicks = torch.from_numpy(clicks).float()
        self.pics = torch.from_numpy(pics).float() / 255
        self.times = torch.from_numpy(times).float()

        self.window_size = window_size
        self.seq_len = len(self.heatmaps)

    def __len__(self):
        return self.seq_len - self.window_size

    def __getitem__(self, idx):
        start = idx
        end = idx + self.window_size

        heat_seq = self.heatmaps[start:end]  # (W, H, W)
        pic_seq = self.pics[start:end]
        click_seq = self.clicks[start:end]

        # 加 channel 维度 (W, H, W) -> (W, 1, H, W)
        if heat_seq.ndim == 3:
            heat_seq = heat_seq.unsqueeze(1)

        if pic_seq.ndim == 3:
            pic_seq = pic_seq.unsqueeze(1)

        return [
            heat_seq,  # (W, 1, H, W)
            pic_seq,  # (W, 1, H, W)
            click_seq,  # (W,)
        ]
