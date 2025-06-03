import os
import matplotlib.pyplot as plt
import time
import cv2
import torch.utils
from Controller import Controller
import pygetwindow as gw
from MemReader import MemReader
import torch
import torch_directml
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from CnnDataSet import MyDataSet
import setting
from Camera import Camera
import torch
import torch.nn as nn


# 超参
BATCH_SIZE = 32
net_version = 1
EPOCHS = 1

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch_directml.device()


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # (B, 1, H, W)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, H, W)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # downsample (B, 64, H/2, W/2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 128, H/2, W/2)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # downsample (B, 128, H/4, W/4)
        )

        self.heat_predict = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.click_predict = torch.nn.Sequential(
            torch.nn.Flatten(1, -1),
            torch.nn.Linear(
                int(128 * setting.heatmap_size[0] / 4 * setting.heatmap_size[1] / 4),
                256,
            ),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, frames):
        frames = self.cnn(frames)

        heatmap_predict = self.heat_predict(frames)
        click_predict = self.click_predict(frames)
        return [heatmap_predict, click_predict]


# === 多任务损失 ===
class MultiTaskLoss(nn.Module):
    def __init__(self, heat_weight=1.0, click_weight=1.0):
        super().__init__()
        self.heat_loss = nn.MSELoss()
        self.click_loss = nn.BCELoss()
        self.heat_weight = heat_weight
        self.click_weight = click_weight

    def forward(self, heat_pred, heat_gt, click_pred, click_gt):
        loss1 = self.heat_loss(heat_pred, heat_gt)
        loss2 = self.click_loss(click_pred.squeeze(), click_gt.squeeze())
        return self.heat_weight * loss1 + self.click_weight * loss2


def train(parameter_path=None):

    net = Net().to(device)
    if parameter_path is not None:
        try:
            net.load_state_dict(torch.load(parameter_path), weights_only=True)
        except:
            print("can't find dataset_parameter")
            exit

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    mtloss = MultiTaskLoss().to(device)

    # 创建数据集列表并压缩为单个loader
    map_sets = []
    for dir in os.listdir(setting.dataset_path):
        dir = setting.dataset_path + "/" + dir
        try:
            if not os.path.isdir(dir):
                continue
            if os.path.exists(dir + "/heats.npy"):
                map_sets.append(MyDataSet(dir))

        except Exception as e:
            continue

    loader = DataLoader(ConcatDataset(map_sets), shuffle=True, batch_size=BATCH_SIZE)
    print("dataset size: " + str(len(loader) * BATCH_SIZE))

    net.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for heat, pic, click in pbar:
            heat = heat.to(device)
            pic = pic.to(device)
            click = click.to(device)
            optimizer.zero_grad()

            heat_predict, click_predict = net(pic)

            loss = mtloss(heat_predict, heat, click_predict, click)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1} finished. Average loss: {total_loss/len(loader):.6f}")

    # 保存训练好的权重
    torch.save(
        net.state_dict(),
        f"{setting.net_path}/heatmap_regression_net{net_version}_{EPOCHS}.pth",
    )
    print(f"Model saved to heatmap_regression_net{net_version}_{EPOCHS}.pth")


def test(parameter_path):
    cam = Camera()
    mem = MemReader()
    net = Net().to(device)
    net.load_state_dict(torch.load(parameter_path, map_location="cpu"))
    joy = Controller(gw.getWindowsWithTitle(setting.window_name)[0])

    pics = []
    heatmaps = []
    clicks = []
    net.eval()

    with torch.no_grad():
        while True:
            mem.update()
            if mem.status != "play":
                print("waiting play...")
            else:
                break

        print("heatmap loaded")
        mem.update()

        # 等待加载
        last = mem.time
        while last <= mem.time:
            mem.update()

        while mem.status == "play":
            mem.update()

            if mem.time > mem.end:
                print("end")
                break
            if mem.time == last:
                continue
            last = mem.time

            pic = cam.get()

            if pic is None:
                continue
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            pic = cv2.resize(
                pic,
                (setting.heatmap_size[0], setting.heatmap_size[1]),
                interpolation=cv2.INTER_AREA,
            )
            pics.append(pic)
            pic = torch.from_numpy(pic).float() / 255

            input = pic.unsqueeze(0).unsqueeze(0).to(device)
            heat_predict, click_predict = net(input)

            heatmaps.append(heat_predict.squeeze(0).squeeze(0).cpu().numpy())
            clicks.append(click_predict.cpu().numpy())

            if click_predict > 0.5:
                pos, _ = get_peak_position(heat_predict[0][0])
                joy.move_to_game_pos(pos)
                joy.hold()
            else:
                joy.unhold()
            mem.update()
        pass
    print("song over")
    pics = np.stack(pics)
    heatmaps = np.stack(heatmaps)
    clicks = np.stack(clicks)
    np.save(f"history/pics.npy", pics)
    np.save(f"history/clicks.npy", clicks)
    np.save(f"history/heats.npy", heatmaps)


def get_peak_position(heatmap):
    """
    输入：
        heatmap: shape (H, W)
    输出：
        norm_pos: [x, y] 归一化坐标（范围 0~1）
        value: 最大热度值
    """
    heatmap = heatmap.cpu().numpy()
    h, w = heatmap.shape
    max_index = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    max_y, max_x = max_index

    norm_x = max_x / w
    norm_y = max_y / h

    return [norm_x, norm_y], heatmap[max_y, max_x]


if __name__ == "__main__":
    # train(f"{setting.net_path}/heatmap_regression_net{net_version}_{EPOCHS}.pth")
    # train()
    test(f"{setting.net_path}/heatmap_regression_net{net_version}_{EPOCHS}.pth")
