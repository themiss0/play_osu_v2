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
from MyDataSet import MyDataSet
import setting
from Camera import Camera
import torch
import torch.nn as nn
import torchvision


# 超参
batch_size = 32
net_version = 1
input_len = 16
hidden_size = 128
status_len = 16
EPOCHS = 1

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch_directml.device()


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        height, weight = setting.heatmap_size
        self.cnn = torchvision.models.resnet18(
            torchvision.models.ResNet18_Weights.DEFAULT
        )
        self.cnn.conv1 = torch.nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.cnn.fc = torch.nn.Identity()
        self.rnn = torch.nn.RNN(
            512,
            hidden_size,
            batch_first=True,
        )
        self.rnn_norm = torch.nn.LayerNorm((128))
        self.heat_predict = nn.Sequential(
            nn.Linear(hidden_size, 4 * 18 * 24),
            nn.ReLU(),
            nn.Unflatten(-1, (4, 18, 24)),
            nn.ConvTranspose2d(4, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.click_predict = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, frames, status=[0.0 for i in range(status_len)]):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        frames = torch.nn.functional.interpolate(
            frames, size=(224, 224), mode="bilinear", align_corners=False
        )  # (B, T, 1, 224, 224)
        frames = self.cnn(frames)
        frames = frames.view(B, T, -1)  # (B, T, 512)
        rnn_out, _ = self.rnn(frames)
        rnn_out = self.rnn_norm(rnn_out[:, -1])  # (B, hiddensize)

        heatmap_predict = self.heat_predict(rnn_out)
        click_predict = self.click_predict(rnn_out)
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
    map_sets.append(MyDataSet("Zeke and Luther Theme Song (TV Size) - Smoke's Insane"))

    loader = DataLoader(ConcatDataset(map_sets), shuffle=True, batch_size=batch_size)

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

            loss = mtloss(heat_predict, heat[:, -1], click_predict, click[:, -1])
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
    module_input = [
        torch.zeros(
            setting.heatmap_size[1],
            setting.heatmap_size[0],
            dtype=torch.float32,
        )
        for _ in range(input_len)
    ]

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

            if len(module_input) < input_len:
                continue
            module_input.pop(0)
            module_input.append(pic)

            input = torch.stack(module_input).unsqueeze(1).unsqueeze(0).to(device)
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
    test(f"{setting.net_path}/heatmap_regression_net{net_version}_{EPOCHS}.pth")
