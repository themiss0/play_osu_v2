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
from RuntimeViewer import RuntimeViewer


# 0：训练；1：在已有模型参数上训练；2：推理
MODE = 0

# 控制参数
net_version = 3
train_ecpoch = 0
DATASET_PATH = (
    f"{setting.net_path}/heatmap_regression_net{net_version}_{train_ecpoch}.pth"
)

# 超参
EPOCHS = 5
BATCH_SIZE = 128
# 在训练时click分支的loss权重会逐步增加，这是权重范围
SMOOTH_MULTI_LOSS_WEIGHT_RANGE = [0.0, 0.0]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch_directml.device()


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # (B, 1, H, W)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, H, W)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)  # (B, 64, H/2, W/2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 128, H/2, W/2)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.pool2 = nn.MaxPool2d(2)  # (B, 128, H/4, W/4)

        # heat decode
        self.up1 = nn.ConvTranspose2d(
            128, 128, kernel_size=2, stride=2  # (B, 128, H/2, W/2)
        )

        self.conv_up1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),  # (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # (B, 64, H, W)

        self.conv_up2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),  #
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )  # (B, 32, H, W)

        self.heat_predict = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )  # (B, 1, H, W)

        # click decode
        self.click_predict = torch.nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, 128, 1, 1)
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, frames):
        x1 = self.conv1(frames)  # (B, 32, H, W)
        x2 = self.conv2(x1)  # (B, 64, H, W)
        x3 = self.pool1(x2)  # (B, 64, H/2, W/2)
        x4 = self.conv3(x3)  # (B, 128, H/2, W/2)
        x5 = self.pool2(x4)  # (B, 128, H/4, W/4)

        u1 = self.up1(x5)  # (B, 128, H/2, W/2)
        u1 = torch.cat((u1, x4), dim=1)  # (B, 256, H/2, W/2)
        u2 = self.conv_up1(u1)  # (B, 64, H/2, W/2)
        u3 = self.up2(u2)  # (B, 32, H, W)
        u3 = torch.cat((u3, x2), dim=1)  # (B, 96, H, W)
        u4 = self.conv_up2(u3)  # (B, 32, H, W)

        heatmap_predict = self.heat_predict(u4)

        # 暂时砍掉了click分支
        # click_predict = self.click_predict(frames)
        return [heatmap_predict, torch.zeros(heatmap_predict.shape[0])]


# === 多任务损失 ===
class MultiTaskLoss(nn.Module):
    def __init__(self, batchsize=BATCH_SIZE, epochs=EPOCHS):
        super().__init__()
        self.heat_loss = nn.MSELoss()
        self.click_loss = nn.BCELoss()
        self.heat_weight = 1 - SMOOTH_MULTI_LOSS_WEIGHT_RANGE[0]
        self.click_weight = SMOOTH_MULTI_LOSS_WEIGHT_RANGE[0]
        self.weight_sub = self.heat_weight - self.click_weight
        self.total_batch_size = batchsize * epochs

    def forward(self, heat_pred, heat_gt, click_pred, click_gt):
        loss1 = self.heat_loss(heat_pred, heat_gt) * self.heat_weight

        return loss1
        # loss2 = (
        #     self.click_loss(click_pred.squeeze(), click_gt.squeeze())
        #     * self.click_weight
        # )

        # weight_offset = self.weight_sub * 1 / self.total_batch_size
        # self.click_weight -= weight_offset
        # self.heat_weight += weight_offset

        # return loss1 + loss2


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
                _ = MyDataSet(dir)
                if len(_) < 1:
                    print(f"Dataset {dir} is empty, skipping.")
                    continue
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

        # 保存训练好的权重，每训练一轮保存一次
        torch.save(
            net.state_dict(),
            f"{setting.net_path}/heatmap_regression_net{net_version}_{epoch +train_ecpoch}.pth",
        )
        print(
            f"Model saved to heatmap_regression_net{net_version}_{epoch + train_ecpoch}.pth"
        )


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
    runtimeViewer = RuntimeViewer()

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

        while mem.status == "play" or mem.time < mem.end:
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
            runtimeViewer.update_frame(pics[-1], heatmaps[-1], clicks[-1])

            pos, _ = get_peak_position(heat_predict[0][0])
            joy.move_to_game_pos(pos)
            if click_predict > 0.5:
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

    if MODE == 0:
        train()
    elif MODE == 1:
        train(DATASET_PATH)
    elif MODE == 2:
        test(DATASET_PATH)
