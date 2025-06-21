import os
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from MyDataSet import MyDataSet
import setting
import torch
import torch.nn as nn
from HeatNet import HeatNet

device = None
try:
    import torch_directml

    device = torch_directml.device()
except Exception as e:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0：训练；1：在已有模型参数上训练
MODE = 0

# 控制参数
heat_net_version = 5
heat_net_train_ecpoch = 20

net_version = 0
train_ecpoch = 0


# 超参
HEAT_NET_PARAM_PATH = (
    f"{setting.net_path}/heat_net{heat_net_version}_{heat_net_train_ecpoch}.pth"
)

DATASET_PATH = f"{setting.net_path}/click_hold_net{net_version}_{train_ecpoch}.pth"
EPOCHS = 1
BATCH_SIZE = 1024

import torch
import torch.nn.functional as F
import torch.nn as nn


class ClickHoldNet(nn.Module):
    def __init__(self):
        super().__init__()
        h, w = 72, 96

        # click分支
        self.click_predict = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        # hold分支
        self.hold_predict = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, heatmap, frames):
        """
        frames: (B, 1, H, W)
        heatmap: (B, 1, H, W)
        """
        B, _, H, W = frames.shape

        # ---- Padding操作 ----
        pad = 16  # 足够大的padding，确保crop不越界
        frames_padded = F.pad(frames, pad=(pad, pad, pad, pad), mode="replicate")
        heatmap_padded = F.pad(heatmap, pad=(pad, pad, pad, pad), mode="replicate")

        # ---- 找最大点坐标 ----
        max_vals, max_idxs = torch.max(heatmap.view(B, -1), dim=1)
        max_rows = max_idxs // W
        max_cols = max_idxs % W

        crops = []
        crop_size = 32  # 可以根据需求微调
        half_crop = crop_size // 2

        for b in range(B):
            r, c = max_rows[b] + pad, max_cols[b] + pad  # offset for padding
            r_start = r - half_crop
            r_end = r + half_crop
            c_start = c - half_crop
            c_end = c + half_crop

            # 裁剪patch并resize到原图大小
            crop = frames_padded[b : b + 1, :, r_start:r_end, c_start:c_end]
            crop_resized = F.interpolate(
                crop, size=(H, W), mode="bilinear", align_corners=False
            )
            crops.append(crop_resized)

        click_input = torch.cat(crops, dim=0)  # (B, 1, H, W)

        # ---- 推理 ----
        click_out = self.click_predict(click_input).squeeze(-1)
        hold_out = self.hold_predict(frames).squeeze(-1)

        return click_out, hold_out


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-7  # 防止log(0)

    def forward(self, pred, target):
        """
        pred: (B,) sigmoid输出
        target: (B,) 0或1
        """
        pred = pred.clamp(min=self.eps, max=1.0 - self.eps)  # 防止log(0)

        pos_loss = -self.alpha * ((1 - pred) ** self.gamma) * target * torch.log(pred)
        neg_loss = (
            -(1 - self.alpha) * (pred**self.gamma) * (1 - target) * torch.log(1 - pred)
        )

        loss = pos_loss + neg_loss
        return loss.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, batchsize=BATCH_SIZE, epochs=EPOCHS):
        super().__init__()
        self.click_loss = FocalLoss(alpha=0.25, gamma=2)
        self.hold_loss = nn.BCELoss()  # 也可以换成FocalLoss
        self.total_batch_size = batchsize * epochs

    def forward(self, epoch, click_pred, click_gt, hold_pred, hold_gt):
        click_loss = self.click_loss(click_pred, click_gt)
        hold_loss = self.hold_loss(hold_pred, hold_gt)

        return click_loss * 0.5 + hold_loss * 0.5


def train(parameter_path=None):
    heat_net = HeatNet().to(device)
    heat_net.load_state_dict(
        torch.load(HEAT_NET_PARAM_PATH, weights_only=True, map_location=device)
    )
    heat_net.eval()

    net = ClickHoldNet().to(device)
    if parameter_path is not None:
        try:
            net.load_state_dict(torch.load(parameter_path, weights_only=True))
        except:
            print("can't find dataset_parameter")
            exit()

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
                # 代码测试仅使用一个
                break

        except Exception as e:
            print("error in map " + dir + str(e))
            continue

    concat = ConcatDataset(map_sets)
    test_ratio = 0.2
    test_set, train_set = torch.utils.data.random_split(
        concat,
        [int(len(concat) * test_ratio), len(concat) - int(len(concat) * test_ratio)],
    )
    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=BATCH_SIZE)

    print("dataset size: " + str(len(train_loader) * BATCH_SIZE))

    avg_train_losses = []
    avg_test_losses = []

    for epoch in range(EPOCHS):
        net.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for heat, pic, click, hold in pbar:
            pic = pic.to(device)
            click = click.to(device)
            hold = hold.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                heat_out = heat_net(pic)
            click_predict, hold_predict = net(heat_out, pic)

            loss = mtloss(epoch, click_predict, click, hold_predict, hold)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(
            f"Epoch {epoch+1} finished. Average loss: {total_loss/len(train_loader):.6f}"
        )

        # 推理测试集并记录损失
        net.eval()
        test_loss = 0
        with torch.no_grad():
            for heat, pic, click, hold in test_loader:
                pic = pic.to(device)
                click = click.to(device)
                hold = hold.to(device)
                click_predict, hold_predict = net(heat_net(pic), pic)
                loss = mtloss(click_predict, click, hold_predict, hold)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Test loss: {avg_test_loss:.6f}")

        avg_train_losses.append(total_loss / len(train_loader))
        avg_test_losses.append(avg_test_loss)

        # 绘制损失曲线
        plt.figure(figsize=(8, 5))
        plt.plot(avg_train_losses, "-o", label="Train Loss")
        plt.plot(avg_test_losses, "-o", label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Test Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()

        # 保存训练好的权重，每训练一轮保存一次
        torch.save(
            net.state_dict(),
            f"{setting.net_path}/click_hold_net{net_version}_{epoch +train_ecpoch + 1}.pth",
        )
        print(
            f"Model saved to click_hold_net{net_version}_{epoch + train_ecpoch + 1}.pth"
        )


if __name__ == "__main__":
    if MODE == 0:
        train()
    elif MODE == 1:
        train(DATASET_PATH)
