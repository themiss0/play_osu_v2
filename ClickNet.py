# %%
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

device = None
try:
    import torch_directml

    device = torch_directml.device()
except Exception as e:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0：训练；1：在已有模型参数上训练；
MODE = 0

# 控制参数
net_version = 0
train_ecpoch = 0
DATASET_PATH = f"{setting.net_path}/click_net{net_version}_{train_ecpoch}.pth"

# 超参
EPOCHS = 1
BATCH_SIZE = 64


class ClickNet(nn.Module):

    def __init__(self):
        super().__init__()
        # (B, 1, H, W)
        h = 72
        w = 96

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

        self.backbone = nn.Sequential()

        # click decode
        self.click_predict = torch.nn.Sequential(
            self.conv1,
            self.conv2,
            self.pool1,
            # self.conv3,
            # self.pool2,
            nn.AdaptiveAvgPool2d(1),  # (B, 128, 1, 1)
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        # hold decode
        self.hold_predict = torch.nn.Sequential(
            self.conv1,
            self.conv2,
            self.pool1,
            # self.conv3,
            # self.pool2,
            nn.AdaptiveAvgPool2d(1),  # (B, 128, 1, 1)
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, frames):
        click_predict = self.click_predict(frames).squeeze(-1)
        hold_predict = self.hold_predict(frames).squeeze(-1)

        return [click_predict, hold_predict]


# === 多任务损失 ===
class MultiTaskLoss(nn.Module):
    def __init__(self, batchsize=BATCH_SIZE, epochs=EPOCHS):
        super().__init__()
        self.click_loss = nn.BCELoss()
        self.hold_loss = nn.BCELoss()
        self.total_batch_size = batchsize * epochs

    def forward(self, epoch, click_pred, click_gt, hold_pred, hold_gt):
        click_loss = self.click_loss(click_pred, click_gt)
        hold_loss = self.hold_loss(hold_pred, hold_gt)

        return click_loss * 0.5 + hold_loss * 0.5


def train(parameter_path=None):

    net = ClickNet().to(device)
    if parameter_path is not None:
        try:
            net.load_state_dict(torch.load(parameter_path), weights_only=True)
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

            click_predict, hold_predict = net(pic)

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
                click_predict, hold_predict = net(pic)
                loss = mtloss(epoch, click_predict, click, hold_predict, hold)
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
            f"{setting.net_path}/click_net{net_version}_{epoch +train_ecpoch + 1}.pth",
        )
        print(f"Model saved to click_net{net_version}_{epoch + train_ecpoch + 1}.pth")


if __name__ == "__main__":
    if MODE == 0:
        train()
    elif MODE == 1:
        train(DATASET_PATH)

# %%
