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


device = None
try:
    import torch_directml

    device = torch_directml.device()
except Exception as e:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0：训练；1：在已有模型参数上训练
MODE = 0

# 控制参数
net_version = 4
train_ecpoch = 0
DATASET_PATH = (
    f"{setting.net_path}/heat_net{net_version}_{train_ecpoch}.pth"
)

# 超参
EPOCHS = 1
BATCH_SIZE = 64


class HeatNet(nn.Module):
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

        return heatmap_predict


def train(parameter_path=None):

    net = HeatNet().to(device)
    if parameter_path is not None:
        try:
            net.load_state_dict(torch.load(parameter_path), weights_only=True)
        except:
            print("can't find dataset_parameter")
            exit()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    mtloss = nn.MSELoss().to(device)

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
            heat = heat.to(device)
            pic = pic.to(device)
            optimizer.zero_grad()

            heat_predict = net(pic)

            loss = mtloss(heat_predict, heat)
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
                heat = heat.to(device)
                pic = pic.to(device)
                heat_predict = net(pic)
                loss = mtloss(heat_predict, heat)
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
            f"{setting.net_path}/heat_net{net_version}_{epoch +train_ecpoch + 1}.pth",
        )
        print(
            f"Model saved to heat_net{net_version}_{epoch + train_ecpoch + 1}.pth"
        )


if __name__ == "__main__":
    if MODE == 0:
        train()
    elif MODE == 1:
        train(DATASET_PATH)
