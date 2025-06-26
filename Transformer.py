import os
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
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
net_name = "transformer"
net_version = 2
train_ecpoch = 0
DATASET_PATH = f"{setting.net_path}/{net_name}{net_version}_{train_ecpoch}.pth"

# 超参
EPOCHS = 1
BATCH_SIZE = 16


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, map_path, window_size=5):
        pics = np.load(f"{map_path}/pics.npy")
        times = np.load(f"{map_path}/times.npy")
        replays = np.load(f"{map_path}/replays.npy")

        self.replays = torch.from_numpy(replays).float()
        self.pics = torch.from_numpy(pics).float() / 255
        self.times = torch.from_numpy(times).float()

        self.window_size = window_size
        self.seq_len = len(self.times)

    def __len__(self):
        return self.seq_len - self.window_size

    def __getitem__(self, idx):
        pics = self.pics[idx : idx + self.window_size]
        replays = self.replays[idx : idx + self.window_size]
        times = self.times[idx : idx + self.window_size]

        return pics, replays, times


import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, T, transformer_dim=128, nhead=4, nlayers=2):
        super().__init__()
        self.pic_fc = nn.Linear(256, transformer_dim)  # CNN 输出维度
        self.meta_fc = nn.Linear(5, transformer_dim)  # 重放数据维度
        self.input_fc = nn.Linear(2 * transformer_dim, transformer_dim)  # 输入维度

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.output_fc = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, 4),
            nn.Sigmoid(),
        )

    def forward(self, pic_dim, meta_dim):
        """
        pic_dim: (B, 2, 256)
        meta_dim: (B, 2, 5)
        """
        pic_dim = self.pic_fc(pic_dim)
        meta_dim = self.meta_fc(meta_dim)
        x = self.input_fc(torch.cat([pic_dim, meta_dim], -1))
        x = self.transformer(x)

        x_last = x[:, -1, :]  # (B, transformer_dim)

        output = self.output_fc(x_last)

        return output


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torchvision.models.resnet18(pretrained=True)

        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )  # 将 ResNet 的输出维度改为 256

    def forward(self, x):
        """
        x: (B, 1, H, W)
        """
        # resize到 ResNet 输入大小
        x = torchvision.transforms.functional.resize(x, (224, 224))
        x = x.reshape(-1, 1, 224, 224)
        x = x.repeat(1, 3, 1, 1)

        x = self.cnn(x)
        x = x.view(-1, 256)
        return x


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.coord_loss = nn.MSELoss()
        self.click_loss = nn.BCELoss()
        self.hold_loss = nn.BCELoss()

        # log σ²，初始化为0
        self.log_coord_sigma = nn.Parameter(torch.zeros(()))
        self.log_click_sigma = nn.Parameter(torch.zeros(()))
        self.log_hold_sigma = nn.Parameter(torch.zeros(()))

    def forward(self, predict, tag):
        """
        predict: (B, 4)  - [x, y, click, hold]
        tag: (B, 4)
        """
        coord_pred = predict[:, 0:2]
        click_pred = predict[:, 2]
        hold_pred = predict[:, 3]

        coord_gt = tag[:, 0:2]
        click_gt = tag[:, 2]
        hold_gt = tag[:, 3]

        # 计算每个任务的损失
        coord_loss = self.coord_loss(coord_pred, coord_gt)
        click_loss = self.click_loss(click_pred, click_gt)
        hold_loss = self.hold_loss(hold_pred, hold_gt)

        # 基于不确定性的加权损失
        loss = (
            coord_loss * torch.exp(-self.log_coord_sigma)
            + self.log_coord_sigma
            + click_loss * torch.exp(-self.log_click_sigma)
            + self.log_click_sigma
            + hold_loss * torch.exp(-self.log_hold_sigma)
            + self.log_hold_sigma
        ) * 0.5

        return loss


def train(parameter_path=None):

    net = Transformer(261).to(device)
    cnn = CNN().to(device)

    if parameter_path is not None:
        try:
            net.load_state_dict(torch.load(parameter_path), weights_only=True)
        except:
            print("can't find dataset_parameter")
            exit()

    mtloss = MultiTaskLoss().to(device)
    optimizer = torch.optim.Adam(
        list(net.parameters()) + list(mtloss.parameters()), lr=1e-4
    )

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
    test_ratio = 0.1
    test_set, train_set = torch.utils.data.random_split(
        concat,
        [int(len(concat) * test_ratio), len(concat) - int(len(concat) * test_ratio)],
    )
    train_loader = DataLoader(
        train_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True
    )
    test_loader = DataLoader(
        test_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True
    )

    print("dataset size: " + str(len(train_loader) * BATCH_SIZE))

    avg_train_losses = []
    avg_test_losses = []

    for epoch in range(EPOCHS):
        net.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for pics, replays, times in pbar:
            pics = pics.to(device)
            replays = replays.to(device)
            times = times.to(device)

            optimizer.zero_grad()

            pic_dim = cnn(pics).view(-1, 5, 256)  # (B, 256)

            times = torch.cat(
                [torch.zeros_like(times[:, :1]), times[:, 1:] - times[:, :-1]], dim=1
            )

            meta = torch.cat([replays, times.unsqueeze(-1)], -1)
            out = net(pic_dim[:, :3, :], meta[:, :3, :])  # (B, 4)

            loss = mtloss(out, replays[:, -1, :])  # 使用最后一个时间步的标签
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
            for pics, replays, times in tqdm(
                test_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"
            ):
                pics = pics.to(device)
                replays = replays.to(device)
                times = times.to(device)

                pic_dim = cnn(pics).view(-1, 5, 256)  # (B, 256)

                times = torch.cat(
                    [torch.zeros_like(times[:, :1]), times[:, 1:] - times[:, :-1]],
                    dim=1,
                )

                meta = torch.cat([replays, times.unsqueeze(-1)], -1)
                out = net(pic_dim[:, :3, :], meta[:, :3, :])  # (B, 4)

                loss = mtloss(out, replays[:, -1, :])  # 使用最后一个时间步的标签
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

        torch.save(
            net.state_dict(),
            f"{setting.net_path}/{net_name}{net_version}_{epoch +train_ecpoch + 1}.pth",
        )
        torch.save(
            cnn.state_dict(),
            f"{setting.net_path}/{net_name}{net_version}_cnn_{epoch + train_ecpoch + 1}.pth",
        )
        print(f"Model saved to {net_name}{net_version}_{epoch + train_ecpoch + 1}.pth")


if __name__ == "__main__":
    if MODE == 0:
        train()
    elif MODE == 1:
        train(DATASET_PATH)
