import time
import cv2
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


net_version = 10
input_len = 16
hidden_size = 128

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch_directml.device()


class Net(nn.Module):

    def __init__(self):
        # 16 * 1 * 96 * 72
        self.img_size = setting.heatmap_size
        super(Net, self).__init__()
        self.input_len = input_len
        self.rnn_input_size = self.img_size[0] * self.img_size[1] * 4

        # 主编码器 -> 空间特征
        self.pic_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # (B*S, 1, H, W) → (B*S, 32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (B*S, 32, H/2, W/2)
            nn.Conv2d(32, 64, 3, padding=1),  # (B*S, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (B*S, 64, H/4, W/4)
        )
        # -> 时序特征
        self.rnn = nn.RNN(self.rnn_input_size, hidden_size, batch_first=True)

        # heatmap分支解码
        self.heatmap_decoder = nn.Sequential(
            nn.Linear(hidden_size, self.rnn_input_size),
            nn.Unflatten(1, (64, self.img_size[0] // 4, self.img_size[1] // 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Click分支解码
        self.click_predictor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Click probability
        )

    def forward(self, pics):
        B, T, C, H, W = pics.shape
        pics = pics.view(B * T, C, H, W)
        pic_feat = self.pic_encoder(pics)  # (B*T, 16, H/4, W/4)
        # Flatten
        pic_feat = pic_feat.view(B, T, -1)

        # RNN temporal encoder
        rnn_out, _ = self.rnn(pic_feat)  # (B, T, hidden_size)
        last_out = rnn_out[:, -1]  # Last frame's feature (B, H)

        heat_pred = self.heatmap_decoder(last_out).view(B, 1, H, W)
        click_prob = self.click_predictor(last_out).squeeze(1)  # (B,)

        return heat_pred, click_prob


# === 多任务损失 ===
class MultiTaskLoss(nn.Module):
    def __init__(self, heat_weight=1.0, click_weight=1.0):
        super().__init__()
        self.heat_loss = nn.BCELoss()
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

    loader = DataLoader(ConcatDataset(map_sets), shuffle=True, batch_size=32)

    net.train()
    EPOCHS = 5
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
        f"{setting.net_path}\\heatmap_regression_net{net_version + EPOCHS}.pth",
    )
    print(f"Model saved to heatmap_regression_net{net_version + EPOCHS}.pth")


def test(parameter_path):
    cam = Camera()
    mem = MemReader()
    net = Net().to(device)
    net.load_state_dict(torch.load(parameter_path))
    joy = Controller(gw.getWindowsWithTitle(setting.window_name)[0])
    module_input = [
        torch.zeros(
            setting.heatmap_size[1],
            setting.heatmap_size[0],
            dtype=torch.float32,
        )
        for _ in range(input_len)
    ]

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
            print(f"mem time: {mem.time}")
            last = mem.time

            start = time.time()
            pic = cam.get()

            if pic is None:
                continue
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            pic = cv2.resize(
                pic,
                (setting.heatmap_size[0], setting.heatmap_size[1]),
                interpolation=cv2.INTER_AREA,
            )
            pic = torch.from_numpy(pic).float() / 255

            if len(module_input) < input_len:
                continue
            module_input.pop(0)
            module_input.append(pic)

            input = torch.stack(module_input).unsqueeze(1).unsqueeze(0).to(device)
            heat_predict, click_predict = net(input)

            if click_predict > 0.5:
                pos, _ = get_peak_position(heat_predict[0][0])
                joy.move_to_game_pos(pos)
                joy.hold()
                print(f"click at {pos} at time{mem.time}")
            else:
                joy.unhold()
                print(
                    f"click_predict {click_predict} too low, not click at time{mem.time}"
                )
            mem.update()
        pass
    print("song over")


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
    max_y, max_x = max_index  # 注意：行是y，列是x

    norm_x = max_x / w
    norm_y = max_y / h

    return [norm_x, norm_y], heatmap[max_y, max_x]


if __name__ == "__main__":
    # train(f"{setting.net_path}\\heatmap_regression_net{net_version}.pth")
    test(f"{setting.net_path}\\heatmap_regression_net{net_version}.pth")
