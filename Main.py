import os
import time
import cv2
import torch
import numpy as np
import setting
import torch
import pygetwindow as gw
from Controller import Controller
from Transformer import Transformer
from Transformer import CNN
from RuntimeViewer import RuntimeViewer
from Camera import Camera
from MemReader import MemReader
import torch_directml


# 控制参数
net_version = 1
net_train_ecpoch = 1
close_click_net = False  # 是否关闭点击网络

device = torch_directml.device()

NET_PARAM_PATH = f"{setting.net_path}/transformer{net_version}_{net_train_ecpoch}.pth"
CNN_PARAM_PATH = (
    f"{setting.net_path}/transformer{net_version}_cnn_{net_train_ecpoch}.pth"
)


def test(net_path, cnn_path):
    cam = Camera()
    mem = MemReader()
    net = Transformer(261).to(device)
    net.load_state_dict(torch.load(net_path, map_location="cpu"))
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))

    joy = Controller(gw.getWindowsWithTitle(setting.window_name)[0])

    # 两个最后保存为history
    pics = []
    commands = []

    # 大小为5的窗口
    pic_window = []
    command_window = []

    for i in range(5):
        pic_window.append(torch.zeros((224, 224)).to(device))
        command_window.append(torch.rand([5]).to(device))

    net.eval()
    runtimeViewer = RuntimeViewer()

    with torch.no_grad():
        while True:
            mem.update()
            if mem.status != "play":
                print("waiting play...")
            else:
                break

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

            # 截图处理
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            pic = cv2.resize(
                pic,
                (224, 224),
                interpolation=cv2.INTER_AREA,
            )  # (224, 224)

            # 先跑上一次预测的操作
            runtimeViewer.update_frame(pic_window[-1], command_window[-1])

            joy.move_to_game_pos(command_window[-1][:2])

            click_now = False
            if not close_click_net:
                if not click_now:
                    if command_window[-1][2] > 0.5:
                        joy.hold()
                        click_now = True
                else:
                    if click_now:
                        if command_window[-1][3] < 0.5:
                            joy.unhold()

            print(f"Time: {mem.time}, Command: {command_window[-1]}")

            # 再推理下一帧操作
            pic = torch.from_numpy(pic).float().to(device) / 255  # (224, 224)
            pic_window.append(pic)

            while len(pic_window) > 5:

                pics.append(pic_window.pop(0))

            pic_in = torch.stack(pic_window[:3]).unsqueeze(0).to(device)
            pic_in = cnn(pic_in).unsqueeze(0)

            while len(command_window) > 5:
                commands.append(command_window.pop(0))

            command_in = torch.stack(command_window[:3])
            command_in = command_in.unsqueeze(0)

            command_in[0][2][4] = command_in[0][2][4] - command_in[0][1][4]
            command_in[0][1][4] = command_in[0][1][4] - command_in[0][0][4]
            command_in[0][0][4] = 0
            command_in = command_in.to(device)

            out = net(pic_in, command_in).squeeze(0)  # (4)
            out = torch.cat([out, torch.Tensor([last]).to(device)], -1)

            command_window.append(out)

            mem.update()

    print("song over")
    pics = np.stack(pics)
    commands = np.stack(commands)
    np.save(f"history/pics.npy", pics)
    np.save(f"history/commands.npy", commands)


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
    test(NET_PARAM_PATH, CNN_PARAM_PATH)
