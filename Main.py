import os
import time
import cv2
import torch
import numpy as np
import setting
import torch
from HeatNet import HeatNet
from ClickNet import ClickNet
import pygetwindow as gw
from RuntimeViewer import RuntimeViewer
from Controller import Controller
from Camera import Camera
from MemReader import MemReader
import torch_directml


# 控制参数
close_heat_net = False
close_click_net = True
heat_net_version = 5
heat_net_train_ecpoch = 20

click_net_version = 0
click_net_train_ecpoch = 2

device = torch_directml.device()


HEAT_NET_PARAM_PATH = (
    f"{setting.net_path}/heat_net{heat_net_version}_{heat_net_train_ecpoch}.pth"
)
CLICK_NET_PARAM_PATH = (
    f"{setting.net_path}/click_net{click_net_version}_{click_net_train_ecpoch}.pth"
)


def test(heat_net_path, click_net_path):
    cam = Camera()
    mem = MemReader()
    heat_net = HeatNet().to(device)
    click_net = ClickNet().to(device)
    if not close_heat_net:
        heat_net.load_state_dict(torch.load(heat_net_path, map_location="cpu"))
    if not close_click_net:
        click_net.load_state_dict(torch.load(click_net_path, map_location="cpu"))

    joy = Controller(gw.getWindowsWithTitle(setting.window_name)[0])

    pics = []
    heatmaps = []
    clicks = []
    holds = []
    heat_net.eval()
    click_net.eval()
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
            heat_predict = heat_net(input)
            click_predict, hold_predict = click_net(input)

            heatmaps.append(heat_predict.squeeze(0).squeeze(0).cpu().numpy())
            clicks.append(click_predict.cpu().numpy())
            holds.append(hold_predict.cpu().numpy())
            runtimeViewer.update_frame(pics[-1], heatmaps[-1], clicks[-1], holds[-1])

            pos, _ = get_peak_position(heat_predict[0][0])
            joy.move_to_game_pos(pos)
            click_now = False
            if not close_click_net:
                if not click_now:
                    if click_predict > 0.5:
                        joy.hold()
                        click_now = True
                else:
                    if click_now:
                        if hold_predict < 0.5:
                            joy.unhold()
            mem.update()
        pass
    print("song over")
    pics = np.stack(pics)
    heatmaps = np.stack(heatmaps)
    holds = np.stack(holds)
    clicks = np.stack(clicks)
    np.save(f"history/pics.npy", pics)
    np.save(f"history/clicks.npy", clicks)
    np.save(f"history/heats.npy", heatmaps)
    np.save(f"history/holds.npy", holds)


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
    test(HEAT_NET_PARAM_PATH, CLICK_NET_PARAM_PATH)
