import time
import setting
import dxcam
import pygetwindow as gw
import cv2 as cv
import numpy as np
from typing import cast


class Camera:

    def __init__(self):
        self.cam = dxcam.create(output_color="BGR", max_buffer_len=10)
        self.ww = cast(gw.Win32Window, gw.getWindowsWithTitle(setting.window_name)[0])
        # 修改窗口
        self.ww.resizeTo(
            setting.window_resize[0],
            setting.window_resize[1],
        )
        self.ww.moveTo(50, 50)

    def get(self) -> np.array:
        """return windowshot or None if target window out of screen"""
        region = np.array([self.ww.left, self.ww.top, self.ww.right, self.ww.bottom])
        if (
            self.ww.isMinimized
            # window out of screen
            or np.any(region < 0)
            or region[2] >= setting.monitor_size[0]
            or region[3] >= setting.monitor_size[1]
        ):
            print("Window out of screen")
            return None
        pic = self.cam.grab(region)
        if pic is None:
            print("Grab failed")
            return None
        x, y, w, h = setting.play_filed
        pic = pic[y : y + h, x : x + w]
        return pic
