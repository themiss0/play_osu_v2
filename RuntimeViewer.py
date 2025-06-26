import setting
import cv2
import numpy as np


class RuntimeViewer:
    """
    HeatmapViewer: 在推理循环中可每次调用 update_frame() 实时显示一帧。
    """

    def __init__(self, resize_factor=8):
        self.resize_factor = resize_factor
        self.COLORMAPS = {
            0: cv2.COLORMAP_JET,
            1: cv2.COLORMAP_HOT,
            2: cv2.COLORMAP_TURBO,
            3: cv2.COLORMAP_BONE,
            4: cv2.COLORMAP_MAGMA,
        }
        self.initialize()

    def initialize(self):
        """
        初始化窗口和滑动条。
        """
        cv2.namedWindow("Heatmap Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Heatmap Preview", 800, 600)
        cv2.moveWindow("Heatmap Preview", 1340, 50)
        cv2.createTrackbar(
            "Overlay Heatmap (%)", "Heatmap Preview", 30, 100, lambda x: None
        )
        cv2.createTrackbar(
            "Overlay Image (%)", "Heatmap Preview", 70, 100, lambda x: None
        )
        cv2.createTrackbar(
            "Colormap",
            "Heatmap Preview",
            2,
            len(self.COLORMAPS) - 1,
            lambda x: None,
        )
        self.update_frame(
            np.zeros(setting.heatmap_size, dtype=np.uint8),
            np.zeros(setting.heatmap_size, dtype=np.uint8),
        )

    def update_frame(self, gray_img, heatmap, click=0, hold=0, *args):

        return
        alpha = cv2.getTrackbarPos("Overlay Image (%)", "Heatmap Preview") / 100.0
        beta = cv2.getTrackbarPos("Overlay Heatmap (%)", "Heatmap Preview") / 100.0
        cmap_index = cv2.getTrackbarPos("Colormap", "Heatmap Preview")
        cmap = self.COLORMAPS.get(cmap_index, cv2.COLORMAP_TURBO)

        img_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        heatmap_boosted = np.clip(heatmap, 0, 1)
        heatmap_colored = cv2.applyColorMap(
            (heatmap_boosted * 255).astype(np.uint8), cmap
        )

        h, w = gray_img.shape
        hm_resized = cv2.resize(heatmap_colored, (w, h), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(img_color, alpha, hm_resized, beta, 0)
        display = cv2.resize(
            overlay,
            (w * self.resize_factor, h * self.resize_factor),
            interpolation=cv2.INTER_NEAREST,
        )

        cv2.putText(
            display,
            "click:" + str(click),
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            "O" * int(click / 0.1),
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # hold
        cv2.putText(
            display,
            f"hold: {hold}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            "H" * int(hold / 0.1),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Heatmap Preview", display)

        cv2.imshow("Heatmap Preview", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.close()

    def close(self):
        if self.initialized:
            cv2.destroyWindow("Heatmap Preview")
            self.initialized = False
