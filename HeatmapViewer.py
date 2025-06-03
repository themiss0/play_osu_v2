from matplotlib import pyplot as plt
from MemReader import MemReader
import cv2
import numpy as np
import os


def safe_path(name):
    import re
    # 替换为下划线或直接去掉非法字符
    return re.sub(r'[\\/:*?"<>|]', "", name)

def main():
    mode = 0 # 0为从当前游戏所指谱面，1为推理输出
    if mode == 0:
        mem = MemReader()
        mem.update()
        heatmap_viewer(safe_path(mem.song), safe_path(mem.version))
    else:
        heatmap_viewer()


def show_heatmap_preview_gui(heatmap_data, resize_factor=8):
    def nothing(x):
        pass

    COLORMAPS = {
        0: cv2.COLORMAP_JET,
        1: cv2.COLORMAP_HOT,
        2: cv2.COLORMAP_TURBO,
        3: cv2.COLORMAP_BONE,
        4: cv2.COLORMAP_MAGMA,
    }

    # 检查数据是否为空
    if not heatmap_data or len(heatmap_data) == 0:
        print("错误: 没有可显示的数据")
        return

    # 创建窗口和滑动条
    cv2.namedWindow("Heatmap Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Heatmap Preview", 800, 600)

    # 添加控制滑动条
    cv2.createTrackbar("Frame", "Heatmap Preview", 0, len(heatmap_data) - 1, nothing)
    cv2.createTrackbar("Overlay Heatmap (%)", "Heatmap Preview", 30, 100, nothing)
    cv2.createTrackbar("Overlay Image (%)", "Heatmap Preview", 70, 100, nothing)
    # cv2.createTrackbar("Heat Boost (x)", "Heatmap Preview", 10, 50, nothing)
    cv2.createTrackbar("Colormap", "Heatmap Preview", 2, len(COLORMAPS) - 1, nothing)
    cv2.createTrackbar("FPS", "Heatmap Preview", 30, 500, nothing)

    paused = False
    current_frame = 0

    def update_display(frame_idx):
        # 获取当前帧数据
        gray_img, heatmap, click = heatmap_data[frame_idx]

        # 获取参数
        alpha = cv2.getTrackbarPos("Overlay Image (%)", "Heatmap Preview") / 100.0
        beta = cv2.getTrackbarPos("Overlay Heatmap (%)", "Heatmap Preview") / 100.0
        # heat_boost = cv2.getTrackbarPos("Heat Boost (x)", "Heatmap Preview")
        heat_boost = 1
        cmap_index = cv2.getTrackbarPos("Colormap", "Heatmap Preview")
        cmap = COLORMAPS.get(cmap_index, cv2.COLORMAP_TURBO)

        # 处理图像
        img_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        heatmap_boosted = np.clip(heatmap * heat_boost, 0, 1)
        heatmap_colored = cv2.applyColorMap(
            (heatmap_boosted * 255).astype(np.uint8), cmap
        )

        # Resize和叠加
        h, w = gray_img.shape
        hm_resized = cv2.resize(heatmap_colored, (w, h), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(img_color, alpha, hm_resized, beta, 0)
        display = cv2.resize(
            overlay,
            (w * resize_factor, h * resize_factor),
            interpolation=cv2.INTER_NEAREST,
        )

        cv2.putText(
            display,
            "click:" + str(click),
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            "O" * int(click / 0.1),
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Heatmap Preview", display)

    last_frame_time = 0
    while True:
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        fps = cv2.getTrackbarPos("FPS", "Heatmap Preview")
        frame_time = 1.0 / fps

        if not paused:
            if current_time - last_frame_time >= frame_time:
                update_display(current_frame)
                last_frame_time = current_time
                current_frame = (current_frame + 1) % len(heatmap_data)
                cv2.setTrackbarPos("Frame", "Heatmap Preview", current_frame)
        else:
            update_display(current_frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # Space
            paused = not paused
        elif key == ord("r"):  # R
            current_frame = 0
            cv2.setTrackbarPos("Frame", "Heatmap Preview", 0)
        elif key == ord("b"):  # B
            current_frame = max(
                0, current_frame - 2 if not paused else current_frame - 1
            )
            cv2.setTrackbarPos("Frame", "Heatmap Preview", current_frame)
            paused = True
        elif key == ord("f"):  # F
            current_frame = min(
                len(heatmap_data),
                current_frame + 2 if not paused else current_frame + 1,
            )
            cv2.setTrackbarPos("Frame", "Heatmap Preview", current_frame)
            paused = True

        # 检查滑动条变化
        frame_pos = cv2.getTrackbarPos("Frame", "Heatmap Preview")
        if frame_pos != current_frame:
            current_frame = frame_pos
            paused = True

    cv2.destroyAllWindows()


def heatmap_viewer(name=None, version=None):
    if name == None:
        pics = np.load("history/pics.npy")
        heats = np.load("history/heats.npy")
        clicks = np.load("history/clicks.npy")
        heatmap = [(pic, heat, click) for pic, heat, click in zip(pics, heats, clicks)]
        show_heatmap_preview_gui(heatmap)
        return

    def safe_path(name):
        import re
        return re.sub(r'[\\/:*?"<>|]', "", name)

    dir = save_dir = f"dataset/{safe_path(name)} - {safe_path(version)}"

    pics = np.load(f"{dir}/pics.npy")
    heats = np.load(f"{dir}/heats.npy")
    clicks = np.load(f"{dir}/clicks.npy")
    heatmap = [(pic, heat, click) for pic, heat, click in zip(pics, heats, clicks)]
    show_heatmap_preview_gui(heatmap)


if __name__ == "__main__":
    main()
