import MemReader
import cv2
import numpy as np
import os
from MemReader import MemReader

MODE = 0  # 0为当前游戏谱面，1为推理输出，2为指定谱面
show_data = True


def safe_path(name):
    import re

    return re.sub(r'[\\/:*?"<>|]', "", name)


def main():
    if MODE == 0:
        mem = MemReader()
        mem.update()
        replay_viewer(safe_path(mem.song), safe_path(mem.version))
        pass
    elif MODE == 1:
        replay_viewer()
    else:
        replay_viewer("Ai no Sukima", "hard")


def show_replay_preview_gui(replay_data, resize_factor=8):
    def nothing(x):
        pass

    # 窗口与滑条
    cv2.namedWindow("Replay Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Replay Preview", 800, 600)
    cv2.createTrackbar("Frame", "Replay Preview", 0, len(replay_data) - 1, nothing)
    cv2.createTrackbar("Cursor Size", "Replay Preview", 1, 10, nothing)
    cv2.createTrackbar("R", "Replay Preview", 0, 255, nothing)
    cv2.createTrackbar("G", "Replay Preview", 255, 255, nothing)
    cv2.createTrackbar("B", "Replay Preview", 0, 255, nothing)
    cv2.createTrackbar("FPS", "Replay Preview", 200, 500, nothing)

    paused = False
    current_frame = 0

    def update_display(frame_idx):
        img_gray, x_norm, y_norm, click, hold = replay_data[frame_idx]
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        h, w = img_gray.shape
        x = int(x_norm * w)
        y = int(y_norm * h)

        size = cv2.getTrackbarPos("Cursor Size", "Replay Preview")
        r = cv2.getTrackbarPos("R", "Replay Preview")
        g = cv2.getTrackbarPos("G", "Replay Preview")
        b = cv2.getTrackbarPos("B", "Replay Preview")

        # 画光标点
        cv2.circle(img_color, (x, y), size, (b, g, r), -1)

        display = cv2.resize(
            img_color,
            (w * resize_factor, h * resize_factor),
            interpolation=cv2.INTER_NEAREST,
        )

        if show_data:
            cv2.putText(
                display,
                f"click: {click}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                f"hold: {hold}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Replay Preview", display)

    last_frame_time = 0
    while True:
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        fps = cv2.getTrackbarPos("FPS", "Replay Preview")
        frame_time = 1.0 / fps

        if not paused:
            if current_time - last_frame_time >= frame_time:
                update_display(current_frame)
                last_frame_time = current_time
                current_frame = (current_frame + 1) % len(replay_data)
                cv2.setTrackbarPos("Frame", "Replay Preview", current_frame)
        else:
            update_display(current_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
        elif key == ord("r"):
            current_frame = 0
            cv2.setTrackbarPos("Frame", "Replay Preview", 0)
        elif key == ord("b"):
            current_frame = max(0, current_frame - 1)
            cv2.setTrackbarPos("Frame", "Replay Preview", current_frame)
            paused = True
        elif key == ord("f"):
            current_frame = min(len(replay_data) - 1, current_frame + 1)
            cv2.setTrackbarPos("Frame", "Replay Preview", current_frame)
            paused = True

        frame_pos = cv2.getTrackbarPos("Frame", "Replay Preview")
        if frame_pos != current_frame:
            current_frame = frame_pos
            paused = True

    cv2.destroyAllWindows()


def replay_viewer(name=None, version=None):
    if name is None:
        pics = np.load("history/pics.npy")
        replays = np.load("history/replays.npy")
    else:
        dir = f"dataset/{safe_path(name)} - {safe_path(version)}"
        pics = np.load(f"{dir}/pics.npy")
        replays = np.load(f"{dir}/replays.npy")

    # 组装数据
    data = [(pic, x, y, click, hold) for pic, (x, y, click, hold) in zip(pics, replays)]

    show_replay_preview_gui(data)


if __name__ == "__main__":
    main()
