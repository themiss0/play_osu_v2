import os
import setting
import numpy as np
from OsuCoordMapper import OsuCoordMapper
from osrparse import Replay
import pickle


def heat_map_generator():
    heatmap_width = setting.heatmap_size[0]
    heatmap_height = setting.heatmap_size[1]
    for dir in os.listdir("dataset"):
        dir = "dataset/" + dir
        print("now:" + dir)
        try:
            if not os.path.isdir(dir):
                continue

            with open(dir + "/" + "rep.osr", "rb") as f:
                replay = Replay.from_file(f)

            with open(dir + "/" + "meta.pkl", "rb") as meta:
                meta = pickle.load(meta)
                cs = meta["cs"]
                ar = meta["ar"]

            with open(dir + "/" + "times.npy", "rb") as t:
                times = np.load(t)
                times = [times[i] for i in range(len(times))]
        except Exception as e:
            print("error: " + str(e) + "on song: " + dir)
            continue

        radius = 54.4 - 4.48 * cs
        heatmaps = []
        clicks = []
        mapper = OsuCoordMapper(setting.heatmap_size, cs)
        preempt = mapper.get_preempt_time(ar)

        note_radius_normal = radius / (512 + 2 * radius), radius / (384 + 2 * radius)

        ptr = 0
        ptr_time = replay.replay_data[0].time_delta
        for frame_time in times:
            click = 0

            while ptr_time + 40 < frame_time:
                ptr += 1
                ptr_time += replay.replay_data[ptr].time_delta

            cursor_time = ptr_time

            p = ptr
            while p < len(replay.replay_data) and frame_time + 40 > cursor_time:
                f = replay.replay_data[p]

                is_click = f.keys > 0

                if is_click:
                    o = frame_time - cursor_time
                    click = max(o / 40, click)
                cursor_time += f.time_delta
                p += 1

            clicks.append(click)

        ptr = 0
        ptr_time = replay.replay_data[0].time_delta
        for frame_time in times:
            heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)

            while ptr_time + 20 < frame_time:
                ptr += 1
                ptr_time += replay.replay_data[ptr].time_delta

            cursor_time = ptr_time
            p = ptr
            while (
                p < len(replay.replay_data) - 1 and frame_time + preempt > cursor_time
            ):
                f = replay.replay_data[p]
                nf = replay.replay_data[p + 1]

                if f.keys == 0:
                    p += 1
                    cursor_time += nf.time_delta
                    continue

                offset = (float(frame_time - cursor_time)) / (
                    cursor_time + nf.time_delta
                )

                x = int(f.x + (nf.x - f.x) * offset)
                y = int(f.y + (nf.y - f.y) * offset)

                nx, ny = mapper.game_to_image(x, y)

                rw = int(note_radius_normal[0] * heatmap_width)
                rh = int(note_radius_normal[1] * heatmap_height)

                # 高斯分布范围
                xmin = max(nx - rw, 0)
                xmax = min(nx + rw, heatmap_width)
                ymin = max(ny - rh, 0)
                ymax = min(ny + rh, heatmap_height)

                weight = (preempt - abs(frame_time - cursor_time)) / preempt
                for i in range(xmin, xmax):
                    for j in range(ymin, ymax):
                        dx = (i - nx) / rw
                        dy = (j - ny) / rh
                        g = np.exp(-(dx**2 + dy**2) * 4)

                        heatmap[j, i] = max(heatmap[j, i], g * weight)
                p += 1
                cursor_time += nf.time_delta
            heatmaps.append(heatmap)

        for i in range(len(clicks) - 1):
            if clicks[i] > 0.5:
                if clicks[i + 1] > 0.5:
                    clicks[i + 1] = 1
                elif clicks[i + 1] == 0:
                    clicks[i] = 1

        clicks = np.stack(clicks)
        heatmaps = np.stack(heatmaps)
        np.save(dir + "/" + "clicks.npy", clicks)
        np.save(dir + "/" + "heats.npy", heatmaps)

if __name__ == "__main__":
    heat_map_generator()
