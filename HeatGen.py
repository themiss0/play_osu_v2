import os
import setting
import numpy as np
from OsuCoordMapper import OsuCoordMapper
from osrparse import Replay
import pickle

# 每次更新heatgen的代码时，应该修改为不同的值
VERSION = 1


def heat_map_generator():
    heatmap_width = setting.heatmap_size[0]
    heatmap_height = setting.heatmap_size[1]
    for dir in os.listdir("dataset"):
        dir = "dataset/" + dir
        print("now:" + dir)
        try:
            if not os.path.isdir(dir):
                continue

            version_path = dir + "/" + "version.txt"

            need_update = True
            pic_version = 1
            # 读图像版本
            with open(dir + "/" + "pic_version.txt", "r") as v:
                try:
                    pic_version = int(v.read())
                except Exception:
                    continue
            if os.path.exists(version_path):
                with open(version_path, "r") as v:
                    try:
                        lines = v.readlines()
                        file_version = int(lines[0].strip())
                        my_pic_version = int(lines[1].strip())
                        if VERSION == file_version and my_pic_version == pic_version:
                            need_update = False
                    except Exception:
                        pass
            if not need_update:
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

        p = 0
        ptr_time = replay.replay_data[0].time_delta

        # click数据集
        for frame_time in times:
            click = 0

            while ptr_time + 10 < frame_time:
                p += 1
                ptr_time += replay.replay_data[p].time_delta

            while p < len(replay.replay_data) and frame_time + 10 > ptr_time:
                f = replay.replay_data[p]

                is_click = f.keys > 0

                if is_click:
                    click = 1
                    
                ptr_time += f.time_delta
                p += 1

            clicks.append(click)

        p = 0
        ptr_time = replay.replay_data[0].time_delta

        # heat数据集
        for frame_time in times:
            heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)

            while ptr_time + 20 < frame_time:
                ptr_time += replay.replay_data[p].time_delta

            while p < len(replay.replay_data) - 1 and frame_time + preempt > ptr_time:
                f = replay.replay_data[p]
                nf = replay.replay_data[p + 1]

                if f.keys == 0:
                    p += 1
                    ptr_time += nf.time_delta
                    continue

                offset = (float(frame_time - ptr_time)) / (ptr_time + nf.time_delta)

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

                weight = (preempt - abs(frame_time - ptr_time)) / preempt
                for i in range(xmin, xmax):
                    for j in range(ymin, ymax):
                        dx = (i - nx) / rw
                        dy = (j - ny) / rh
                        g = np.exp(-(dx**2 + dy**2) * 4)

                        heatmap[j, i] = max(heatmap[j, i], g * weight)
                p += 1
                ptr_time += nf.time_delta
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

        with open(version_path, "w") as v:
            v.writelines([str(VERSION) + "\n", str(pic_version) + "\n"])


if __name__ == "__main__":
    heat_map_generator()
