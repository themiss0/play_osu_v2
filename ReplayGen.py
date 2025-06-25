import os
import setting
import numpy as np
from OsuCoordMapper import OsuCoordMapper
from osrparse import Replay
import pickle
from tqdm import tqdm


def heat_map_generator():
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
        replays = []
        mapper = OsuCoordMapper(setting.heatmap_size, cs)
        preempt = mapper.get_preempt_time(ar)

        note_radius_normal = radius / (512 + 2 * radius), radius / (384 + 2 * radius)
        click_frame_cnt = 0

        click_pro = []
        hold_pro = []

        # 生成hold标签
        for i in range(len(replay.replay_data)):
            key = replay.replay_data[i].keys
            if key > 0:
                if click_frame_cnt < 3:
                    click_pro.append(1)
                    hold_pro.append(0)
                else:
                    click_pro.append(0)
                    hold_pro.append(1)
                    # 滑条起始阶段也补充hold
                    if click_frame_cnt == 3:
                        p = i - 1
                        while p >= 0 and click_pro[p] == 1 and hold_pro[p] == 0:
                            hold_pro[p] = 1
                            p -= 1
                click_frame_cnt += 1
            else:
                click_pro.append(0)
                hold_pro.append(0)
                click_frame_cnt = 0

        p = 0
        p_time = replay.replay_data[0].time_delta

        for frame_time in tqdm(times, "Processing Replays"):
            # 向前找到小于等于 frame_time 的时间戳
            while p + 1 < len(replay.replay_data) and p_time < frame_time:
                p += 1
                p_time += replay.replay_data[p].time_delta

            f = replay.replay_data[p]
            lf = replay.replay_data[p - 1]
            lf_time = p_time - f.time_delta
            f_time = p_time
            delta = f_time - lf_time

            if delta != 0:
                offset = (frame_time - lf_time) / delta
            else:
                offset = 0  # 防止除零（瞬移情况）

            x = int(lf.x + (f.x - lf.x) * offset)
            y = int(lf.y + (f.y - lf.y) * offset)
            replays.append(
                [
                    *mapper.game_to_normalized(x, y),
                    click_pro[p],
                    hold_pro[p],
                ]
            )

        replays = np.stack(replays)
        np.save(dir + "/" + "replays.npy", replays)


if __name__ == "__main__":
    heat_map_generator()
