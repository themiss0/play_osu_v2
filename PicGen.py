import pickle
import os
import cv2
from Camera import Camera
import setting
from MemReader import MemReader
import numpy as np




def safe_path(name):
    import re
    # 替换为下划线或直接去掉非法字符
    return re.sub(r'[\\/:*?"<>|]', "", name)




def main():
    camera = Camera()
    mem = MemReader()
    pics = []
    times = []
    meta = {}

    meta["cs"] = mem.cs
    meta["ar"] = mem.ar

    while True:
        mem.update()
        if mem.status == "play":
            print("Playing...")
            last = mem.time
            first = mem.start
            isStart = False
            while mem.time < mem.end:
                mem.update()
                if mem.time < first:
                    isStart = True
                if (
                    isStart
                    and mem.time >= first
                    and mem.time != last
                    and mem.time <= mem.end
                ):
                    grab = camera.get()
                    if grab is None:
                        continue
                    grab = cv2.cvtColor(grab, cv2.COLOR_BGR2GRAY)
                    grab = cv2.resize(
                        grab,
                        (setting.heatmap_size[0], setting.heatmap_size[1]),
                        interpolation=cv2.INTER_AREA,
                    )
                    pics.append(grab)
                    times.append(mem.time)
                    print(f"current frame: {len(pics)}, time: {mem.time}")
                last = mem.time
            else:
                print("result:" + mem.status)
            break
        else:
            print("waiting play")

    print("song over")
    save_dir = f"dataset/{safe_path(mem.song)} - {safe_path(mem.version)}"

    pics_array = np.stack(pics)
    times_array = np.stack(times)

    # 保存
    os.makedirs(save_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错

    np.save(
        f"{save_dir}/pics.npy",
        pics_array,
    )
    np.save(
        f"{save_dir}/times.npy",
        times_array,
    )
    pickle.dump(
        meta,
        open(
            f"{save_dir}/meta.pkl",
            "wb",
        ),
    )


    
    VERSION = 1
    version_path = save_dir + "/" + "pic_version.txt"
    if os.path.exists(version_path):
        with open(version_path, "r") as v:
            try:
                VERSION = int(v.read()) + 1
            except Exception:
                pass


    with open(version_path, "w") as v:
        v.write(str(VERSION))


if __name__ == "__main__":
    main()
