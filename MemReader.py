from OsuCoordMapper import OsuCoordMapper
import json
import requests
import os
import time
import setting


class MemReader:

    def __init__(self):
        self.session = requests.Session()
        self.data = {}  # WebSocket 原始数据
        self.perfect = 0
        self.great = 0
        self.ok = 0
        self.loss = 0
        self.time = 0
        self.cs = 0
        self.ar = 0
        self.start = 0
        self.end = 0
        self.websocket = None  # WebSocket 连接对象
        self.status = ""
        self.song = ""
        self.version = ""
        self.update()

    def get_map(self):
        try:
            js = self.get_json()
            file_path = js["folders"]["songs"] + "\\" + js["directPath"]["beatmapFile"]
            map_data = []
            with open(file_path, "r", encoding="utf-8") as file:
                isHitObj = False
                mapper = OsuCoordMapper(
                    [
                        setting.play_filed[2],
                        setting.play_filed[3],
                    ],
                    self.cs,
                )

                for line in file:
                    line = line.strip()
                    if isHitObj != True:
                        if line == "[HitObjects]":
                            isHitObj = True
                            continue
                    else:
                        obj = line.split(",")
                        x, y = mapper.game_to_normalized(int(obj[0]), int(obj[1]))
                        time = int(obj[2])
                        type = int(obj[3])
                        if type & (0b00000001) == 1:
                            type = 0
                        elif type & (0b00000010) == 2:
                            type = 1
                        elif type & (0b0001000) == 8:
                            type = 2
                        map_data.append([x, y, time, type])
            return map_data
        except KeyError as e:
            print(f"KeyError encountered while updating: {e}")
            return None

    def get_last_note_score(self):
        try:
            js = self.data["play"]["hits"]
            if self.perfect != js["300"]:
                return [100, self.data["play"]["hitErrorArray"][-1]]
            if self.great != js["100"]:
                return [75, self.data["play"]["hitErrorArray"][-1]]
            if self.ok != js["50"]:
                return [50, self.data["play"]["hitErrorArray"][-1]]
            if self.loss != js["0"]:
                return [0, self.data["play"]["hitErrorArray"][-1]]
        except KeyError:
            return None

    def update(self):
        try:

            js = self.get_json()
            self.perfect = js["play"]["hits"]["300"]
            self.great = js["play"]["hits"]["100"]
            self.ok = js["play"]["hits"]["50"]
            self.loss = js["play"]["hits"]["0"]
            self.cs = js["beatmap"]["stats"]["cs"]["converted"]
            self.ar = js["beatmap"]["stats"]["ar"]["converted"]
            self.time = js["beatmap"]["time"]["live"]
            self.start = js["beatmap"]["time"]["firstObject"]
            self.end = js["beatmap"]["time"]["lastObject"]
            self.status = js["state"]["name"]
            self.song = js["beatmap"]["titleUnicode"]
            self.version = js["beatmap"]["version"]

        except KeyError as e:
            print(f"KeyError encountered while updating: {e}")
        pass

    def get_json(self):
        self.data = self.session.get(setting.url).json()
        return self.data
