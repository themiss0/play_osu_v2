class OsuCoordMapper:
    def __init__(self, image_size: tuple[int, int], cs: float):
        self.image_width, self.image_height = image_size
        self.cs = cs
        self.r = 54.4 - 4.48 * cs

        # 游戏区域偏移范围
        self.game_origin_x = -self.r
        self.game_origin_y = -self.r

        # 映射范围为：[-r, 512 + r] * [-r, 384 + r]
        self.game_width = 512 + 2 * self.r
        self.game_height = 384 + 2 * self.r

    def normalized_to_game(self, x: float, y: float) -> tuple[float, float]:
        gx = x * self.game_width + self.game_origin_x
        gy = y * self.game_height + self.game_origin_y
        return gx, gy

    def game_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        nx = (x - self.game_origin_x) / self.game_width
        ny = (y - self.game_origin_y) / self.game_height
        return nx, ny

    def game_to_image(self, x: float, y: float) -> tuple[int, int]:
        nx, ny = self.game_to_normalized(x, y)
        img_x = int(nx * self.image_width)
        img_y = int(ny * self.image_height)
        return img_x, img_y

    def image_to_game(self, x: int, y: int) -> tuple[float, float]:
        nx = x / self.image_width
        ny = y / self.image_height
        return self.normalized_to_game(nx, ny)

    def normalized_to_image(self, x: float, y: float) -> tuple[int, int]:
        gx, gy = self.normalized_to_game(x, y)
        return self.game_to_image(gx, gy)

    def image_to_normalized(self, x: int, y: int) -> tuple[float, float]:
        gx, gy = self.image_to_game(x, y)
        return self.game_to_normalized(gx, gy)

    def get_preempt_time(self, ar: float) -> float:
        if ar < 5:
            return 1200 + 600 * (5 - ar) / 5
        else:
            return 1200 - 750 * (ar - 5) / 5

    def get_fade_in_time(self, ar: float) -> float:
        if ar < 5:
            return 800 + 400 * (5 - ar) / 5
        else:
            return 800 - 500 * (ar - 5) / 5
