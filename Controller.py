import pyautogui as ag
import setting
import pygetwindow as gw
import keyboard


class Controller:
    def __init__(self, window: gw.Win32Window):
        self.window = window
        self.play_field = setting.play_filed  # 游戏区域在窗口内的偏移: [x1, y1, x2, y2]

    def click(self):
        ag.leftClick(_pause=False)
        # ag.keyDown(setting.left_click, _pause=False)
        # ag.keyUp(setting.left_click, _pause=False)

    def hold(self):
        keyboard.press(setting.right_click)

    def unhold(self):
        keyboard.release(setting.right_click)

    def move_offset(self, offset):
        ag.move(offset[0], offset[1], _pause=False)

    def move_to_game_pos(self, loc_normalized):
        """
        将归一化坐标 (0~1) 转换为屏幕上的实际像素位置，并移动光标
        """
        x, y, w, h = self.play_field

        win_left = self.window.left
        win_top = self.window.top

        game_x = int(win_left + x + loc_normalized[0] * w)
        game_y = int(win_top + y + loc_normalized[1] * h)

        ag.moveTo(game_x, game_y, _pause=False)

    def now_in_game(self):
        """
        获取当前鼠标在游戏区域内的归一化位置 [0-1]
        注意：如果光标在游戏区域外，结果可能超出0-1范围
        """
        x1, y1, x2, y2 = self.play_field
        width = x2 - x1
        height = y2 - y1

        win_left = self.window.left
        win_top = self.window.top

        mouse_x, mouse_y = ag.position()

        rel_x = mouse_x - win_left - x1
        rel_y = mouse_y - win_top - y1

        norm_x = rel_x / width
        norm_y = rel_y / height

        return [norm_x, norm_y]
