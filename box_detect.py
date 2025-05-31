import cv2 as cv
from Camera import Camera
import numpy as np


# box_detect:
# 输出setting给出的分辨率下白框的位置

def detect_box(img: np.array):
    """识别并返回图像中的白色方框区域"""

    # 转换为灰度图

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 使用二值化方法，将白色区域变为白色，其他区域变为黑色
    _, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    # 寻找轮廓
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 筛选符合白色方框条件的轮廓
    boxes = []
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv.boundingRect(contour)
        # 设定长宽比限制，来判断是否为方框
        aspect_ratio = w / h
        if 1.2 <= aspect_ratio <= 1.3 and w > 300:  # 方框的长宽比应该接近1
            boxes.append((x, y, w, h))
    # 返回最大的方框
    if len(boxes) > 0:
        return max(boxes, key=lambda box: box[2] * box[3])  # 返回最大的方框
    return None


camera = Camera()
while True:
    box = detect_box(camera)
    if box is not None:
        print(box)
