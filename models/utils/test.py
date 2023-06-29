import string

import cv2
import numpy as np
import pytesseract
from PIL import Image

def _calc_factor( img ):
    # if self.underlay_item is None:
    #     return None
    # 预处理
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 配置tesseract
    pytesseract.pytesseract.tesseract_cmd = f'D:\\Program\\Tesseract-OCR\\tesseract.exe'
    # 识别数字
    digits = pytesseract.image_to_string(blurred, config='--psm 6', lang='num')
    # 使用霍夫变换检测直线,窗口大小3
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # 计算直线长度
    line_lengths = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        line_lengths.append(length)

    width = 10  # self.underlay_item["width"]
    height = 15.123  # self.underlay_item["height"]
    digits=[string.atoi(i) for i in digits]
    maxNum=np.max(digits)
    
    print(line_lengths, digits)

    # ratio = height / (floor_plan_img_height)
    # return abs(width), abs(height), abs(1.0 / ratio)
    return


if __name__ == "__main__":
    imgPath=r"./fxy.png"
    readimg = cv2.imread(imgPath)
    # Image.open(imgPath)
    _calc_factor(readimg)
