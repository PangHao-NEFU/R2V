import os
import string
from math import sqrt

import cv2
import numpy as np
import pytesseract
from PIL import Image

def resizeImg(imgPath, width=512, height=512):
    img = cv2.imread(imgPath)
    width,height=img.shape
    width=min(width,height)
    height=min(width,height)
    resizedImg=cv2.resize(img,(width,height))
    fileDirPath=os.path.dirname(imgPath)
    fileName=os.path.basename(imgPath)
    resizedimgName=os.path.splitext(fileName)[0]+'-resized'+os.path.splitext(fileName)[1]
    cv2.imwrite(os.path.join(fileDirPath,resizedimgName),img)

    return

# 保持图片比例缩放图片
def crop_image(image_path,target_size:tuple):
    # 打开图片
    image = Image.open(image_path)

    # 获取图片的宽度和高度
    width, height = image.size

    # 计算宽高比例
    aspect_ratio = width / height

    # 目标宽度和高度
    target_width, target_height = target_size

    # 根据宽高比例调整目标高度
    if aspect_ratio > 1:
        target_height = int(target_width / aspect_ratio)
    else:
        target_width = int(target_height * aspect_ratio)

    # 调整图片大小
    resized_image = image.resize((target_width, target_height))

    # 计算中心剪裁的左上角坐标
    left = (target_width - target_size[0]) // 2
    top = (target_height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]

    # 中心剪裁图片
    cropped_image = resized_image.crop((left, top, right, bottom))
    fileDirPath=os.path.dirname(image_path)
    fileName=os.path.basename(image_path)
    resizedimgName=os.path.splitext(fileName)[0]+'-croped'+os.path.splitext(fileName)[1]
    output_path=os.path.join(fileDirPath,resizedimgName)
    # 保存图片
    cropped_image.save(output_path)

# 正方形图片,以短边为准,超出的地方不要了
def make_square(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 获取图像的宽度和高度
    height, width = image.shape[:2]

    # 计算裁剪后的边长（以短边为准）
    size = min(width, height)

    # 计算裁剪的起始坐标
    start_x = (width - size) // 2
    start_y = (height - size) // 2

    # 进行裁剪
    cropped_image = image[start_y:start_y+size, start_x:start_x+size]
    fileDirPath=os.path.dirname(image_path)
    fileName=os.path.basename(image_path)
    resizedimgName=os.path.splitext(fileName)[0]+'-squared'+os.path.splitext(fileName)[1]
    output_path=os.path.join(fileDirPath,resizedimgName)
    print(output_path)
    # 保存图片
    # cropped_image.save(output_path)
    cv2.imwrite(output_path,cropped_image)

if __name__ == "__main__":
    # imgPath=r"./sjj_v2/data/agl/agl.png"
    # readimg = cv2.imread(imgPath)
    #
    #
    # # Image.open(imgPath)
    # print(_calc_factor(readimg))
    # resizeImg('./sjj_v2/data/agl/agl.png')

    # crop_image('./sjj_v2/data/bly/bly.png',(2970,2970))   #这种效果不好
    make_square('./sjj_v2/data/bly/bly.png')

