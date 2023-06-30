import os
import string
from math import sqrt

import cv2
import numpy as np
import pytesseract
from PIL import Image
import shutil
from tqdm import tqdm

## 读取图像，解决imread不能读取中文路径的问题
def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def cv2_imwrite(file_path,imgfile):
    ext=os.path.splitext(os.path.basename(file_path))[-1]
    cv2.imencode(f'{ext}', imgfile)[1].tofile(file_path)

def resizeImg(imgPath, width=512, height=512):
    img =cv2_imread(imgPath)
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
    image = cv2_imread(image_path)

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
    output_path=os.path.join(os.path.dirname(fileDirPath),'croped',fileName)
    if not os.path.exists(os.path.join(os.path.dirname(fileDirPath),'croped')):
        os.makedirs(os.path.join(os.path.dirname(fileDirPath),'croped'))
    # print(output_path)

    cv2_imwrite(output_path,cropped_image)

def renameImgNameByJson(imagedir,labeldir):
    imageFiles=os.listdir(imagedir)
    labelFiles=os.listdir(labeldir)
    for img in tqdm(imageFiles):
        for label in labelFiles:
            imgFileName=os.path.splitext(img)[0]
            imgExt=os.path.splitext(img)[-1]
            floorName=os.path.splitext(label)[0].split('@')[0]
            if imgFileName == floorName:
                floorId=os.path.splitext(label)[0].split('@')[-1]
                newImgPath=os.path.join(os.path.dirname(imagedir),'result',f"{floorId}",floorId+imgExt)
                newLabelPath=os.path.join(os.path.dirname(imagedir),'result',f"{floorId}",floorId+'.json')
                if not os.path.exists(os.path.join(os.path.dirname(imagedir),'result',f'{floorId}')):
                    os.makedirs(os.path.join(os.path.dirname(imagedir),'result',f'{floorId}'))
                shutil.copy2(os.path.join(imagedir,img),newImgPath)
                shutil.copy2(os.path.join(labeldir,label),newLabelPath)
                break

def batchCropImg(imgDir):
    imageFiles=os.listdir(imgDir)
    for img in tqdm(imageFiles):
        imgPath=os.path.join(imgDir,img)
        if os.path.isfile(imgPath) and (os.path.splitext(os.path.basename(imgPath))[-1] in ('.jpg','.png')):
            make_square(imgPath)


if __name__ == "__main__":
    # step 1:图片中心裁剪:
    imgDirPath=r''
    batchCropImg(imgDirPath)
    # step 2:筛选合适的crop的图片后,将dir输入下方
    labelDirPath=r''

