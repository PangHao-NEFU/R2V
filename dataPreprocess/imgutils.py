import os
import string
from math import sqrt

import cv2
import numpy as np
import pytesseract
from PIL import Image
import shutil
from tqdm import tqdm
from dataprepare.fromSJJ.check import *


# 读取图像，解决imread不能读取中文路径的问题
def cv2_imread(file_path):
    """
    解决opencv读取文件路径不允许中文的问题
    :param file_path:
    :return:
    """
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def cv2_imwrite(file_path, imgfile):
    """
    解决opencv写文件路径不允许有中文的问题
    :param file_path:
    :param imgfile:
    :return:
    """
    ext = os.path.splitext(os.path.basename(file_path))[-1]
    cv2.imencode(f'{ext}', imgfile)[1].tofile(file_path)


def resizeImg(imgPath, width=512, height=512):
    img = cv2_imread(imgPath)
    width, height = img.shape
    width = min(width, height)
    height = min(width, height)
    resizedImg = cv2.resize(img, (width, height))
    fileDirPath = os.path.dirname(imgPath)
    fileName = os.path.basename(imgPath)
    resizedimgName = os.path.splitext(fileName)[0] + '-resized' + os.path.splitext(fileName)[1]
    cv2.imwrite(os.path.join(fileDirPath, resizedimgName), img)

    return


# 保持图片比例缩放图片
def crop_image(image_path, target_size: tuple):
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
    fileDirPath = os.path.dirname(image_path)
    fileName = os.path.basename(image_path)
    resizedimgName = os.path.splitext(fileName)[0] + '-croped' + os.path.splitext(fileName)[1]
    output_path = os.path.join(fileDirPath, resizedimgName)
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
    cropped_image = image[start_y:start_y + size, start_x:start_x + size]
    fileDirPath = os.path.dirname(image_path)
    fileName = os.path.basename(image_path)
    output_path = os.path.join(os.path.dirname(fileDirPath), 'croped', fileName)
    if not os.path.exists(os.path.join(os.path.dirname(fileDirPath), 'croped')):
        os.makedirs(os.path.join(os.path.dirname(fileDirPath), 'croped'))
    # print(output_path)
    cv2_imwrite(output_path, cropped_image)


def renameImgNameByJson(imagedir, labeldir):
    renameImgName(imagedir)
    imageFiles = os.listdir(imagedir)
    labelFiles = os.listdir(labeldir)
    for img in tqdm(imageFiles):
        for label in labelFiles:
            imgFileName = os.path.splitext(img)[0]
            imgExt = os.path.splitext(img)[-1]
            floorName = os.path.splitext(label)[0].split('@')[0]
            if imgFileName == floorName:
                floorId = os.path.splitext(label)[0].split('@')[-1]
                newImgPath = os.path.join(os.path.dirname(imagedir), 'result', f"{floorId}", floorId + imgExt)
                newLabelPath = os.path.join(os.path.dirname(imagedir), 'result', f"{floorId}", floorId + '.json')
                if not os.path.exists(os.path.join(os.path.dirname(imagedir), 'result', f'{floorId}')):
                    os.makedirs(os.path.join(os.path.dirname(imagedir), 'result', f'{floorId}'))
                shutil.copy2(os.path.join(imagedir, img), newImgPath)
                shutil.copy2(os.path.join(labeldir, label), newLabelPath)
                break


def batchCropImg(imgDir):
    imageFiles = os.listdir(imgDir)
    for img in tqdm(imageFiles):
        imgPath = os.path.join(imgDir, img)
        if os.path.isfile(imgPath) and (os.path.splitext(os.path.basename(imgPath))[-1] in ('.jpg', '.png')):
            make_square(imgPath)


def removeInvaild(errorfilePath, combinedDir):
    with open(errorfilePath, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip('\n')
            try:
                # 如果存在此文件夹,级联删除
                if os.path.exists(os.path.join(combinedDir, line)):
                    shutil.rmtree(os.path.join(combinedDir, line))
                    print(f"Removed successfully:{line}\n")
            except OSError as e:
                print(f"{line}文件夹删除失败!,错误原因:{e}\n")


current_folder = ""
output_file = ""


def write_folder_name():
    global current_folder, output_file

    if current_folder != "" and output_file != "":
        with open(output_file, 'a') as file:
            file.write(current_folder + "\n")
        print(f"errorid保存成功!{current_folder}\n")


def browse_images(folder_path, file_path):
    global current_folder, output_file

    output_file = file_path

    image_extensions = ['.jpg', '.jpeg', '.png']  # 支持的图片扩展名

    # 获取文件夹及其子文件夹中所有图片的路径
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            # _, ext = os.path.splitext(file)
            if os.path.exists(os.path.join(root, dir, 'WallPoint.jpg')):
                image_paths.append(os.path.join(root, dir, 'WallPoint.jpg'))
            else:
                # 不存在这张图片直接将此文件夹写入errorfile
                with open(file_path,'a') as f:
                    f.write(os.path.basename(dir)+'\n')
                    f.flush()


    if len(image_paths) == 0:
        print("文件夹中没有找到符合条件的图片")
        return

    current_index = 0
    total_images = len(image_paths)
    print(f"totalimgs:{total_images}")

    # 创建OpenCV窗口和按钮
    cv2.namedWindow("Image Viewer")

    while current_index<total_images:
        image_path = image_paths[current_index]
        current_folder = os.path.basename(os.path.dirname(image_path))

        # 读取图片并显示
        try:
            image = cv2_imread(image_path)
            cv2.putText(image,current_folder, (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Image Viewer", image)

            key = cv2.waitKey(0)

            # 按左箭头键切换到上一张图片
            if key == 81 or key == 113:  # ASCII码值：Q/q
                current_index = current_index - 1

            # 按右箭头键切换到下一张图片
            elif key == 83 or key == 115:  # ASCII码值：S/s
                current_index = current_index + 1
            elif key == 13:  # ASCII码值：Enter
                write_folder_name()
            # 按Esc键退出浏览
            elif key == 27:  # ASCII码值：Esc
                break
        except Exception as e:
            print("Error",e,'\n')

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # step 1: 图片中心裁剪:
    imgDirPath = r'C:\Users\Pang Hao\Desktop\sjjdataset2\img'
    # batchCropImg(imgDirPath)

    # step 2: 筛选合适的crop的图片后,将dir输入下方
    labelDirPath = r'C:\Users\Pang Hao\Desktop\sjjdataset2\json'
    renameImgNameByJson(os.path.join(os.path.dirname(imgDirPath),'croped'),labelDirPath)

    # step 3: 使用sjj_v2生成训练数据

    # step 6: 手工筛选数据
    # folder_path = "./sjj_v2/result"
    # output_file_path = "./history/errorfile.txt"
    # browse_images(folder_path, output_file_path)

    # step 4: 筛选出有效数据:
    # removeInvaild(r"./history/errorfile.txt", r"C:\Users\Pang Hao\Desktop\sjjdataset1\result")

    # step 5 : 重新使用sjj_v2生成训练数据
