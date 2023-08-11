import cv2
import numpy as np
from PIL import Image

critical_dim_min = 200
critical_dim_max = 50000


def dim_rec_num(ocr_result_bboxs):
    num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # 过滤非数字字符
    ocr_rect_num = {}
    for key in range(len(ocr_result_bboxs)):
        rect_list = ocr_result_bboxs[key][0]
        character_list = ocr_result_bboxs[key][1]
        characters = []
        for char in character_list:
            if '.' == char or ' ' == char:
                break
            if (0 != len(characters) and ('o' == char or 'O' == char)):
                characters.append('0')
            if char not in num_list:
                continue
            characters.append(char)
        char_str = "".join(characters)
        if len(char_str) < 1:
            continue
        dim_num = int(char_str)
        if (dim_num > critical_dim_max) or (dim_num <= critical_dim_min):
            continue
        else:
            # 定义一个list，结构为[x1,y1,x2,y2,x3,y3,x4,y4,dim_num]
            point_list = []
            for item in ocr_result_bboxs[key][0]:
                point_list.extend([item[0], item[1]])
            a = {key: [point_list, dim_num]}
            ocr_rect_num.update(a)
    return ocr_rect_num


def get_up_down_dim(ocr_rect_num):
    # 根据rect位置筛选出最上和最下两组标注信息
    # 求出来一副图总最上和最下的两个标记点（y坐标 ）
    up_err_flag = False
    down_err_flag = False
    # 两组尺寸标注结构
    dim_up = {}
    dim_down = {}
    tl_br_list = []
    if len(ocr_rect_num) > 0:
        for key in ocr_rect_num:
            rect = ocr_rect_num[key][0]
            tl_br_list.append(rect)
        sorted(tl_br_list, key=(lambda x: x[7]))
        br_y_max = tl_br_list[-2][7]
        br_y_second_max = tl_br_list[-1][7]
        br_y_min = tl_br_list[0][7]
        br_y_second_min = tl_br_list[1][7]
        if br_y_max - br_y_second_max > 5:
            br_y_max = br_y_second_max
        if br_y_second_min - br_y_min > 5:
            br_y_min = br_y_second_min
        for key in ocr_rect_num:
            if abs(ocr_rect_num[key][0][7] - br_y_min) < 100:
                if (ocr_rect_num[key][1] > critical_dim_max or ocr_rect_num[key][1] < critical_dim_min):
                    continue
                dim_up[key] = [ocr_rect_num[key][0], ocr_rect_num[key][1]]
            elif abs(ocr_rect_num[key][0][7] - br_y_max) < 100:
                if (ocr_rect_num[key][1] > critical_dim_max or ocr_rect_num[key][1] < critical_dim_min):
                    continue
                dim_down[key] = [ocr_rect_num[key][0], ocr_rect_num[key][1]]
            else:
                continue
    return dim_up, dim_down


def get_up_down_sort_list(dim_up, dim_down):
    dim_up_lists = []
    dim_down_lists = []
    for key in dim_up:
        dim_up_list = []
        for num in dim_up[key][0]:
            dim_up_list.append(num)
        dim_up_list.append(dim_up[key][1])
        dim_up_lists.append(dim_up_list)
    for key in dim_down:
        dim_down_list = []
        for num in dim_down[key][0]:
            dim_down_list.append(num)
        dim_down_list.append(dim_down[key][1])
        dim_down_lists.append(dim_down_list)
    dim_up_lists_sorted = sorted(dim_up_lists, key=lambda x: x[8], reverse=True)
    dim_down_lists_sorted = sorted(dim_down_lists, key=lambda x: x[8], reverse=True)
    return dim_up_lists_sorted, dim_down_lists_sorted


def get_up_down_center_point(dim_up_lists_sorted, dim_down_lists_sorted):
    # 计算真实的一个像素代表的尺寸信息,core_up存储每个标注框的中心点的x坐标和对应的真实标注长度：格式为-坐标，长度，坐标，长度。。。。。。
    core_up_x_list = []
    core_up_y_list = []
    core_down_x_list = []
    core_down_y_list = []
    core_up_list = []
    core_down_list = []
    for idx in range(len(dim_up_lists_sorted)):
        up_tmp_list = []
        dim_up_list = dim_up_lists_sorted[idx]
        if len(dim_up_lists_sorted) < 1:
            break
        else:
            core_up_x = 0.5 * (dim_up_list[0] + dim_up_list[2])
            core_up_y = 0.5 * (dim_up_list[1] + dim_up_list[5])
            core_up_x_list.append(int(core_up_x))
            core_up_y_list.append(int(core_up_y))
            up_tmp_list.append(int(core_up_x))
            up_tmp_list.append(int(core_up_y))
            up_tmp_list.append(dim_up_list[8])
            core_up_list.append(up_tmp_list)

    for idx in range(len(dim_down_lists_sorted)):
        down_tmp_list = []
        dim_down_list = dim_down_lists_sorted[idx]
        if len(dim_down_lists_sorted) < 2:
            break
        else:
            core_down_x = 0.5 * (dim_down_list[0] + dim_down_list[2])
            core_down_y = 0.5 * (dim_down_list[1] + dim_down_list[5])
            core_down_x_list.append(core_down_x)
            core_down_y_list.append(core_down_y)
            down_tmp_list.append(int(core_down_x))
            down_tmp_list.append(int(core_down_y))
            down_tmp_list.append(dim_down_list[8])
            core_down_list.append(down_tmp_list)
    return core_up_list, core_down_list, core_up_x_list, core_up_y_list, core_down_x_list, core_down_y_list


def get_cropped_img(img_path, core_up_list, core_down_list):
    # originalImg = cv2.imread(img_path)
    cvtOriginalImg = cv2.cvtColor(np.asarray(img_path), cv2.COLOR_RGB2BGR)
    # cvtOriginalImg = cv2.cvtColor(originalImg, cv2.COLOR_RGBA2BGR)
    height, width = cvtOriginalImg.shape[:2]
    # print("height, width",height, width)
    crop_height = int(height / 18)
    if len(core_up_list) > 0:
        y_min_up = (core_up_list[0][1] - crop_height) if (core_up_list[0][1] - crop_height) > 0 else 0
        up_cropped_img = cvtOriginalImg[y_min_up:core_up_list[0][1] + crop_height, :]
        return up_cropped_img, y_min_up, 0
    elif len(core_down_list) > 0:
        y_max_down = (core_down_list[0][1] + crop_height) if (core_down_list[0][1] + crop_height) < height else height
        dwon_cropped_img = cvtOriginalImg[core_down_list[0][1] - crop_height:y_max_down, :]
        return dwon_cropped_img, y_max_down, 1


def get_boundaryDirection_yValue_ocrPoints(img_path, ocr_result_bboxs):
    ocr_rect_num = dim_rec_num(ocr_result_bboxs)
    dim_up, dim_down = get_up_down_dim(ocr_rect_num)
    dim_up_lists_sorted, dim_down_lists_sorted = get_up_down_sort_list(dim_up, dim_down)
    core_up_list, core_down_list, core_up_x_list, core_up_y_list, core_down_x_list, core_down_y_list = get_up_down_center_point(
        dim_up_lists_sorted, dim_down_lists_sorted)
    cropped_img, y_value, boundary_idx = get_cropped_img(img_path, core_up_list, core_down_list)
    # cv2.imwrite("check_result/[boundary]"+img.split('.')[0]+"_boundary_"+str(boundary_idx)+".jpg", cropped_img)
    ocr_point_list = []
    if boundary_idx == 0:
        ocr_point_list = core_up_list
    elif boundary_idx == 2:
        ocr_point_list = core_down_list
    # print("boundary:",boundary_idx," y_value:",y_value, "ocr_point_list:",ocr_point_list)
    return cropped_img, boundary_idx, y_value, ocr_point_list


def get_corner_points(image, bboxs):
    corner_point_map = {}
    tmp_corner_y = []
    tmp_corner_x = []
    corner_point_list = []
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for bound in bboxs:
        x1, y1, x2, y2, conf = bound
        crop_img = image.crop((x1, y1, x2, y2))
        crop_img_array = cv2.cvtColor(np.asarray(crop_img), cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(crop_img_array, 10, qualityLevel=0.6, minDistance=3, useHarrisDetector=False)
        if corners is not None and len(corners) > 0:
            for pt in corners:
                x = np.int32(pt[0][0]) + x1
                y = np.int32(pt[0][1]) + y1
                tmp_corner_y.append(int(y))
                corner_point_list.append([int(x), int(y)])
    center_y = int((max(tmp_corner_y) + min(tmp_corner_y)) / 2)
    max_y = max(tmp_corner_y)
    min_y = min(tmp_corner_y)
    tmp_up = []
    tmp_dwon = []
    for y in tmp_corner_y:
        if y > center_y:
            tmp_up.append(y)
        else:
            tmp_dwon.append(y)
    up_center_y = center_y
    down_center_y = center_y
    if len(tmp_up) > 0:
        up_center_y = int((max(tmp_up) + min(tmp_up)) / 2)
    if len(tmp_dwon) > 0:
        down_center_y = int((max(tmp_dwon) + min(tmp_dwon)) / 2)

    for item in corner_point_list:
        tmp_corner_x.append(int(item[0]))
        if np.abs(max_y - min_y) > 10:
            if item[1] > center_y:
                tmp = up_center_y
            else:
                tmp = down_center_y
            corner_point_map[item[0]] = tmp
        else:
            corner_point_map[item[0]] = center_y
    return corner_point_map


def closest(mylist, number):
    answer = []
    for i in mylist:
        answer.append(abs(number - i))
    index = answer.index(min(answer))
    if index > 0 and mylist[index] > number > mylist[index - 1]:
        index = index - 1
    return index


def get_ocr_ratio(boundary_idx, ocr_point_list, corner_point_map, y_value):
    print("boundary_idx:", boundary_idx, " ocr_points:", ocr_point_list, " corner_points:", corner_point_map," y_value:",y_value)
    ocr_corner_list = []
    corner_x_list = sorted(list(corner_point_map.keys()))
    corner_y_list = list(set(corner_point_map.values()))
    for ocr_point in ocr_point_list:
        ocr_x = ocr_point[0]
        ocr_y = ocr_point[1]
        ocr_value = ocr_point[2]
        for corner_x in corner_x_list:
            corner_y = corner_point_map[corner_x] + y_value
            closest_y_index = closest(corner_y_list, ocr_y - y_value)
            y_tmp = corner_y_list[closest_y_index] + y_value
            if np.abs(y_tmp - corner_y) < 10:
                closest_left_index = closest(corner_x_list, ocr_x)
                if closest_left_index + 1 < len(corner_x_list) and corner_x_list[closest_left_index] <= corner_x <= corner_x_list[closest_left_index + 1]:
                    if np.abs(corner_x_list[closest_left_index] - corner_x_list[closest_left_index + 1]) < 5 or np.abs(
                            corner_x_list[closest_left_index] - ocr_x) < 5 or np.abs(
                            corner_x_list[closest_left_index + 1] - ocr_x) < 5:
                        continue
                    else:
                        ocr_corner_list.append(
                            [ocr_x, corner_x_list[closest_left_index], corner_x_list[closest_left_index + 1], corner_y,
                             ocr_value])
                        break
    # print("ocr_corner_list:", ocr_corner_list)
    up_ratios = []
    for item in ocr_corner_list:
        ocr_x = item[0]
        corner_start_x = item[1]
        corner_end_x = item[2]
        corner_y = item[3]
        ocr_value = item[4]
        ratio = ocr_value / (corner_end_x - corner_start_x) / 10
        item.append(ratio)
        up_ratios.append(item)
    # print("up_ratios:", up_ratios)
    return up_ratios[0]
