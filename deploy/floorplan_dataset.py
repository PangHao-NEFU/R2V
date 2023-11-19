import copy

from torch.utils.data import Dataset

import time

from utils import *
from skimage import measure
import cv2
import os


def calcLineDirection(line, threshold=5):
    # 0 水平；1 垂直; -1 斜墙
    if np.abs(line[0][0] - line[1][0]) > threshold and np.abs(line[0][1] - line[1][1]) > threshold:
        return -1
    else:
        return int(abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]))


def lineRange(line):
    # direction: 0 水平；1 垂直； -1 斜墙
    direction = calcLineDirection(line)
    if direction == -1:
        fixedValue = 0
        minValue = 0
        maxValue = 0
    else:
        # 修正水平｜竖直的Wall Point
        fixedValue = (line[0][1 - direction] + line[1][1 - direction]) // 2
        minValue = min(line[0][direction], line[1][direction])
        maxValue = max(line[0][direction], line[1][direction])
    return direction, fixedValue, minValue, maxValue


def pointDistance(point_1, point_2):
    return max(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))


def divideWalls(walls):
    horizontalWalls = []
    verticalWalls = []
    for wall in walls:
        if calcLineDirection(wall) == 0:
            horizontalWalls.append(wall)
        else:
            verticalWalls.append(wall)
            pass
        continue
    return horizontalWalls, verticalWalls


def mergeLines(line_1, line_2):
    direction_1, fixedValue_1, min_1, max_1 = lineRange(line_1)
    direction_2, fixedValue_2, min_2, max_2 = lineRange(line_2)
    fixedValue = (fixedValue_1 + fixedValue_2) // 2
    if direction_1 == 0:
        return [(min(min_1, min_2), fixedValue), (max(max_1, max_2), fixedValue)]
    else:
        return [(fixedValue, min(min_1, min_2)), (fixedValue, max(max_1, max_2))]
    return


def findConnections(line_1, line_2, gap):
    connection_1 = -1
    connection_2 = -1
    pointConnected = False
    direction_line1 = calcLineDirection(line_1)
    direction_line2 = calcLineDirection(line_2)
    for c_1 in range(2):
        if pointConnected:
            break
        for c_2 in range(2):
            # 新增斜墙判断逻辑 2023.06.28 henry.hao
            if direction_line1 == -1 or direction_line2 == -1:
                distance = pointDistance(line_1[c_1], line_2[c_2])
                if distance > 10:
                    continue
                connectionPoint = ((line_1[c_1][0] + line_2[c_2][0]) // 2, (line_1[c_1][1] + line_2[c_2][1]) // 2)
                connection_1 = 6
                if c_1 == 0 and c_2 == 1:
                    connection_2 = 1
                elif c_1 == 0 and c_2 == 0:
                    connection_2 = 2
                elif c_1 == 1 and c_2 == 0:
                    connection_2 = 3
                elif c_1 == 1 and c_2 == 1:
                    connection_2 = 2
                pointConnected = True
                break
            else:
                distance = pointDistance(line_1[c_1], line_2[c_2])
                if distance > gap:
                    continue
                connection_1 = c_1
                connection_2 = c_2
                connectionPoint = ((line_1[c_1][0] + line_2[c_2][0]) // 2, (line_1[c_1][1] + line_2[c_2][1]) // 2)
                pointConnected = True
                break
        continue
    if pointConnected:
        # L｜钩 Shape，假设封闭区域的L corner Point distance<gap
        return [connection_1, connection_2], connectionPoint
    direction_1, fixedValue_1, min_1, max_1 = lineRange(line_1)
    direction_2, fixedValue_2, min_2, max_2 = lineRange(line_2)
    
    # 平行 或 降噪后平行，这里的gpa容易把小的断头墙干掉
    if direction_1 == direction_2:
        return [-1, -1], (0, 0)
    if min(fixedValue_1, max_2) < max(fixedValue_1, min_2) - gap or min(fixedValue_2, max_1) < max(
        fixedValue_2,
        min_1
    ) - gap:
        return [-1, -1], (0, 0)
    
    # T Shape
    if abs(min_1 - fixedValue_2) <= gap:
        return [0, 2], (fixedValue_2, fixedValue_1)
    if abs(max_1 - fixedValue_2) <= gap:
        return [1, 2], (fixedValue_2, fixedValue_1)
    if abs(min_2 - fixedValue_1) <= gap:
        return [2, 0], (fixedValue_2, fixedValue_1)
    if abs(max_2 - fixedValue_1) <= gap:
        return [2, 1], (fixedValue_2, fixedValue_1)
    return [2, 2], (fixedValue_2, fixedValue_1)


def lines2Corners(lines, gap):
    success = True
    lineConnections = []
    for _ in range(len(lines)):
        lineConnections.append({})
        continue
    
    connectionCornerMap = {}
    connectionCornerMap[(6, 1)] = 0  # 上钩
    connectionCornerMap[(6, 2)] = 1  # 右钩
    connectionCornerMap[(6, 3)] = 2  # 下钩
    connectionCornerMap[(6, 4)] = 3  # 左钩
    connectionCornerMap[(1, 1)] = 4
    connectionCornerMap[(0, 1)] = 5
    connectionCornerMap[(0, 0)] = 6
    connectionCornerMap[(1, 0)] = 7
    connectionCornerMap[(2, 0)] = 8
    connectionCornerMap[(1, 2)] = 9
    connectionCornerMap[(2, 1)] = 10
    connectionCornerMap[(0, 2)] = 11
    connectionCornerMap[(2, 2)] = 12
    corners = []
    for lineIndex_1, line_1 in enumerate(lines):
        for lineIndex_2, line_2 in enumerate(lines):
            if lineIndex_2 == lineIndex_1:
                continue
            connections, connectionPoint = findConnections(line_1, line_2, gap=gap)
            if connections[0] == -1 and connections[1] == -1:
                continue
            if calcLineDirection(line_1) == calcLineDirection(line_2) and isManhattan(line_1) and isManhattan(line_2):
                success = False
                continue
            if calcLineDirection(line_1) == 1:
                continue
            
            indices = [lineIndex_1, lineIndex_2]
            for c in range(2):
                if connections[c] in [0, 1] and connections[c] in lineConnections[indices[c]] and isManhattan(
                    line_1
                ) and isManhattan(line_2):
                    success = False
                    continue
                lineConnections[indices[c]][connections[c]] = True
                continue
            corners.append((connectionPoint, connectionCornerMap[tuple(connections)]))
            continue
        continue
    return corners, success


def getRoomLabelMap():
    labelMap = {}
    labelMap['living_room'] = 1
    labelMap['kitchen'] = 2
    labelMap['bedroom'] = 3
    labelMap['bathroom'] = 4
    labelMap['restroom'] = 4
    labelMap['washing_room'] = 4
    labelMap['office'] = 3
    labelMap['closet'] = 6
    labelMap['balcony'] = 7
    labelMap['corridor'] = 8
    labelMap['dining_room'] = 9
    labelMap['laundry_room'] = 10
    labelMap['PS'] = 10
    return labelMap


def getIconLabelMap():
    labelMap = {}
    labelMap['bathtub'] = 1
    labelMap['cooking_counter'] = 2
    labelMap['toilet'] = 3
    labelMap['entrance'] = 4
    labelMap['washing_basin'] = 5
    labelMap['special'] = 6
    labelMap['stairs'] = 7
    return labelMap


def loadLabelMap():
    roomMap = getRoomLabelMap()
    iconMap = getIconLabelMap()
    
    labelMap = {}
    for icon, label in iconMap.items():
        labelMap[icon] = ('icons', label)
        continue
    for room, label in roomMap.items():
        labelMap[room] = ('rooms', label)
        continue
    labelMap['door'] = 8
    return labelMap


def augmentSample(options, image, background_colors=[], split='train'):
    max_size = np.random.randint(low=int(options.width * 3 / 4), high=options.width + 1)
    if split != 'train':
        max_size = options.width
        pass
    image_sizes = np.array(image.shape[:2]).astype(np.float32)
    transformation = np.zeros((3, 3))
    transformation[0][0] = transformation[1][1] = float(max_size) / image_sizes.max()
    transformation[2][2] = 1
    image_sizes = (image_sizes / image_sizes.max() * max_size).astype(np.int32)
    
    if image_sizes[1] == options.width or split != 'train':
        offset_x = 0
    else:
        offset_x = np.random.randint(options.width - image_sizes[1])
        pass
    if image_sizes[0] == options.height or split != 'train':
        offset_y = 0
    else:
        offset_y = np.random.randint(options.height - image_sizes[0])
        pass
    
    transformation[0][2] = offset_x
    transformation[1][2] = offset_y
    
    if len(background_colors) == 0:
        full_image = np.full((options.height, options.width, 3), fill_value=255)
    else:
        full_image = background_colors[np.random.choice(
            np.arange(len(background_colors), dtype=np.int32),
            options.width * options.height
        )].reshape(
            (options.height, options.width, 3)
        )
        pass
    
    # full_image = np.full((options.height, options.width, 3), fill_value=-1, dtype=np.float32)
    full_image[offset_y:offset_y + image_sizes[0], offset_x:offset_x + image_sizes[1]] = cv2.resize(
        image, (
            image_sizes[1], image_sizes[0])
    )
    image = full_image
    
    return image, transformation


def convertToPoint(x, y):
    return (int(round(float(x))), int(round(float(y))))


def transformPoint(transformation, point):
    point = np.array(point)
    point = np.concatenate([point, np.ones(1)], axis=0)
    point = np.matmul(transformation, point)
    return tuple(np.round(point[:2] / point[2]).astype(np.int32).tolist())


# Plane dataset class
class FloorplanDataset(Dataset):
    def __init__(self, options, split, dataFolder, random=False):
        self.options = options
        self.split = split
        self.random = random
        self.imagePaths = []
        # self.dataFolder = '/Users/hehao/Desktop/Henry/IKEA/Prometheus/IKEA_img2floorplan/models/datasets/'
        self.dataFolder = dataFolder
        with open(os.path.join(self.dataFolder, split + '.txt')) as f:
            
            for line in f:
                self.imagePaths.append([value.strip() for value in line.split('\t')])  # 第一项是图片地址,第二项是标注地址
                continue
        
        if options.numTrainingImages > 0 and split == 'train':
            self.numImages = len(self.imagePaths)
        else:
            self.numImages = len(self.imagePaths)
            pass
        self.labelMap = loadLabelMap()  # icon和room的label
        return
    
    def __len__(self):
        return self.numImages
    
    def __getitem__(self, index):
        
        if self.random:
            t = int(time.time() * 1000000)
            np.random.seed(
                ((t & 0xff000000) >> 24) +
                ((t & 0x00ff0000) >> 8) +
                ((t & 0x0000ff00) << 8) +
                ((t & 0x000000ff) << 24)
            )
            index = np.random.randint(len(self.imagePaths))
        else:
            index = index % len(self.imagePaths)
            pass
        
        debug = -1
        if debug >= 0:
            index = debug
            pass
        
        image = cv2.imread(os.path.join(self.dataFolder, self.imagePaths[index][0]))
        random_int = np.random.randint(0, 2)
        transpose_flag = False if random_int == 0 else True
        if transpose_flag:
            image = np.transpose(image, axes=(1, 0, 2))
        image_width, image_height = image.shape[1], image.shape[0]
        
        walls = []
        wall_types = []
        openings = []
        doors = []
        semantics = {}
        with open(os.path.join(self.dataFolder, self.imagePaths[index][1])) as info_file:
            # 打开的是图片对应的标注
            line_index = 0
            for line in info_file:
                line = line.split('\t')
                label = line[4].strip()
                if label == 'wall':
                    walls.append((convertToPoint(line[0], line[1]), convertToPoint(line[2], line[3])))
                    wall_types.append(int(line[5].strip()) - 1)
                elif label in ['opening']:
                    opening_type = line[5].strip()
                    openings.append((convertToPoint(line[0], line[1]), convertToPoint(line[2], line[3]), opening_type))
                elif label in ['door']:
                    door_type = line[5].strip()
                    door_direction = line[6].strip()
                    # Transpose the door direction.
                    if transpose_flag:
                        if int(door_direction) == 1:
                            door_direction = '3'
                        elif int(door_direction) == 3:
                            door_direction = '1'
                    
                    doors.append(
                        (convertToPoint(line[0], line[1]), convertToPoint(line[2], line[3]), door_type, door_direction)
                    )
                else:
                    if label not in semantics:
                        semantics[label] = []
                        pass
                    semantics[label].append((convertToPoint(line[0], line[1]), convertToPoint(line[2], line[3])))
                    pass
                continue
            pass
        
        gap = 5
        # print(semantics)
        invalid_indices = {}
        for wall_index_1, (wall_1, wall_type_1) in enumerate(zip(walls, wall_types)):
            for wall_index_2, (wall_2, wall_type_2) in enumerate(zip(walls, wall_types)):
                if wall_type_1 == 0 and wall_type_2 == 1 and calcLineDirection(wall_1) == calcLineDirection(wall_2):
                    if min(
                        [pointDistance(wall_1[c_1], wall_2[c_2]) for c_1, c_2 in
                            [(0, 0), (0, 1), (1, 0), (1, 1)]]
                    ) <= gap * 2:
                        walls[wall_index_1] = mergeLines(wall_1, wall_2)
                        invalid_indices[wall_index_2] = True
                        pass
                    pass
                continue
            continue
        walls = [wall for wall_index, wall in enumerate(walls) if wall_index not in invalid_indices]
        
        background_mask = measure.label(1 - drawWallMask(walls, image_width, image_height), background=0)
        wall_index = background_mask.min()
        background_colors = []
        if np.random.randint(2) == 0:
            for pixel in [(0, 0), (0, background_mask.shape[0] - 1), (background_mask.shape[1] - 1, 0),
                (background_mask.shape[1] - 1, background_mask.shape[0] - 1)]:
                index = background_mask[pixel[1]][pixel[0]]
                if index != wall_index:
                    background_colors = image[background_mask == index]
                    break
                continue
            pass
        
        corners, success = lines2Corners(walls, gap=gap)
        if not success:
            pass
        
        if self.split == 'train':
            image, transformation = augmentSample(self.options, image, background_colors)
        else:
            image, transformation = augmentSample(self.options, image, background_colors, split=self.split)
            pass
        
        corners = [(transformPoint(transformation, corner[0]), corner[1]) for corner in corners]
        walls = [[transformPoint(transformation, wall[c]) for c in range(2)] for wall in walls]
        openings = [[transformPoint(transformation, opening[0]), transformPoint(transformation, opening[1]), opening[2]]
            for opening in openings]
        doors = [[transformPoint(transformation, door[0]), transformPoint(transformation, door[1]), door[2], door[3]]
            for door in doors]
        for semantic, items in semantics.items():
            semantics[semantic] = [[transformPoint(transformation, item[c]) for c in range(2)] for item in items]
            continue
        
        width = self.options.width
        height = self.options.height
        
        roomSegmentation = np.zeros((height, width), dtype=np.uint8)
        for line in walls:
            cv2.line(roomSegmentation, line[0], line[1], color=NUM_ROOMS + 1, thickness=gap)
            continue
        
        rooms = measure.label(roomSegmentation == 0, background=0)
        # 墙
        corner_gt = []
        for corner in corners:
            corner_gt.append((corner[0][0], corner[0][1], corner[1] + 1))
            continue
        # 窗
        openingCornerMap = [[3, 1], [0, 2]]
        for opening in openings:
            direction = calcLineDirection(opening)
            opening_type = int(opening[2])
            for cornerIndex, corner in enumerate(opening):
                if cornerIndex > 1:
                    break
                corner_gt.append(
                    (int(round(corner[0])), int(round(corner[1])),
                    14 + 4 * opening_type + openingCornerMap[direction][cornerIndex])
                )
                continue
            continue
        # 4种类型的门
        for door in doors:
            # direction 水平还是竖直
            direction = calcLineDirection(door)
            # 1. 单开门 2.双开门 3.门窗合体，类似单开门 4. 双移门
            # 单开门, 双开门，门窗合体的点各有八种情况。(4 * 2, 2表示是在门direction是在墙左右，还是在墙上下。)
            # 双移门： 4种情况， 不需要区分门的direction是否在墙左右还是墙上下。
            door_type = int(door[2]) - 1
            door_direction = int(door[3])
            #
            for cornerIndex, corner in enumerate(door):
                # 单开门，双开门，门窗合体
                if door_type < 3:
                    # corner_index = 0, 1, 2, 3
                    corner_index = openingCornerMap[direction][cornerIndex]
                    # 水平门
                    # 开门的方向在水平线之下。
                    if corner_index in [3, 1] and door_direction in [3, 0]:
                        sub_offset = 1
                    # 开门的方向在竖直线右侧
                    elif corner_index in [2, 0] and door_direction in [2, 3]:
                        sub_offset = 1
                    else:
                        sub_offset = 0
                    offset = 2 * corner_index + sub_offset
                    corner_gt.append(
                        (int(round(corner[0])), int(round(corner[1])), 22 + 8 * door_type + offset)
                    )
                else:
                    # door_type = 4 双移门
                    corner_gt.append(
                        (int(round(corner[0])), int(round(corner[1])),
                        22 + 24 + openingCornerMap[direction][cornerIndex])
                    )
                
                if cornerIndex >= 1:
                    break
            continue
        wallIndex = rooms.min()
        for pixel in [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)]:
            backgroundIndex = rooms[pixel[1]][pixel[0]]
            if backgroundIndex != wallIndex:
                break
            continue
        iconSegmentation = np.zeros((height, width), dtype=np.uint8)
        for line in doors:
            cv2.line(iconSegmentation, line[0], line[1], color=self.labelMap['door'], thickness=gap - 1)
            continue
        
        roomLabelMap = {}
        for semantic, items in semantics.items():
            group, label = self.labelMap[semantic]
            for corners in items:
                if group == 'icons':
                    if label == 0:
                        continue
                    cv2.rectangle(
                        iconSegmentation, (int(round(corners[0][0])), int(round(corners[0][1]))),
                        (int(round(corners[1][0])), int(round(corners[1][1]))), color=label, thickness=-1
                    )
                    corner_gt.append((corners[0][0], corners[0][1], 18 + 2))
                    corner_gt.append((corners[0][0], corners[1][1], 18 + 1))
                    corner_gt.append((corners[1][0], corners[0][1], 18 + 3))
                    corner_gt.append((corners[1][0], corners[1][1], 18 + 0))
                continue
            continue
        
        if debug >= 0:
            cv2.imwrite('test/floorplan/rooms.png', drawSegmentationImage(rooms, blackIndex=backgroundIndex))
            exit(1)
            pass
        
        for roomIndex in range(rooms.min(), rooms.max() + 1):
            if roomIndex == wallIndex or roomIndex == backgroundIndex:
                continue
            if roomIndex not in roomLabelMap:
                roomSegmentation[rooms == roomIndex] = 10
                pass
            continue
        
        cornerSegmentation = np.zeros((height, width, NUM_CORNERS), dtype=np.uint8)
        for corner in corner_gt:
            cornerSegmentation[min(max(corner[1], 0), height - 1), min(max(corner[0], 0), width - 1), corner[2] - 1] = 1
            continue
        
        image_copy = copy.deepcopy(image)
        image = (image.astype(np.float32) / 255 - 0.5).transpose((2, 0, 1))
        kernel = np.zeros((3, 3), dtype=np.uint8)
        kernel[1] = 1
        kernel[:, 1] = 1
        cornerSegmentation = cv2.dilate(cornerSegmentation, kernel, iterations=5)
        
        if True:
            cv2.imwrite('test/image' + str(index) + '.png', image_copy)
            cv2.imwrite('test/icon_segmentation' + str(index) + '.png', drawSegmentationImage(iconSegmentation))
            cv2.imwrite('test/room_segmentation' + str(index) + '.png', drawSegmentationImage(roomSegmentation))
            cv2.imwrite(
                'test/corner_segmentation' + str(index) + '.png',
                drawSegmentationImage(cornerSegmentation, blackIndex=0)
            )
            # exit(1)
            # pass
        
        sample = [image, cornerSegmentation.astype(np.float32), iconSegmentation.astype(np.int64),
            roomSegmentation.astype(np.int64)]
        return sample
