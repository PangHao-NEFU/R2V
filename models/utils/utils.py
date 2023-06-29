import numpy as np
import cv2


def lines2Corners(lines, gap):
    success = True
    lineConnections = []
    for _ in range(len(lines)):
        lineConnections.append({})
        continue

    connectionCornerMap = {}
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
                        line_1) and isManhattan(line_2):
                    success = False
                    continue
                lineConnections[indices[c]][connections[c]] = True
                continue
            corners.append((connectionPoint, connectionCornerMap[tuple(connections)]))
            continue
        continue
    return corners, success


def isManhattan(line, gap=3):
    return min(abs(line[0][0] - line[1][0]), abs(line[0][1] - line[1][1])) < gap


def calcLineDirection(line, gap=3):
    # 0 水平；1 垂直
    return int(abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]))


def lineRange(line):
    # direction: 0 水平；1 垂直
    direction = calcLineDirection(line)
    fixedValue = (line[0][1 - direction] + line[1][1 - direction]) // 2
    minValue = min(line[0][direction], line[1][direction])
    maxValue = max(line[0][direction], line[1][direction])
    return direction, fixedValue, minValue, maxValue


def pointDistance(point_1, point_2):
    return max(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))


def findConnections(line_1, line_2, gap):
    connection_1 = -1
    connection_2 = -1
    pointConnected = False
    for c_1 in range(2):
        if pointConnected:
            break
        for c_2 in range(2):
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
        return [connection_1, connection_2], connectionPoint
    direction_1, fixedValue_1, min_1, max_1 = lineRange(line_1)
    direction_2, fixedValue_2, min_2, max_2 = lineRange(line_2)
    if direction_1 == direction_2:
        return [-1, -1], (0, 0)
    if min(fixedValue_1, max_2) < max(fixedValue_1, min_2) - gap or min(fixedValue_2, max_1) < max(fixedValue_2,
                                                                                                   min_1) - gap:
        return [-1, -1], (0, 0)

    if abs(min_1 - fixedValue_2) <= gap:
        return [0, 2], (fixedValue_2, fixedValue_1)
    if abs(max_1 - fixedValue_2) <= gap:
        return [1, 2], (fixedValue_2, fixedValue_1)
    if abs(min_2 - fixedValue_1) <= gap:
        return [2, 0], (fixedValue_2, fixedValue_1)
    if abs(max_2 - fixedValue_1) <= gap:
        return [2, 1], (fixedValue_2, fixedValue_1)
    return [2, 2], (fixedValue_2, fixedValue_1)


def convertToPoint(x, y):
    return (int(round(float(x))), int(round(float(y))))


def mergeLines(line_1, line_2):
    direction_1, fixedValue_1, min_1, max_1 = lineRange(line_1)
    direction_2, fixedValue_2, min_2, max_2 = lineRange(line_2)
    fixedValue = (fixedValue_1 + fixedValue_2) // 2
    if direction_1 == 0:
        return [(min(min_1, min_2), fixedValue), (max(max_1, max_2), fixedValue)]
    else:
        return [(fixedValue, min(min_1, min_2)), (fixedValue, max(max_1, max_2))]
    return


def draw_points(all_wall_points, line_width=5, file_name="Node", background_img_data=None, rgb_color=[0, 0, 255]):
    img_data = background_img_data
    line_color = np.random.rand(3) * 255
    line_color[0] = rgb_color[0]
    line_color[1] = rgb_color[1]
    line_color[2] = rgb_color[2]

    floor_plan_img_height = img_data.shape[0]
    floor_plan_img_width = img_data.shape[1]

    for cur_wall_point in all_wall_points:
        for pointInfo in cur_wall_point:
            x = pointInfo[0]
            y = pointInfo[1]
            img_data[max(y - line_width, 0):min(y + line_width, floor_plan_img_height - 1),
            max(x - line_width, 0):min(x + line_width, floor_plan_img_width - 1)] = line_color
            cv2.putText(img_data, "("+str(x)+","+str(y)+")", (x, y),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255, 255, 0))
    cv2.imwrite(file_name, img_data)

def calc_line_dim(self, point_1, point_2, threshold=5, space_flag=False):
    # space_flag.
    if not space_flag:
        if np.abs(point_2.x - point_1.x) > threshold and np.abs(point_2.y - point_1.y) > threshold:
            return -1

    if np.abs(point_2.x - point_1.x) > np.abs(point_2.y - point_1.y):
        return 0
    else:
        return 1
def draw_lines(all_wall_lines, line_width=2, file_name="Node", background_img_data=None, rgb_color=[0, 0, 255]):
    try:
        image = background_img_data

        floor_plan_img_height = image.shape[0]
        floor_plan_img_width = image.shape[1]

        i = 0
        for wall_line in all_wall_lines:
            line_color = np.random.rand(3) * 255
            if rgb_color is not None:
                line_color[0] = rgb_color[0]
                line_color[1] = rgb_color[1]
                line_color[2] = rgb_color[2]

            point_1 = wall_line[0]
            point_2 = wall_line[1]

            i += 1

            line_dim = calcLineDirection(wall_line)

            fixedValue = int(round((point_1[1] + point_2[1]) / 2)) if line_dim == 0 else int(
                round((point_1[0] + point_2[0]) / 2))
            minValue = int(round(min(point_1[0], point_2[0]))) if line_dim == 0 else int(
                round(min(point_1[1], point_2[1])))
            maxValue = int(round(max(point_1[0], point_2[0]))) if line_dim == 0 else int(
                round(max(point_1[1], point_2[1])))

            if line_dim == 0:
                image[max(fixedValue - line_width, 0):min(fixedValue + line_width, floor_plan_img_height),
                minValue:maxValue + 1, :] = line_color
            else:
                image[minValue:maxValue + 1,
                max(fixedValue - line_width, 0):min(fixedValue + line_width, floor_plan_img_width),:] = line_color
        cv2.imwrite(file_name, image)

    except Exception as err:
        print(err)


def draw_points_with_corners(all_wall_points, line_width=5, file_name="Node", background_img_data=None,
                             rgb_color=[0, 0, 255]):
    try:
        img_data = background_img_data
        line_color = np.random.rand(3) * 255
        line_color[0] = rgb_color[0]
        line_color[1] = rgb_color[1]
        line_color[2] = rgb_color[2]

        floor_plan_img_height = img_data.shape[0]
        floor_plan_img_width = img_data.shape[1]

        for cur_wall_point, corner in all_wall_points:
            x = cur_wall_point[0]
            y = cur_wall_point[1]

            img_data[max(y - line_width, 0):min(y + line_width, floor_plan_img_height - 1),
            max(x - line_width, 0):min(x + line_width, floor_plan_img_width - 1)] = line_color
            cv2.putText(img_data, str(corner), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        cv2.imwrite(file_name, img_data)
    except Exception as err:
        print(err)


if __name__ == '__main__':
    walls = []
    wall_types = []
    semantics = {}
    with open(
            "/Users/hehao/Desktop/Henry/IKEA/Prometheus/IKEA_img2floorplan/models/test/0a0eccef-2277-4da0-9fe1-7277299af870/0a0eccef-2277-4da0-9fe1-7277299af870.txt") as info_file:
        line_index = 0
        for line in info_file:
            line = line.split('\t')
            label = line[4].strip()
            if label == 'wall':
                walls.append((convertToPoint(line[0], line[1]), convertToPoint(line[2], line[3])))
                wall_types.append(int(line[5].strip()) - 1)
            continue
        pass

    gap = 5
    invalid_indices = {}
    for wall_index_1, (wall_1, wall_type_1) in enumerate(zip(walls, wall_types)):
        for wall_index_2, (wall_2, wall_type_2) in enumerate(zip(walls, wall_types)):
            if wall_type_1 == 0 and wall_type_2 == 1 and calcLineDirection(wall_1) == calcLineDirection(wall_2):
                if min([pointDistance(wall_1[c_1], wall_2[c_2]) for c_1, c_2 in
                        [(0, 0), (0, 1), (1, 0), (1, 1)]]) <= gap * 2:
                    walls[wall_index_1] = mergeLines(wall_1, wall_2)
                    invalid_indices[wall_index_2] = True
                    pass
                pass
            continue
        continue
    walls = [wall for wall_index, wall in enumerate(walls) if wall_index not in invalid_indices]

    cv2_img = cv2.imread(
        "/Users/hehao/Desktop/Henry/IKEA/Prometheus/IKEA_img2floorplan/models/test/0a0eccef-2277-4da0-9fe1-7277299af870/0a0eccef-2277-4da0-9fe1-7277299af870_resized.png")
    draw_points(walls, file_name="DoorPoints.png", background_img_data=cv2_img)

    draw_lines(walls, file_name="DoorPoints.png", background_img_data=cv2_img)
    corners, success = lines2Corners(walls, gap=gap)
    print(walls)
    print(success, corners)
    cv2_img_copy = cv2.imread(
        "/Users/hehao/Desktop/Henry/IKEA/Prometheus/IKEA_img2floorplan/models/test/0a0eccef-2277-4da0-9fe1-7277299af870/0a0eccef-2277-4da0-9fe1-7277299af870_resized.png")
    draw_points_with_corners(corners, file_name="DoorPointsWithCorner.png", background_img_data=cv2_img_copy)

import numpy as np
import cv2

NUM_WALL_CORNERS = 13
NUM_CORNERS = 21
#CORNER_RANGES = {'wall': (0, 13), 'opening': (13, 17), 'icon': (17, 21)}

NUM_ICONS = 7
NUM_ROOMS = 10
POINT_ORIENTATIONS = [[(2, ), (3, ), (0, ), (1, )], [(0, 3), (0, 1), (1, 2), (2, 3)], [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)], [(0, 1, 2, 3)]]

class ColorPalette:
    def __init__(self, numColors):
        #np.random.seed(2)
        #self.colorMap = np.random.randint(255, size = (numColors, 3))
        #self.colorMap[0] = 0


        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [255, 255, 0],
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [0, 128, 255],
                                  [255, 0, 128],
                                  [128, 0, 255],
                                  [255, 128, 0],
                                  [128, 255, 0],
                                  ])

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.random.randint(255, size = (numColors, 3))
            pass

        return

    def getColorMap(self):
        return self.colorMap

    def getColor(self, index):
        if index >= self.colorMap.shape[0]:
            return np.random.randint(255, size = (3))
        else:
            return self.colorMap[index]
            pass
        return

def isManhattan(line, gap=3):
    return min(abs(line[0][0] - line[1][0]), abs(line[0][1] - line[1][1])) < gap

def calcLineDim(points, line):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    if abs(point_2[0] - point_1[0]) > abs(point_2[1] - point_1[1]):
        lineDim = 0
    else:
        lineDim = 1
        pass
    return lineDim

def calcLineDirection(line, gap=3):
    """
    计算一个墙的方向
    line: [(x1,y1),(x2,y2)]
    return : 0:水平的墙  1: 垂直的墙
    """
    return int(abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]))

# Draw segmentation image. The input could be either HxW or HxWxC
def drawSegmentationImage(segmentations, numColors=42, blackIndex=-1, blackThreshold=-1):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2)
    else:
        if blackThreshold > 0:
            segmentations = np.concatenate([segmentations, np.ones((segmentations.shape[0], segmentations.shape[1], 1)) * blackThreshold], axis=2)
            blackIndex = segmentations.shape[2] - 1
            pass

        numColors = max(numColors, segmentations.shape[2] + 2)
        pass
    randomColor = ColorPalette(numColors).getColorMap()
    if blackIndex >= 0:
        randomColor[blackIndex] = 0
        pass
    width = segmentations.shape[1]
    height = segmentations.shape[0]
    if segmentations.ndim == 3:
        #segmentation = (np.argmax(segmentations, 2) + 1) * (np.max(segmentations, 2) > 0.5)
        segmentation = np.argmax(segmentations, 2)
    else:
        segmentation = segmentations
        pass

    segmentation = segmentation.astype(np.int32)
    return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))


def drawWallMask(walls, width, height, thickness=3, indexed=False):
    if indexed:
        wallMask = np.full((height, width), -1, dtype=np.int32)
        for wallIndex, wall in enumerate(walls):
            cv2.line(wallMask, (int(wall[0][0]), int(wall[0][1])), (int(wall[1][0]), int(wall[1][1])), color=wallIndex, thickness=thickness)
            continue
    else:
        wallMask = np.zeros((height, width), dtype=np.int32)
        for wall in walls:
            cv2.line(wallMask, (int(wall[0][0]), int(wall[0][1])), (int(wall[1][0]), int(wall[1][1])), color=1, thickness=thickness)
            continue
        wallMask = wallMask.astype(np.bool)
        pass
    return wallMask


def extractCornersFromHeatmaps(heatmaps, heatmapThreshold=0.5, numPixelsThreshold=5, returnRanges=True):
    """Extract corners from heatmaps"""
    from skimage import measure
    heatmaps = (heatmaps > heatmapThreshold).astype(np.float32)
    orientationPoints = []
    #kernel = np.ones((3, 3), np.float32)
    for heatmapIndex in range(0, heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, heatmapIndex]
        #heatmap = cv2.dilate(cv2.erode(heatmap, kernel), kernel)
        components = measure.label(heatmap, background=0)
        points = []
        for componentIndex in range(components.min() + 1, components.max() + 1):
            ys, xs = (components == componentIndex).nonzero()
            if ys.shape[0] <= numPixelsThreshold:
                continue
            #print(heatmapIndex, xs.shape, ys.shape, componentIndex)
            if returnRanges:
                points.append(((xs.mean(), ys.mean()), (xs.min(), ys.min()), (xs.max(), ys.max())))
            else:
                points.append((xs.mean(), ys.mean()))
                pass
            continue
        orientationPoints.append(points)
        continue
    return orientationPoints

def extractCornersFromSegmentation(segmentation, cornerTypeRange=[0, 13]):
    """Extract corners from segmentation"""
    from skimage import measure
    orientationPoints = []
    for heatmapIndex in range(cornerTypeRange[0], cornerTypeRange[1]):
        heatmap = segmentation == heatmapIndex
        #heatmap = cv2.dilate(cv2.erode(heatmap, kernel), kernel)
        components = measure.label(heatmap, background=0)
        points = []
        for componentIndex in range(components.min()+1, components.max() + 1):
            ys, xs = (components == componentIndex).nonzero()
            points.append((xs.mean(), ys.mean()))
            continue
        orientationPoints.append(points)
        continue
    return orientationPoints

def getOrientationRanges(width, height):
    orientationRanges = [[width, 0, 0, 0], [width, height, width, 0], [width, height, 0, height], [0, height, 0, 0]]
    return orientationRanges

def getIconNames():
    iconNames = []
    iconLabelMap = getIconLabelMap()
    for iconName, _ in iconLabelMap.items():
        iconNames.append(iconName)
        continue
    return iconNames

def getIconLabelMap():
    labelMap = {}
    labelMap['bathtub'] = 1
    labelMap['cooking_counter'] = 2
    labelMap['toilet'] = 3
    labelMap['entrance'] = 4
    labelMap['washing_basin'] = 5
    labelMap['special'] = 6
    labelMap['stairs'] = 7
    labelMap['door'] = 8
    return labelMap


def drawPoints(filename, width, height, points, backgroundImage=None, pointSize=5, pointColor=None):
    colorMap = ColorPalette(NUM_CORNERS).getColorMap()
    if np.all(np.equal(backgroundImage, None)):
        image = np.zeros((height, width, 3), np.uint8)
    else:
        if backgroundImage.ndim == 2:
            image = np.tile(np.expand_dims(backgroundImage, -1), [1, 1, 3])
        else:
            image = backgroundImage
            pass
    pass
    no_point_color = pointColor is None
    for point in points:
        if no_point_color:
            pointColor = colorMap[point[2] * 4 + point[3]]
            pass
        #print('used', pointColor)
        #print('color', point[2] , point[3])
        image[max(int(round(point[1])) - pointSize, 0):min(int(round(point[1])) + pointSize, height), max(int(round(point[0])) - pointSize, 0):min(int(round(point[0])) + pointSize, width)] = pointColor
        continue

    if filename != '':
        cv2.imwrite(filename, image)
        return
    else:
        return image

def drawPointsSeparately(path, width, height, points, backgroundImage=None, pointSize=5):
    if np.all(np.equal(backgroundImage, None)):
        image = np.zeros((height, width, 13), np.uint8)
    else:
        image = np.tile(np.expand_dims(backgroundImage, -1), [1, 1, 13])
        pass

    for point in points:
        image[max(int(round(point[1])) - pointSize, 0):min(int(round(point[1])) + pointSize, height), max(int(round(point[0])) - pointSize, 0):min(int(round(point[0])) + pointSize, width), int(point[2] * 4 + point[3])] = 255
        continue
    for channel in range(13):
        cv2.imwrite(path + '_' + str(channel) + '.png', image[:, :, channel])
        continue
    return

def drawLineMask(width, height, points, lines, lineWidth = 5, backgroundImage = None):
    lineMask = np.zeros((height, width))

    for lineIndex, line in enumerate(lines):
        point_1 = points[line[0]]
        point_2 = points[line[1]]
        direction = calcLineDirectionPoints(points, line)

        fixedValue = int(round((point_1[1 - direction] + point_2[1 - direction]) / 2))
        minValue = int(min(point_1[direction], point_2[direction]))
        maxValue = int(max(point_1[direction], point_2[direction]))
        if direction == 0:
            lineMask[max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue + 1] = 1
        else:
            lineMask[minValue:maxValue + 1, max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, width)] = 1
            pass
        continue
    return lineMask



def drawLines(filename, width, height, points, lines, lineLabels = [], backgroundImage = None, lineWidth = 5, lineColor = None):
    colorMap = ColorPalette(len(lines)).getColorMap()
    if backgroundImage is None:
        image = np.ones((height, width, 3), np.uint8) * 0
    else:
        if backgroundImage.ndim == 2:
            image = np.stack([backgroundImage, backgroundImage, backgroundImage], axis=2)
        else:
            image = backgroundImage
            pass
        pass

    for lineIndex, line in enumerate(lines):
        point_1 = points[line[0]]
        point_2 = points[line[1]]
        direction = calcLineDirectionPoints(points, line)


        fixedValue = int(round((point_1[1 - direction] + point_2[1 - direction]) / 2))
        minValue = int(round(min(point_1[direction], point_2[direction])))
        maxValue = int(round(max(point_1[direction], point_2[direction])))
        if len(lineLabels) == 0:
            if np.any(lineColor == None):
                lineColor = np.random.rand(3) * 255
                pass
            if direction == 0:
                image[max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue + 1, :] = lineColor
            else:
                image[minValue:maxValue + 1, max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, width), :] = lineColor
        else:
            labels = lineLabels[lineIndex]
            isExterior = False
            if direction == 0:
                for c in range(3):
                    image[max(fixedValue - lineWidth, 0):min(fixedValue, height), minValue:maxValue, c] = colorMap[labels[0]][c]
                    image[max(fixedValue, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue, c] = colorMap[labels[1]][c]
                    continue
            else:
                for c in range(3):
                    image[minValue:maxValue, max(fixedValue - lineWidth, 0):min(fixedValue, width), c] = colorMap[labels[1]][c]
                    image[minValue:maxValue, max(fixedValue, 0):min(fixedValue + lineWidth + 1, width), c] = colorMap[labels[0]][c]
                    continue
                pass
            pass
        continue

    if filename == '':
        return image
    else:
        cv2.imwrite(filename, image)


def drawRectangles(filename, width, height, points, rectangles, labels, lineWidth = 2, backgroundImage = None, rectangleColor = None):
    colorMap = ColorPalette(NUM_ICONS).getColorMap()
    if backgroundImage is None:
        image = np.ones((height, width, 3), np.uint8) * 0
    else:
        image = backgroundImage
        pass

    for rectangleIndex, rectangle in enumerate(rectangles):
        point_1 = points[rectangle[0]]
        point_2 = points[rectangle[1]]
        point_3 = points[rectangle[2]]
        point_4 = points[rectangle[3]]


        if len(labels) == 0:
            if rectangleColor is None:
                color = np.random.rand(3) * 255
            else:
                color = rectangleColor
        else:
            color = colorMap[labels[rectangleIndex]]
            pass

        x_1 = int(round((point_1[0] + point_3[0]) / 2))
        x_2 = int(round((point_2[0] + point_4[0]) / 2))
        y_1 = int(round((point_1[1] + point_2[1]) / 2))
        y_2 = int(round((point_3[1] + point_4[1]) / 2))

        cv2.rectangle(image, (x_1, y_1), (x_2, y_2), color=tuple(color.tolist()), thickness = 2)
        continue

    if filename == '':
        return image
    else:
        cv2.imwrite(filename, image)
        pass

def pointDistance(point_1, point_2):
    #return np.sqrt(pow(point_1[0] - point_2[0], 2) + pow(point_1[1] - point_2[1], 2))
    return max(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))

def calcLineDirectionPoints(points, line):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    if isinstance(point_1[0], tuple):
        point_1 = point_1[0]
        pass
    if isinstance(point_2[0], tuple):
        point_2 = point_2[0]
        pass
    return calcLineDirection((point_1, point_2))
