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
