import math

import cv2
from Primitive import *


class WallAlignment(object):
    def __init__(self, image_data, all_wall_lines, all_wall_points, all_opening_points, all_door_points):

        self.image_data = image_data
        self.image_height = self.image_data.shape[0]
        self.image_width = self.image_data.shape[1]

        self.all_wall_lines = all_wall_lines
        self.all_wall_points = all_wall_points

        self.all_opening_points = all_opening_points
        self.all_door_points = all_door_points

        self.image_gray_data = image_data
        self.max_scan_number = 10  # 20个像素点。

        self.outer_walls_thickness = []
        self.inner_walls_thickness = []

        super(WallAlignment, self).__init__()

    def _predict_outer_walls(self):
        for cur_wall in self.all_wall_lines:
            cur_wall_dim = cur_wall.line_dim()
            line_x_center = int(0.5 * (cur_wall.start_point.x + cur_wall.end_point.x))
            line_y_center = int(0.5 * (cur_wall.start_point.y + cur_wall.end_point.y))

            # 水平墙，用两条竖直去搜索。
            if cur_wall_dim == 0:
                line1 = [line_x_center, 0, line_x_center, line_y_center]
                line2 = [line_x_center, line_y_center, line_x_center, self.image_height]
            elif cur_wall_dim == 1:
                line1 = [0, line_y_center, line_x_center, line_y_center]
                line2 = [line_x_center, line_y_center, self.image_width, line_y_center]
            elif cur_wall_dim == -1:  # 斜墙处理
                line1 = [cur_wall.start_point.x, cur_wall.start_point.y, cur_wall.end_point.x, cur_wall.end_point.y]
                line2 = [cur_wall.start_point.x, cur_wall.start_point.y, cur_wall.end_point.x, cur_wall.end_point.y]
            # line coordinate.
            intersect_res = []
            for line_coordinate in [line1, line2]:
                # 如果两条line都有相交的wall，那么是内墙
                # 如果其中有一条line有相交，另外一条line没有相交，那么是外墙。
                has_intersect_wall = False
                for other_wall in self.all_wall_lines:
                    if cur_wall == other_wall:
                        continue

                    other_wall_dim = other_wall.line_dim()
                    if cur_wall_dim != other_wall_dim and other_wall_dim!=-1:
                        continue

                    if other_wall_dim == 0:
                        # 有相交
                        if line_coordinate[1] <= other_wall.start_point.y <= line_coordinate[3]:
                            if other_wall.start_point.x <= line_coordinate[0] <= other_wall.end_point.x:
                                has_intersect_wall = True
                                break
                    elif other_wall_dim == 1:
                        if line_coordinate[0] <= other_wall.start_point.x <= line_coordinate[2]:
                            if other_wall.start_point.y <= line_coordinate[1] <= other_wall.end_point.y:
                                has_intersect_wall = True
                                break
                    elif other_wall_dim == -1:
                        if line_coordinate[0] <= other_wall.start_point.x <= line_coordinate[2]:
                            if other_wall.start_point.y <= line_coordinate[1] <= other_wall.end_point.y or other_wall.start_point.y >= line_coordinate[1] >= other_wall.end_point.y:
                                has_intersect_wall = True
                                break
                intersect_res.append(has_intersect_wall)
            cur_wall.is_outer_wall = True if False in intersect_res else False

    # 从start_pos开始，direction表示line的方向。
    # wall_sections, 由于门可能在墙上，这会增加判断精度，需要把门去掉。
    # increase_flag: direction = 0时， True向上寻找，False向下寻找。
    #                direction = 1时，True向右寻找，False向左寻找。
    #                direction = -1时，True向斜上寻找，False向斜下寻找。
    # 设置最大寻找次数，超过了寻找次数，寻找失败。
    def _find_boundary_line(self, start_pos, wall_sections, direction, increase_flag=True, max_scan_number=15):
        prev_wall_sections_gray_values = None
        boundary_pos = start_pos
        # 是否由深色变成浅色
        dark_to_light_flag = False
        prev_avg_gray_value = 0.0
        cur_avg_gray_value = 0.0

        found = False
        while not found:
            # 没有找到边界
            if np.abs(boundary_pos - start_pos) >= max_scan_number:
                return -1, False, cur_avg_gray_value

            cur_wall_sections_gray_values = []
            sum_gray_value = 0.0
            sum_length = 0.0

            for i in range(len(wall_sections)):
                cur_wall = wall_sections[i]
                if direction == 0:
                    if isinstance(cur_wall[0], int):
                        start_find_pos = cur_wall[0]
                        end_find_pos = cur_wall[1]
                    else:
                        start_find_pos = cur_wall[0].x
                        end_find_pos = cur_wall[1].x

                    # 找水平线
                    tmp_wall_gray_values = self.image_gray_data[boundary_pos, start_find_pos:end_find_pos + 1]
                    cur_wall_sections_gray_values.append(tmp_wall_gray_values)

                    sum_gray_value += np.sum(self.image_gray_data[boundary_pos, start_find_pos:end_find_pos + 1],
                                             axis=0)
                    sum_length += (end_find_pos - start_find_pos + 1)
                elif direction == 1:
                    if isinstance(cur_wall[0], int):
                        start_find_pos = cur_wall[0]
                        end_find_pos = cur_wall[1]
                    else:
                        start_find_pos = cur_wall[0].y
                        end_find_pos = cur_wall[1].y

                    # 找竖直线
                    tmp_wall_gray_values = self.image_gray_data[start_find_pos:end_find_pos + 1, boundary_pos]
                    cur_wall_sections_gray_values.append(tmp_wall_gray_values)

                    sum_gray_value += np.sum(self.image_gray_data[start_find_pos:end_find_pos + 1, boundary_pos],
                                             axis=0)
                    sum_length += (end_find_pos - start_find_pos + 1)
                elif direction == -1:
                    step = int(np.abs(boundary_pos - start_pos))
                    if increase_flag:
                        start_find_pos_x = cur_wall[0].x - step
                        start_find_pos_y = cur_wall[0].y - step
                        end_find_pos_x = cur_wall[1].x - step
                        end_find_pos_y = cur_wall[1].y - step
                    else:
                        start_find_pos_x = cur_wall[0].x + step
                        start_find_pos_y = cur_wall[0].y + step
                        end_find_pos_x = cur_wall[1].x + step
                        end_find_pos_y = cur_wall[1].y + step

                    line_piexl = self._calculate_line_piexl(start_find_pos_x, start_find_pos_y, end_find_pos_x,
                                                            end_find_pos_y)
                    gray_value = []
                    for point_info in line_piexl:
                        gray_value.append(np.array(self.image_gray_data[point_info[0], point_info[1]]))
                    gray_value = np.array(gray_value)
                    cur_wall_sections_gray_values.append(gray_value)
                    sum_gray_value += np.sum(gray_value, axis=0)
                    sum_length += len(line_piexl)

            cur_avg_gray_value = sum_gray_value / sum_length
            cur_avg_gray_value = np.clip(cur_avg_gray_value, 0, 255)
            cur_avg_gray_value = cur_avg_gray_value.reshape((1, 1, 3))
            cur_avg_gray_value = cur_avg_gray_value.astype(np.uint8)
            if prev_wall_sections_gray_values is None:
                prev_wall_sections_gray_values = cur_wall_sections_gray_values
                prev_avg_gray_value = cur_avg_gray_value
            else:
                diff_pixels_number = 0
                for i in range(len(cur_wall_sections_gray_values)):
                    cur_wall_section = cur_wall_sections_gray_values[i]

                    # 颜色变化比较大
                    prev_wall_hls_color = cv2.cvtColor(prev_avg_gray_value, cv2.COLOR_BGR2HLS)
                    cur_wall_hls_color = cv2.cvtColor(cur_avg_gray_value, cv2.COLOR_BGR2HLS)
                    prev_h, prev_l, prev_s = prev_wall_hls_color[0, 0, 0], prev_wall_hls_color[0, 0, 1], \
                                             prev_wall_hls_color[0, 0, 2]
                    cur_h, cur_l, cur_s = cur_wall_hls_color[0, 0, 0], cur_wall_hls_color[0, 0, 1], cur_wall_hls_color[
                        0, 0, 2]

                    if np.abs(np.int32(prev_l) - np.int32(cur_l)) > 50:
                        diff_pixels_number += cur_wall_section.shape[0]
                    if np.abs(np.int32(prev_s) - np.int32(cur_s)) > 50:
                        diff_pixels_number += cur_wall_section.shape[0]

                # 像素变化比较多。
                if diff_pixels_number > 0.1 * sum_length:
                    if np.mean(prev_avg_gray_value) > np.mean(cur_avg_gray_value):
                        dark_to_light_flag = False
                    else:
                        dark_to_light_flag = True
                    break

            # 往那个方向寻找
            if increase_flag:
                boundary_pos += 1
                if (direction == 0 and boundary_pos >= self.image_height) or (
                        direction == 1 and boundary_pos >= self.image_width):
                    return -1, False, np.mean(cur_avg_gray_value)
            else:
                boundary_pos -= 1
                if boundary_pos < 0:
                    return -1, False, np.mean(cur_avg_gray_value)
        if dark_to_light_flag:
            if direction == 0 and increase_flag:
                boundary_pos
            elif direction == 0 and not increase_flag:
                boundary_pos += 1
            elif direction == 1 and not increase_flag:
                boundary_pos += 1
        # return boundary_pos - 1 if increase_flag else boundary_pos+1
        return boundary_pos, dark_to_light_flag, np.mean(cur_avg_gray_value)

    def _calculate_line_piexl(self, x1, y1, x2, y2):
        if (x2 - x1) == 0:
            print('斜率不存在')
        a = (y2 - y1) / (x2 - x1)
        b = y1 - x1 * ((y2 - y1) / (x2 - x1))
        line_piexl = []
        for i in range(int(x2)):
            if i <= int(x1):
                continue
            elif i > int(x1) & i <= int(x2):
                y = int(a * i + b)
                line_piexl.append([i, y])  # 原直线
        line_piexl = np.array(line_piexl)
        return line_piexl

    def _align_wall_line(self, cur_wall_line):
        start_pt = cur_wall_line.start_point
        end_pt = cur_wall_line.end_point

        wall_length = cur_wall_line.get_wall_length()
        opening_length = 0
        for cur_opening in cur_wall_line.openings:
            # sum(墙上门/窗的length)
            opening_length += cur_opening.get_wall_length()
        target_wall_length = wall_length - opening_length
        # 比较小的wall判断容易出现问题,暂时不做处理 by henry.hao 2023.07.01
        # 墙是墙的像素差，窗是窗的像素差 by henry.hao 2023.07.11
        if target_wall_length < 10 or float(target_wall_length) / wall_length < 0.05:
            return False

        wall_line_dim = cur_wall_line.line_dim()
        wall_section_points = [start_pt, end_pt]
        for cur_opening in cur_wall_line.openings:
            wall_section_points.append(cur_opening.start_point)
            wall_section_points.append(cur_opening.end_point)
        if wall_line_dim == 0 or wall_line_dim == -1:
            wall_section_points = sorted(wall_section_points, key=lambda obj: obj.x)
        else:
            # 没有对斜墙的点做排序，可以通过斜率来计算斜墙方向 或者就按
            wall_section_points = sorted(wall_section_points, key=lambda obj: obj.y)

        wall_sections = []
        for i in range(0, len(wall_section_points), 2):
            wall_sections.append([wall_section_points[i], wall_section_points[i + 1]])

        start_pt_heatmap_x_center, start_pt_heatmap_y_center = start_pt.get_heat_map_centroid()
        end_pt_heatmap_x_center, end_pt_heatmap_y_center = end_pt.get_heat_map_centroid()

        if wall_line_dim == 0:
            # 如果平行X轴，计算平行方向上的厚度。找到变化的那条横线。
            center_pos = int(0.5 * (start_pt_heatmap_y_center + end_pt_heatmap_y_center))
        elif wall_line_dim == 1:
            # 如果平行Y轴，计算竖直方向上的厚度。找到变化的那条竖线。
            center_pos = int(0.5 * (start_pt_heatmap_x_center + end_pt_heatmap_x_center))
        elif wall_line_dim == -1:
            center_pos = start_pt_heatmap_x_center

        up_boundary_pos, up_dark_to_light_flag, up_boundary_gray_value = self._find_boundary_line(center_pos,
                                                                                                  wall_sections,
                                                                                                  wall_line_dim)
        down_boundary_pos, down_dark_to_light_flag, down_boundary_gray_value = self._find_boundary_line(center_pos,
                                                                                                        wall_sections,
                                                                                                        wall_line_dim,
                                                                                                        increase_flag=False)

        success_flag = True if up_boundary_pos > 0 and down_boundary_pos > 0 else False
        if not success_flag:
            return False

        # 上下边界都找到了，把上下边界的值设置给Wall Line.
        if cur_wall_line.is_outer_wall:
            if float(target_wall_length) / wall_length > 0.6:  # 经验值
                self.outer_walls_thickness.append(np.abs(up_boundary_pos - down_boundary_pos))
        else:
            self.inner_walls_thickness.append(np.abs(up_boundary_pos - down_boundary_pos))

        if wall_line_dim == 0:
            cur_wall_line.start_point.set_boundary(up_boundary_pos, 3, up_flag=True)
            cur_wall_line.start_point.set_boundary(down_boundary_pos, 3, up_flag=False)

            cur_wall_line.end_point.set_boundary(up_boundary_pos, 1, up_flag=True)
            cur_wall_line.end_point.set_boundary(down_boundary_pos, 1, up_flag=False)

            cur_wall_line.boundary_range_box = [start_pt_heatmap_x_center, down_boundary_pos,
                                                end_pt_heatmap_x_center, up_boundary_pos]
        else:
            cur_wall_line.start_point.set_boundary(up_boundary_pos, 0, up_flag=True)
            cur_wall_line.start_point.set_boundary(down_boundary_pos, 0, up_flag=False)

            cur_wall_line.end_point.set_boundary(up_boundary_pos, 2, up_flag=True)
            cur_wall_line.end_point.set_boundary(down_boundary_pos, 2, up_flag=False)

            cur_wall_line.boundary_range_box = [down_boundary_pos, start_pt_heatmap_y_center,
                                                up_boundary_pos, end_pt_heatmap_y_center]
        return True

    def _pre_process_img(self):
        if self.image_gray_data is None:
            return None

        gray_threshold = self._calc_walls_color_threshold()

        print("Color Threshold: {0}".format(gray_threshold))

        ret, binary = cv2.threshold(self.image_gray_data, gray_threshold, 255, cv2.THRESH_BINARY)

        cv2.imwrite(r"D:\binary.png", binary)

        self.image_gray_data = binary

    def _calc_wall_color_threshold(self, cur_wall_line):
        start_pt = cur_wall_line.start_point
        end_pt = cur_wall_line.end_point

        wall_length = cur_wall_line.get_wall_length()
        opening_length = 0
        for cur_opening in cur_wall_line.openings:
            opening_length += cur_opening.get_wall_length()
        target_wall_length = wall_length - opening_length
        # 如果墙上的门/窗过大，利用这种方式来判断厚度，会出现太大的问题。
        # 暂时不做处理。比较小的wall判断容易出现问题。
        if target_wall_length < 15 or float(target_wall_length) / wall_length < 0.2:
            return -1, -1

        wall_line_dim = cur_wall_line.line_dim()
        wall_section_points = [start_pt, end_pt]
        for cur_opening in cur_wall_line.openings:
            wall_section_points.append(cur_opening.start_point)
            wall_section_points.append(cur_opening.end_point)
        if wall_line_dim == 0:
            wall_section_points = sorted(wall_section_points, key=lambda obj: obj.x)
        else:
            wall_section_points = sorted(wall_section_points, key=lambda obj: obj.y)

        wall_sections = []
        for i in range(0, len(wall_section_points), 2):
            wall_sections.append([wall_section_points[i], wall_section_points[i + 1]])

        start_pt_heatmap_x_center, start_pt_heatmap_y_center = start_pt.get_heat_map_centroid()
        end_pt_heatmap_x_center, end_pt_heatmap_y_center = end_pt.get_heat_map_centroid()

        if wall_line_dim == 0:
            # 如果平行X轴，计算平行方向上的厚度。找到变化的那条横线。
            center_pos = int(0.5 * (start_pt_heatmap_y_center + end_pt_heatmap_y_center))
        else:
            # 如果平行Y轴，计算竖直方向上的厚度。找到变化的那条竖线。
            center_pos = int(0.5 * (start_pt_heatmap_x_center + end_pt_heatmap_x_center))

        up_boundary_pos, up_dark_to_light_flag, up_boundary_gray_value = self._find_boundary_line(center_pos,
                                                                                                  wall_sections,
                                                                                                  wall_line_dim)
        down_boundary_pos, down_dark_to_light_flag, down_boundary_gray_value = self._find_boundary_line(center_pos,
                                                                                                        wall_sections,
                                                                                                        wall_line_dim,
                                                                                                        increase_flag=False)
        up_gray_value = up_boundary_gray_value if up_boundary_pos > 0 else -1
        down_gray_value = down_boundary_gray_value if down_boundary_pos > 0 else -1
        return up_gray_value, down_gray_value

    def _calc_walls_color_threshold(self):
        wall_line_color_threshold = []
        for cur_wall_line in self.all_wall_lines:
            up_color_threshold, down_color_threshold = self._calc_wall_color_threshold(cur_wall_line)
            if up_color_threshold > 0:
                wall_line_color_threshold.append(up_color_threshold)
            if down_color_threshold > 0:
                wall_line_color_threshold.append(down_color_threshold)
        threshold = 1.1 * np.median(np.array(wall_line_color_threshold))
        max_value = np.max(np.array(wall_line_color_threshold))
        if threshold > max_value:
            threshold = max_value
        return int(threshold)

    def _calc_walls_thickness(self):
        avg_outer_wall_thickness = 8
        if len(self.outer_walls_thickness) > 0:
            thickness = np.array(self.outer_walls_thickness)
            avg_outer_wall_thickness = np.mean(thickness)
            filter_thickness = thickness - avg_outer_wall_thickness
            mask_flag = filter_thickness < 3
            thickness = thickness[mask_flag]
            avg_outer_wall_thickness = np.mean(thickness)

        avg_inner_wall_thickness = 8
        if len(self.inner_walls_thickness) > 0:
            thickness = np.array(self.inner_walls_thickness)
            avg_inner_wall_thickness = np.mean(thickness)
            filter_thickness = thickness - avg_inner_wall_thickness
            mask_flag = filter_thickness < 3
            thickness = thickness[mask_flag]
            avg_inner_wall_thickness = np.mean(thickness)

        if avg_outer_wall_thickness < avg_inner_wall_thickness:
            avg_outer_wall_thickness = avg_inner_wall_thickness
        if avg_inner_wall_thickness == 0:
            avg_inner_wall_thickness = avg_outer_wall_thickness

        if abs(avg_inner_wall_thickness - avg_outer_wall_thickness) <= 1:
            avg_inner_wall_thickness = avg_outer_wall_thickness

        return avg_inner_wall_thickness, avg_outer_wall_thickness

    def _get_align_wall_sections_by_point(self, start_point, direction):
        target_wall_sections = []
        for cur_wall in start_point.wall_lines:
            if cur_wall.visit_flag or (cur_wall.line_dim() != direction):
                continue

            target_wall_sections.append(cur_wall)
            cur_wall.visit_flag = True

            next_point = cur_wall.start_point if cur_wall.start_point != start_point else cur_wall.end_point

            res_wall_sections = self._get_align_wall_sections_by_point(next_point, direction)
            target_wall_sections.extend(res_wall_sections)
        return target_wall_sections

    def _fine_tune_align_wall_sections_by_point(self, point, direction):
        wall_sections = self._get_align_wall_sections_by_point(point, direction)
        if len(wall_sections) <= 1:
            return

        # RangeBox的中心
        avg_value = 0.0
        for cur_wall in wall_sections:
            avg_value += 0.5 * (cur_wall.boundary_range_box[0] + cur_wall.boundary_range_box[2]) if direction == 1 else \
                0.5 * (cur_wall.boundary_range_box[1] + cur_wall.boundary_range_box[3])
        avg_value /= len(wall_sections)

        # fine tune
        for cur_wall in wall_sections:
            thickness = cur_wall.get_wall_thickness()
            if direction == 1:
                cur_wall.boundary_range_box[0] = avg_value - 0.5 * thickness
                cur_wall.boundary_range_box[2] = avg_value + 0.5 * thickness
            else:
                cur_wall.boundary_range_box[1] = avg_value - 0.5 * thickness
                cur_wall.boundary_range_box[3] = avg_value + 0.5 * thickness

    # fine tune the opening position with heatmap position.
    def _fine_tune_opening_position(self, opening, host_wall):
        opening_start_pt, opening_end_pt = opening.start_point, opening.end_point
        start_pt_heatmap_x_center, start_pt_heatmap_y_center = opening_start_pt.get_heat_map_centroid()
        end_pt_heatmap_x_center, end_pt_heatmap_y_center = opening_end_pt.get_heat_map_centroid()

        wall_line_dim = host_wall.line_dim()

        # 这里注意wall_line_dim == 0的时候， 去获得wall的Y边界
        wall_sections = [
            [int(host_wall.boundary_range_box[1]), int(host_wall.boundary_range_box[3])]] if wall_line_dim == 0 else \
            [[int(host_wall.boundary_range_box[0]), int(host_wall.boundary_range_box[2])]]
        center_pos = start_pt_heatmap_x_center if wall_line_dim == 0 else start_pt_heatmap_y_center
        scan_number = 3
        boundary_pos, up_dark_to_light_flag, up_boundary_gray_value = self._find_boundary_line(center_pos,
                                                                                               wall_sections,
                                                                                               wall_line_dim,
                                                                                               increase_flag=False,
                                                                                               max_scan_number=scan_number)
        if boundary_pos > 0:
            if wall_line_dim == 0:
                opening.start_point.x = boundary_pos
            else:
                opening.start_point.y = boundary_pos
        else:
            boundary_pos, up_dark_to_light_flag, up_boundary_gray_value = self._find_boundary_line(center_pos,
                                                                                                   wall_sections,
                                                                                                   wall_line_dim,
                                                                                                   max_scan_number=scan_number)
            # find the boundary position.
            if boundary_pos > 0:
                if wall_line_dim == 0:
                    opening.start_point.x = boundary_pos
                else:
                    opening.start_point.y = boundary_pos

        # 从大到小去扫描.
        # center_pos -= int(0.3 * scan_number)
        center_pos = end_pt_heatmap_x_center if wall_line_dim == 0 else end_pt_heatmap_y_center
        boundary_pos, up_dark_to_light_flag, up_boundary_gray_value = self._find_boundary_line(center_pos,
                                                                                               wall_sections,
                                                                                               wall_line_dim,
                                                                                               increase_flag=False,
                                                                                               max_scan_number=scan_number)
        if boundary_pos > 0:
            if wall_line_dim == 0:
                opening.end_point.x = boundary_pos
            else:
                opening.end_point.y = boundary_pos

            boundary_pos, up_dark_to_light_flag, up_boundary_gray_value = self._find_boundary_line(center_pos,
                                                                                                   wall_sections,
                                                                                                   wall_line_dim,
                                                                                                   increase_flag=True,
                                                                                                   max_scan_number=scan_number)
            # find the boundary position.
            if boundary_pos > 0:
                if wall_line_dim == 0:
                    opening.end_point.x = boundary_pos
                else:
                    opening.end_point.y = boundary_pos

    # 对于某些墙来说，self._predict_outer_walls() 有可能会判断错误。
    # 根据一些特性来把inner wall标记成outer wall.
    def _update_walls_type(self, inner_wall_thickness, outer_wall_thickness):
        for cur_wall in self.all_wall_lines:
            if cur_wall.is_outer_wall:
                continue
            # 这个时候，wall thickness还不是特别准确。所以有些wall thickness 为-1.
            cur_wall_thickness = cur_wall.get_wall_thickness()
            if cur_wall_thickness < 0:
                continue
            # 如果wall thickness 比平均的inner wall thickness还小，
            # cur_wall_thickness - inner_wall_thickness < 0, 那么还是inner wall.
            if cur_wall_thickness - inner_wall_thickness < 0.0:
                continue

            # 比较两者的差距，偏向那边就是属于哪种类型的墙。
            if cur_wall_thickness - inner_wall_thickness > abs(cur_wall_thickness - outer_wall_thickness):
                all_connect_walls = []
                all_connect_walls.extend(cur_wall.start_point.wall_lines)
                all_connect_walls.extend(cur_wall.end_point.wall_lines)
                is_outer_wall = True
                for tmp_wall in all_connect_walls:
                    if tmp_wall.id == cur_wall.id:
                        continue
                    if not tmp_wall.is_outer_wall:
                        is_outer_wall = False
                        break
                if is_outer_wall:
                    cur_wall.is_outer_wall = True

    def _point_distance(self, x1, y1, x2, y2):
        return max(abs(x1 - x2), abs(y1 - y2))
        # return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    def align(self):

        self._predict_outer_walls()

        # 1. 第一次align wall line. 处理完了之后，有可能会存在一些没有align的wall，比如窗户比较大，如果强行去处理的话
        unalign_wall_lines = []
        for cur_wall_line in self.all_wall_lines:
            if not self._align_wall_line(cur_wall_line):
                unalign_wall_lines.append(cur_wall_line)

        inner_wall_thickness, outer_wall_thickness = self._calc_walls_thickness()

        self._update_walls_type(inner_wall_thickness, outer_wall_thickness)

        # 2.补救一些没有完全设置好的boundary value.
        for cur_unaligned_wall in unalign_wall_lines:
            line_dim = cur_unaligned_wall.line_dim()
            if line_dim == 0:
                if cur_unaligned_wall.boundary_range_box[0] < 0:
                    cur_unaligned_wall.boundary_range_box[0] = cur_unaligned_wall.start_point.x
                if cur_unaligned_wall.boundary_range_box[2] < 0:
                    cur_unaligned_wall.boundary_range_box[2] = cur_unaligned_wall.end_point.x
            elif line_dim == 1:
                if cur_unaligned_wall.boundary_range_box[1] < 0:
                    cur_unaligned_wall.boundary_range_box[1] = cur_unaligned_wall.start_point.y
                if cur_unaligned_wall.boundary_range_box[3] < 0:
                    cur_unaligned_wall.boundary_range_box[3] = cur_unaligned_wall.end_point.y
            elif line_dim == -1:
                # 接近无用功
                if cur_unaligned_wall.boundary_range_box[0] < 0:
                    cur_unaligned_wall.boundary_range_box[0] = cur_unaligned_wall.start_point.x
                if cur_unaligned_wall.boundary_range_box[2] < 0:
                    cur_unaligned_wall.boundary_range_box[2] = cur_unaligned_wall.end_point.x
                if cur_unaligned_wall.boundary_range_box[1] < 0:
                    cur_unaligned_wall.boundary_range_box[1] = cur_unaligned_wall.start_point.y
                if cur_unaligned_wall.boundary_range_box[3] < 0:
                    cur_unaligned_wall.boundary_range_box[3] = cur_unaligned_wall.end_point.y

        # 更新这些unalign的墙厚。
        new_unalign_wall_lines = unalign_wall_lines
        for i in range(5):
            if len(new_unalign_wall_lines) == 0:
                break
            unalign_wall_lines = new_unalign_wall_lines
            new_unalign_wall_lines = []
            for cur_unaligned_wall in unalign_wall_lines:
                direction = cur_unaligned_wall.line_dim()
                if direction == 0:
                    if cur_unaligned_wall.boundary_range_box[1] < 0 or cur_unaligned_wall.boundary_range_box[3] < 0:
                        # y值未定，就是水平方向上的厚度没定
                        if cur_unaligned_wall.set_thickness_boundary_by_connect_wall(inner_wall_thickness,
                                                                                     outer_wall_thickness):
                            pass
                elif direction == 1:
                    if cur_unaligned_wall.boundary_range_box[0] < 0 or cur_unaligned_wall.boundary_range_box[2] < 0:
                        # 找到start point and end point， 看它们是否被设置。
                        # x boundary值未定，也就是竖直方向上的厚度没定。
                        cur_unaligned_wall.set_thickness_boundary_by_connect_wall(inner_wall_thickness,
                                                                                  outer_wall_thickness)

                if not cur_unaligned_wall.is_aligned():
                    new_unalign_wall_lines.append(cur_unaligned_wall)

        # fine tune thickness.
        for cur_wall_line in self.all_wall_lines:
            cur_wall_line.fine_tune_wall_thickness(inner_wall_thickness, outer_wall_thickness)
            cur_wall_line.visit_flag = False

        # fine tune wall sections. 同一条直线的墙中线应该在同一直线上。
        # 通过cur_wall_point去查找 wall sections.
        for cur_wall_point in self.all_wall_points:
            self._fine_tune_align_wall_sections_by_point(cur_wall_point, 0)
            self._fine_tune_align_wall_sections_by_point(cur_wall_point, 1)
        #
        # # 设置墙Junction点的坐标
        for cur_wall_line in self.all_wall_lines:
            direction = cur_wall_line.line_dim()
            if direction == 0:
                center = 0.5 * (cur_wall_line.boundary_range_box[1] + cur_wall_line.boundary_range_box[3])
                if np.abs(center-cur_wall_line.start_point.y)<20:
                    cur_wall_line.start_point.y = center
                    cur_wall_line.end_point.y = center
            elif direction == 1:
                center = 0.5 * (cur_wall_line.boundary_range_box[0] + cur_wall_line.boundary_range_box[2])
                if np.abs(center - cur_wall_line.start_point.x) < 20:
                    cur_wall_line.start_point.x = center
                    cur_wall_line.end_point.x = center
            elif direction == -1:
                x1 = cur_wall_line.start_point.x
                y1 = cur_wall_line.start_point.y
                x2 = cur_wall_line.end_point.x
                y2 = cur_wall_line.end_point.y
                for point in self.all_wall_points:
                    if x1 != point.x or y1 != point.y:
                        start_distance = self._point_distance(x1, y1, point.x, point.y)
                        if start_distance <= 15:
                            cur_wall_line.start_point.x = point.x
                            cur_wall_line.start_point.y = point.y
                    if x2 != point.x or y2 != point.y:
                        end_distance = self._point_distance(x2, y2, point.x, point.y)
                        if end_distance <= 15:
                            cur_wall_line.end_point.x = point.x
                            cur_wall_line.end_point.y = point.y
