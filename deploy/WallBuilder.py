# encoding:utf-8
import json
import time
from WallAlignment import *
from debug_utils import *
from floorplan_data_dump import *

import Primitive


class Builder(object):
    def __init__(self, options):

        self.options = options

        Primitive.global_id = 0

        self.heat_map_wall_threshold = 0.4
        self.heat_map_wall_connect_threshold = 0.8
        self.heat_map_opening_threshold = 0.4

        self.same_point_distance_threshold = 10

        self.floor_plan_img_data = None
        self.floor_plan_img_data_resize = None
        self.floor_plan_img_height = 0
        self.floor_plan_img_width = 0
        self.gap = 8

        # windows or door point.
        self.all_opening_points = []
        self.all_opening_lines = []

        self.all_wall_lines = []
        self.all_wall_points = []

        self.all_door_points = []
        self.all_door_lines = []

        self.tmp_pixel_list = []

        super(Builder, self).__init__()

    def transfrom_image_data_base(self, image, gray_image=True):
        return self.transform_image_data(image, gray_image=gray_image)

    def transform_image_data(self, image, gray_image=True):
        max_size = max(self.options.width, self.options.height)
        # X, Y
        image_sizes = np.array(image.shape[:2]).astype(np.float32)
        self.ratio = image_sizes.max() / float(max_size)
        image_sizes = (image_sizes / image_sizes.max() * max_size).astype(np.int32)
        resized_image_data = cv2.resize(image, (image_sizes[1], image_sizes[0]))

        resized_image_height = image_sizes[0]
        resized_image_width = image_sizes[1]

        # 放置在图像中间. x>y
        if image_sizes[1] > image_sizes[0]:
            offset_x = 0
            offset_y = int(0.5 * (self.options.height - image_sizes[0]))
        else:
            offset_x = int(0.5 * (self.options.width - image_sizes[1]))
            offset_y = 0

        if gray_image:
            full_image = np.full((self.options.height, self.options.width), fill_value=0.0)
            full_image[offset_y:offset_y + resized_image_height,
            offset_x:offset_x + resized_image_width] = resized_image_data
        else:
            full_image = np.full((self.options.height, self.options.width, 3), fill_value=0.0)
            full_image[offset_y:offset_y + resized_image_height, offset_x:offset_x + resized_image_width,
            :] = resized_image_data

        self.offset_x = offset_x
        self.offset_y = offset_y
        return full_image

    def _transform_point(self, point_obj):
        point_obj.y = int((point_obj.y - self.offset_y) * self.ratio)
        point_obj.x = int((point_obj.x - self.offset_x) * self.ratio)

    def transform_back_all_points(self):
        for point_obj in self.all_opening_points:
            self._transform_point(point_obj)
        for point_obj in self.all_wall_points:
            self._transform_point(point_obj)
        for point_obj in self.all_door_points:
            self._transform_point(point_obj)
        # transform wall lines' range box.
        for cur_wall in self.all_wall_lines:
            direction = cur_wall.line_dim()
            thickness = cur_wall.get_wall_thickness() * self.ratio
            cur_wall.boundary_range_box[0] = int((cur_wall.boundary_range_box[0] - self.offset_x) * self.ratio)
            cur_wall.boundary_range_box[1] = int((cur_wall.boundary_range_box[1] - self.offset_y) * self.ratio)
            cur_wall.boundary_range_box[2] = int((cur_wall.boundary_range_box[2] - self.offset_x) * self.ratio)
            cur_wall.boundary_range_box[3] = int((cur_wall.boundary_range_box[3] - self.offset_y) * self.ratio)
            #
            if direction == 0:
                cur_wall.boundary_range_box[2] = int((cur_wall.boundary_range_box[2] - self.offset_x) * self.ratio)
                cur_wall.boundary_range_box[3] = int(cur_wall.boundary_range_box[1] + thickness + 0.5)
            elif direction == 1:
                cur_wall.boundary_range_box[2] = int(cur_wall.boundary_range_box[0] + thickness + 0.5)
                cur_wall.boundary_range_box[3] = int((cur_wall.boundary_range_box[3] - self.offset_y) * self.ratio)
            elif direction == -1:
                continue

        # 设置点的坐标
        for cur_wall_line in self.all_wall_lines:
            cur_wall_line.fine_tune_openings_coordinate()

    def build_floorplan_json(self, floor_plan_img_data, heat_map_img_list, measuring_scale_ratio=-1.0):
        try:
            self._build_floorplan_primitives(floor_plan_img_data, heat_map_img_list)

            start_time = time.time()
            json_res = self.dump_floorplan_data(measuring_scale_ratio)
            print("dump room design json (including convert json) cost {0}".format(time.time() - start_time))
            print("WallBuilder.options.debugFlag = {0}".format(self.options.debugFlag))
            if self.options.debugFlag == 1:
                self.dump_debug_data(json_res)

            return json_res
        except Exception as err:
            print(err)

    def draw_points(self, wallPointsTemp, line_width=5, file_name="Node", background_img_data=None,
                    rgb_color=[0, 0, 255]):
        img_data = background_img_data
        line_color = np.random.rand(3) * 255
        line_color[0] = rgb_color[0]
        line_color[1] = rgb_color[1]
        line_color[2] = rgb_color[2]

        floor_plan_img_height = img_data.shape[0]
        floor_plan_img_width = img_data.shape[1]

        for cur_wall_point in wallPointsTemp:
            x = cur_wall_point.x
            y = cur_wall_point.y
            img_data[max(y - line_width, 0):min(y + line_width, floor_plan_img_height - 1),
            max(x - line_width, 0):min(x + line_width, floor_plan_img_width - 1)] = line_color
            # cv2.putText(img_data, str(corner), (x, y),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0, 255, 0))
        cv2.imwrite(file_name, img_data)

    def _build_floorplan_primitives(self, floor_plan_img_data, heat_map_img_list):
        try:
            self.floor_plan_img_data = floor_plan_img_data
            self.floor_plan_img_height = floor_plan_img_data.shape[0]
            self.floor_plan_img_width = floor_plan_img_data.shape[1]
            resize_floor_plan_img_data = self.floor_plan_img_data.copy()
            resize_floor_plan_img_data = self.transfrom_image_data_base(resize_floor_plan_img_data, gray_image=False)
            self.floor_plan_img_data_resize = resize_floor_plan_img_data

            # 1. For junction heatmap.
            all_wall_points = []
            for i in range(13):
                cur_heat_map_img = heat_map_img_list[i]
                if cur_heat_map_img is None:
                    continue
                cur_category = int(i / 4)
                cur_sub_category = i % 4

                # transform image data to 512 * 512.
                cur_heat_map_img = self.transform_image_data(cur_heat_map_img)

                junctions_list = self._extract_junctions(cur_heat_map_img, 300, cur_category, cur_sub_category,
                                                         heat_map_threshold=self.heat_map_wall_threshold)
                all_wall_points.extend(junctions_list)

            # for opening points.
            all_opening_points = []
            for i in range(8):
                cur_heat_map_img = heat_map_img_list[13 + i]
                if cur_heat_map_img is None:
                    continue

                cur_category = int(i / 4)
                cur_sub_category = i % 4
                cur_heat_map_img = self.transform_image_data(cur_heat_map_img)
                all_opening_points.extend(self._extract_junctions(cur_heat_map_img, 100, cur_category, cur_sub_category,
                                                                  self.heat_map_wall_threshold))

            all_door_points = []
            for i in range(28):
                cur_heat_map_img = heat_map_img_list[21 + i]
                if cur_heat_map_img is None:
                    continue

                if i < 24:
                    cur_category = int(i / 8)
                    cur_sub_category = i % 8
                else:
                    # 第4种门，sub category只有4种
                    cur_category = 3
                    cur_sub_category = (i - 28) % 4

                cur_heat_map_img = self.transform_image_data(cur_heat_map_img)
                all_door_points.extend(self._extract_junctions(cur_heat_map_img, 100, cur_category, cur_sub_category,
                                                               self.heat_map_wall_threshold, junction_class=2))

            # 去除水平｜垂直类型的重复点，斜墙类别的点暂不做处理
            self.all_wall_points = self._remove_duplicate_points(all_wall_points, is_wall=True)
            self.all_opening_points = self._remove_duplicate_points(all_opening_points)
            self.all_door_points = self._remove_duplicate_points(all_door_points)

            # self.all_wall_points里有斜墙对应的Point
            self.all_wall_lines = self._build_wall_lines(self.all_wall_points)
            self.all_opening_lines = self._build_opening_lines(self.all_opening_points)
            self.all_door_lines = self._build_opening_lines(self.all_door_points, building_door=True)

            # 考虑完备性的问题。（是点的话就必须相连，往同方向最近的点靠拢）
            self._repair_points_completeness()

            self._fit_wall_openings()  # 计算墙和门/窗的关系，判断门/窗是否在某一墙上
            # self.draw_lines(all_wall_lines=self.all_wall_lines,
            #                 file_name=os.path.join(self.options.res_folder_path, "debugLins.png"),
            #                 background_img_data=resize_floor_plan_img_data)

            self._remove_invalid_inclined_wall()  # 斜墙剪枝逻辑

            self.draw_lines(all_wall_lines=self.all_wall_lines,
                            file_name=os.path.join(self.options.res_folder_path, "debugLins_remove_invalid_inclined.png"),
                            background_img_data=resize_floor_plan_img_data)

            # 查找并剔除孤立的门/窗点，即剔除没有跟wall有关系的门/窗 Point/Line
            self._remove_isolate_openings()

            alignment_obj = WallAlignment(resize_floor_plan_img_data,
                                          self.all_wall_lines,
                                          self.all_wall_points,
                                          self.all_opening_points,
                                          self.all_door_points)

            alignment_obj.align()

            # transform the points to original image scope.
            self.transform_back_all_points()

            # 当都准备好的时候，后续操作。
            self.post_process()

        except Exception as err:
            print(err)

    def draw_lines(self, all_wall_lines, line_width=2, file_name="Node", background_img_data=None):
        try:
            image = background_img_data

            floor_plan_img_height = image.shape[0]
            floor_plan_img_width = image.shape[1]

            i = 0
            for wall_line in all_wall_lines:
                line_color = np.random.rand(3) * 255

                point_1 = wall_line.start_point
                point_2 = wall_line.end_point

                i += 1

                line_dim = wall_line.line_dim()

                fixedValue = int(round((point_1.y + point_2.y) / 2)) if line_dim == 0 else int(
                    round((point_1.x + point_2.x) / 2))
                minValue = int(round(min(point_1.x, point_2.x))) if line_dim == 0 else int(
                    round(min(point_1.y, point_2.y)))
                maxValue = int(round(max(point_1.x, point_2.x))) if line_dim == 0 else int(
                    round(max(point_1.y, point_2.y)))

                if line_dim == 0:
                    image[max(fixedValue - line_width, 0):min(fixedValue + line_width, floor_plan_img_height),
                    minValue:maxValue + 1, :] = line_color
                elif line_dim == 1:
                    image[minValue:maxValue + 1,
                    max(fixedValue - line_width, 0):min(fixedValue + line_width, floor_plan_img_width), :] = line_color
                elif line_dim == -1:
                    print((point_1.x, point_1.y), (point_2.x, point_2.y))
                    cv2.line(image, (point_1.x, point_1.y), (point_2.x, point_2.y), (255, 0, 0), 2)
            cv2.imwrite(file_name, image)

        except Exception as err:
            print(err)

    def dump_floorplan_data(self, measuring_scale_ratio=-1.0):
        floorplan_data_dump_obj = FloorplanDataDump(self)
        json_res = floorplan_data_dump_obj.dump_room_design_json(measuring_scale_ratio)
        return json_res

    # 在本地，需要保存一些测试结果
    def dump_debug_data(self, json_res):
        # 保存数据
        dump_obj = WallBuilderDataDump(self)
        dump_obj.dump_data()

        # save the json results.
        json_file_path = os.path.join(self.options.res_folder_path, "Floorplan3D.json")
        with open(json_file_path, "w") as outfile:
            json.dump(json_res, outfile, indent=4)

    # 后处理
    def post_process(self):
        return True

    def _create_wall_line(self, start_point, end_point):
        cur_wall_line = WallLine(start_point, end_point)
        start_point.wall_lines.append(cur_wall_line)
        end_point.wall_lines.append(cur_wall_line)
        start_point.add_connect_point(end_point)
        end_point.add_connect_point(start_point)
        return cur_wall_line

    def _repair_connect_points_completeness(self, next_point, tmp_wall_point, next_index, index, target_direction):
        repaired_flag = False
        # 如果next_point是完备的，那么next_point不应该和tmp_wall_point相连。
        if len(next_point.get_noncomplete_direction()) == 0:
            # 关于next_point, 由于存在candidate_categories, 有可能把它的category判断错误。
            if len(next_point.candidate_categories) != 0:
                # 临时保留type category.
                point_type_category = next_point.type_category
                point_type_sub_category = next_point.type_sub_category

                updated_category_flag = False
                point_feasible_dir = next_point.get_feasible_direction()
                for cur_category in next_point.candidate_categories:
                    cur_type_category = cur_category[0]
                    cur_type_sub_category = cur_category[1]

                    # 由于next_point是完备的，如果要添加新的连接，那么type_category是需要升级的。
                    if cur_type_category < point_type_category:
                        continue

                    # 临时设置type_category 以及 type_sub_category.
                    next_point.type_category = cur_type_category
                    next_point.type_sub_category = cur_type_sub_category

                    cur_feasible_dir = next_point.get_feasible_direction()

                    if len(list(set(point_feasible_dir).difference(set(cur_feasible_dir)))) != 0:
                        continue
                    else:
                        # 两者是否可以相连。
                        start_point = next_point if next_index < index else tmp_wall_point
                        end_point = tmp_wall_point if next_index < index else next_point

                        if self._is_feasible_wall_line_points(start_point, end_point, target_direction):
                            cur_wall_line = self._create_wall_line(start_point, end_point)
                            self.all_wall_lines.append(cur_wall_line)

                            updated_category_flag = True
                            repaired_flag = True
                            break

                # no need to update the category, roll back it.
                if not updated_category_flag:
                    next_point.type_category = point_type_category
                    next_point.type_sub_category = point_type_sub_category
        else:
            # 两者是否可以相连。
            start_point = next_point if next_index < index else tmp_wall_point
            end_point = tmp_wall_point if next_index < index else next_point

            if self._is_feasible_wall_line_points(start_point, end_point, target_direction):
                cur_wall_line = self._create_wall_line(start_point, end_point)
                self.all_wall_lines.append(cur_wall_line)
                repaired_flag = True
            else:
                # 两者并不能直接相连，看是否是类型判断错了。这个是特殊处理的情况。
                if end_point.type_category == 1 and end_point.type_sub_category == 2:
                    # 这个经常弄混淆
                    end_point.type_sub_category = 1
                    if self._is_feasible_wall_line_points(start_point, end_point, target_direction):
                        cur_wall_line = self._create_wall_line(start_point, end_point)
                        self.all_wall_lines.append(cur_wall_line)
                        repaired_flag = True
                    else:
                        end_point.type_sub_category = 2
                elif start_point.type_category == 1 and start_point.type_sub_category == 0:
                    # 这个经常弄混淆
                    start_point.type_sub_category = 3
                    if self._is_feasible_wall_line_points(start_point, end_point, target_direction):
                        cur_wall_line = self._create_wall_line(start_point, end_point)
                        self.all_wall_lines.append(cur_wall_line)
                        repaired_flag = True
                    else:
                        end_point.type_sub_category = 0  # 把sub_category设置为原来value，因为没有做任何改动。
        return repaired_flag

    def _repair_points_completeness(self):
        for tmp_wall_point in self.all_wall_points:
            # 某个方向上没有完备
            un_fit_directions = tmp_wall_point.get_noncomplete_direction()
            for tmp_un_fit_dir in un_fit_directions:
                target_direction = 0 if tmp_un_fit_dir == 1 or tmp_un_fit_dir == 3 else 1
                same_dir_points = self._find_points_on_same_direction(self.all_wall_points, tmp_wall_point,
                                                                      target_direction)

                contained_point_flag = False
                for tmp_point in same_dir_points:
                    if tmp_point.id == tmp_wall_point.id:
                        contained_point_flag = True
                        break
                if contained_point_flag == False:
                    same_dir_points.append(tmp_wall_point)

                if target_direction == 0:
                    all_sorted_points = sorted(same_dir_points, key=lambda obj: obj.x)
                else:
                    all_sorted_points = sorted(same_dir_points, key=lambda obj: obj.y)

                index = all_sorted_points.index(tmp_wall_point)
                if tmp_un_fit_dir == 3 or tmp_un_fit_dir == 0:
                    next_index = index + 1
                else:
                    next_index = index - 1
                if next_index < 0 or next_index > len(all_sorted_points) - 1:
                    continue
                next_point = all_sorted_points[next_index]

                # 是否被修复?
                self._repair_connect_points_completeness(next_point, tmp_wall_point, next_index, index,
                                                         target_direction)

    def _remove_isolate_openings(self):
        isolated_openings = []
        isolated_opening_points = []
        for cur_opening in self.all_opening_lines:
            if len(cur_opening.host_walls) != 0:
                continue
            isolated_openings.append(cur_opening)
            isolated_opening_points.append(cur_opening.start_point)
            isolated_opening_points.append(cur_opening.end_point)

        self.all_opening_lines = list(set(self.all_opening_lines).difference(set(isolated_openings)))
        self.all_opening_points = list(set(self.all_opening_points).difference(set(isolated_opening_points)))

        isolated_openings = []
        isolated_opening_points = []
        for cur_door in self.all_door_lines:
            if len(cur_door.host_walls) != 0:
                continue
            isolated_openings.append(cur_door)
            isolated_opening_points.append(cur_door.start_point)
            isolated_opening_points.append(cur_door.end_point)
        self.all_door_lines = list(set(self.all_door_lines).difference(set(isolated_openings)))
        self.all_door_points = list(set(self.all_door_points).difference(set(isolated_opening_points)))

    def _remove_invalid_inclined_wall(self):
        # need_remove_lines = []
        not_inclined_walls = []
        inclined_wall_without_opening = []  # 不带门/窗的斜墙
        inclined_wall_with_opening = []  # 带门/窗的斜墙
        for index in range(len(self.all_wall_lines)):
            line = self.all_wall_lines[index]
            if line.is_inclined_wall and line.get_wall_length() > 0:
                if len(line.openings) > 0:
                    inclined_wall_with_opening.append(line)
                else:
                    inclined_wall_without_opening.append(line)
            elif not line.is_inclined_wall:
                not_inclined_walls.append(line)
        inclined_wall_with_opening = sorted(inclined_wall_with_opening, key=lambda obj: obj.start_point.x)
        inclined_wall_without_opening = sorted(inclined_wall_without_opening, key=lambda obj: obj.start_point.x)

        inclined_wall_without_opening = self._remove_duplicate_inclined_wall(inclined_wall_without_opening)
        inclined_wall_with_opening = self._remove_duplicate_inclined_wall(inclined_wall_with_opening)

        inclined_wall_without_opening = self._remove_redundance_inclined_wall(inclined_wall_without_opening)

        not_inclined_walls.extend(inclined_wall_without_opening)
        not_inclined_walls.extend(inclined_wall_with_opening)
        self.all_wall_lines = not_inclined_walls

    def _remove_duplicate_inclined_wall(self, inclined_walls):
        need_remove_index = []
        gap = 20
        for i in range(len(inclined_walls)):
            for j in range(i + 1, len(inclined_walls)):
                line_1 = inclined_walls[i]
                line_2 = inclined_walls[j]
                if i != j:
                    start_start_point_distance = self._get_point_distance(line_1.start_point, line_2.start_point)
                    start_end_point_distance = self._get_point_distance(line_1.start_point, line_2.end_point)
                    end_end_point_distance = self._get_point_distance(line_1.end_point, line_2.end_point)
                    end_start_point_distance = self._get_point_distance(line_1.end_point, line_2.start_point)
                    # 剔除重复的墙，包含来回墙
                    if (start_start_point_distance < gap and end_end_point_distance < gap) or (
                            start_end_point_distance < gap and end_start_point_distance < gap):
                        need_remove_index.append(j)
                        continue

                    if line_1.start_point.x == line_1.start_point.x or line_2.start_point.x == line_2.start_point.x:
                        continue
                    if line_1.start_point.x < line_1.start_point.x:
                        line_1_slpoe, line_1_intercept = self._calculate_line_slope(line_1.start_point.x,
                                                                                    line_1.start_point.y,
                                                                                    line_1.end_point.x, line_1.end_point.y)
                    else:
                        line_1_slpoe, line_1_intercept = self._calculate_line_slope(line_1.end_point.x, line_1.end_point.y,
                                                                                    line_1.start_point.x,
                                                                                    line_1.start_point.y)
                    if line_2.start_point.x < line_2.start_point.x:
                        line_2_slpoe, line_2_intercept = self._calculate_line_slope(line_2.start_point.x,
                                                                                    line_2.start_point.y,
                                                                                    line_2.end_point.x,
                                                                                    line_2.end_point.y)
                    else:
                        line_2_slpoe, line_2_intercept = self._calculate_line_slope(line_2.end_point.x,
                                                                                    line_2.end_point.y,
                                                                                    line_2.start_point.x,
                                                                                    line_2.start_point.y)
                    if np.abs(line_1_slpoe-line_2_slpoe)<1: # and np.abs(line_2_intercept-line_1_intercept)<100:
                        if line_1.get_wall_length() >=line_2.get_wall_length():
                            need_remove_index.append(j)
                        else:
                            need_remove_index.append(i)


        need_remove_index = list(set(need_remove_index))
        need_remove_index.sort(reverse=True)
        for index in need_remove_index:
            inclined_walls.remove(inclined_walls[index])
        return inclined_walls

    def _remove_redundance_inclined_wall(self, inclined_walls):
        need_remove_index = []
        for i in range(len(inclined_walls)):
            cur_wall = inclined_walls[i]
            lr = []
            lg = []
            lb = []
            if cur_wall.start_point.x == cur_wall.end_point.x:
                continue
            elif cur_wall.start_point.x < cur_wall.end_point.x:
                line_piexl = self._calculate_line_piexl(cur_wall.start_point.x, cur_wall.start_point.y,
                                                        cur_wall.end_point.x, cur_wall.end_point.y)
            else:
                line_piexl = self._calculate_line_piexl(cur_wall.end_point.x, cur_wall.end_point.y,
                                                        cur_wall.start_point.x, cur_wall.start_point.y)
            sum_length = len(line_piexl)
            for point_info in line_piexl:
                r, g, b = self.floor_plan_img_data_resize[point_info[0], point_info[1]]
                lr.append(r)
                lg.append(g)
                lb.append(b)
            if sum_length < 1:
                a = 1
            else:
                avg_r = np.mean(lr)
                avg_g = np.mean(lg)
                avg_b = np.mean(lb)

                r1, g1, b1 = self.floor_plan_img_data[cur_wall.start_point.x, cur_wall.start_point.y]
                r2, g2, b2 = self.floor_plan_img_data[cur_wall.end_point.x, cur_wall.end_point.y]
                r3, g3, b3 = self.floor_plan_img_data[cur_wall.start_point.x, cur_wall.start_point.y + 1]
                r3, g3, b3 = self.floor_plan_img_data[cur_wall.end_point.x, cur_wall.end_point.y + 1]
                avg_r_p = np.mean([r1,r2])
                avg_g_p = np.mean([g1,g2])
                avg_b_p = np.mean([b1,b2])

                color_dis = self._colour_distance(avg_r, avg_g, avg_b, avg_r_p, avg_g_p, avg_b_p)
                if color_dis > 300:
                    need_remove_index.append(i)
        need_remove_index = list(set(need_remove_index))
        need_remove_index.sort(reverse=True)
        for index in need_remove_index:
            inclined_walls.remove(inclined_walls[index])
        return inclined_walls

    def _colour_distance(self, R_1, G_1, B_1, R_2, G_2, B_2):
        rmean = (R_1 + R_2) / 2
        R = R_1 - R_2
        G = G_1 - G_2
        B = B_1 - B_2
        return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))

    # 输入两个像素点坐标，返回这两个点所确定直线上所有的像素点坐标(返回值代表这条直线的线宽为3个像素)，但实质上并不是直线，跟像素点上画园一个道理
    # 输入坐标只能是按图片位置上的从左到右，坐标点1（x1， y1）一定要在坐标点2（x2， y2)的左侧，否则无法计算
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
                for t in range(5):
                    line_piexl.append([i, y - t])  # 直线向上平移一个像素
                    line_piexl.append([i, y + t])  # 直线向下平移一个像素
        line_piexl = np.array(line_piexl)
        return line_piexl

    def _calculate_line_slope(self, x1, y1, x2, y2):
        if (x2 - x1) == 0:
            print('斜率不存在')
        a = (y2 - y1) / (x2 - x1)
        b = y1 - x1 * ((y2 - y1) / (x2 - x1))
        return a, b

    def _get_point_distance(self, point_1, point_2):
        return max(abs(point_1.x - point_2.x), abs(point_1.y - point_2.y))

    def _calc_wall_length(self):
        return np.sqrt((self.start_point.x - self.end_point.x) * (self.start_point.x - self.end_point.x) +
                       (self.start_point.y - self.end_point.y) * (self.start_point.y - self.end_point.y))

    def _fit_wall_openings(self):
        for cur_opening in self.all_opening_lines:
            cur_opening.visit_flag = False
        for cur_door in self.all_door_lines:
            cur_door.visit_flag = False

        for cur_wall in self.all_wall_lines:
            for cur_opening in self.all_opening_lines:
                if cur_opening.visit_flag:
                    continue
                if cur_wall.is_opening_on_wall(cur_opening):
                    cur_opening.visit_flag = True
                    cur_wall.openings.append(cur_opening)
                    cur_opening.host_walls.append(cur_wall)

            for cur_door in self.all_door_lines:
                if cur_door.visit_flag:
                    continue
                if cur_wall.is_opening_on_wall(cur_door):
                    cur_door.visit_flag = True
                    cur_wall.openings.append(cur_door)
                    cur_door.host_walls.append(cur_wall)

    def _find_other_opening_line_point(self, all_opening_points, opening_point):
        other_point = None

        # find other opening line point.
        nearest_dist = 100000.0
        for tmp_opening_point in all_opening_points:
            if tmp_opening_point.id == opening_point.id:
                continue

            flag = False
            if (opening_point.type_sub_category == 0 and tmp_opening_point.type_sub_category == 2) or \
                    (opening_point.type_sub_category == 2 and tmp_opening_point.type_sub_category == 0) or \
                    (opening_point.type_sub_category == 1 and tmp_opening_point.type_sub_category == 3) or \
                    (opening_point.type_sub_category == 3 and tmp_opening_point.type_sub_category == 1):
                flag = True

            if not flag:
                continue

            # 平行 X轴
            if np.abs(tmp_opening_point.y - opening_point.y) < 5:
                if np.abs(tmp_opening_point.x - opening_point.x) < nearest_dist:
                    other_point = tmp_opening_point
                    nearest_dist = np.abs(tmp_opening_point.x - opening_point.x)

            elif np.abs(tmp_opening_point.x - opening_point.x) < 5:
                if np.abs(tmp_opening_point.y - opening_point.y) < nearest_dist:
                    other_point = tmp_opening_point
                    nearest_dist = np.abs(tmp_opening_point.y - opening_point.y)
            else:
                continue
        return other_point

    def _build_opening_lines(self, all_opening_points, building_door=False):
        try:
            all_opening_points_copied = all_opening_points.copy()

            i = 0
            all_lines = []
            while len(all_opening_points_copied) > 0:
                cur_opening_point = all_opening_points_copied[0]
                # remove current wall point.
                all_opening_points_copied.remove(cur_opening_point)
                direction = cur_opening_point.get_direction()

                all_points = self._find_points_on_same_direction(all_opening_points_copied, cur_opening_point,
                                                                 direction)
                if len(all_points) == 0:
                    continue

                # 对整体排序
                all_sorted_points = copy.copy(all_points)
                all_sorted_points.append(cur_opening_point)
                if direction == 0:
                    all_sorted_points = sorted(all_sorted_points, key=lambda obj: obj.x)
                else:
                    all_sorted_points = sorted(all_sorted_points, key=lambda obj: obj.y)

                # fix a bug which points are on the same direction, but have different sub categories.
                # 如果这个Opening Point是水平的，但是tmp_point不是水平的，就不需要加到list中去。
                all_align_points = []
                for tmp_point in all_points:
                    if direction == tmp_point.get_direction():
                        all_align_points.append(tmp_point)
                    else:
                        continue

                # 全部的点进行排序
                all_align_points.append(cur_opening_point)
                if direction == 0:
                    sorted_points = sorted(all_align_points, key=lambda obj: obj.x)
                else:
                    sorted_points = sorted(all_align_points, key=lambda obj: obj.y)

                # 从左到右，或者从上到下， start_point 必定为 Line的起始点
                # end_point必定为Line的端点。
                for i in range(0, len(sorted_points) - 1, 1):
                    start_pt = sorted_points[i]
                    # .type_sub_category != 0 and start_pt.type_sub_category != 3:
                    if not start_pt.is_line_start_point():
                        continue
                    end_pt = sorted_points[i + 1]
                    # if end_pt.type_sub_category != 1 and end_pt.type_sub_category != 2:
                    if not end_pt.is_line_end_point():
                        continue

                    # 如果两者之前不是连续的，那么他们是不能生成Opening的。
                    start_pt_index = all_sorted_points.index(start_pt)
                    if (start_pt_index == len(all_sorted_points) - 1) or (
                            end_pt != all_sorted_points[start_pt_index + 1]):
                        continue

                    i += 1
                    if start_pt != cur_opening_point:
                        all_opening_points_copied.remove(start_pt)
                    if end_pt != cur_opening_point:
                        all_opening_points_copied.remove(end_pt)

                    if not building_door:
                        cur_opening_line = OpeningLine(start_pt, end_pt)
                    else:
                        cur_opening_line = DoorLine(start_pt, end_pt)

                    all_lines.append(cur_opening_line)
            return all_lines
        except Exception as err:
            print(err)

    def is_same_points_by_other_condition(self, point, target_point):
        return False

    # remove the duplicate points.
    def _remove_duplicate_points(self, all_wall_points, is_wall=False):
        try:
            new_all_wall_points = []

            while len(all_wall_points) > 0:
                cur_wall_point = all_wall_points[0]
                all_wall_points.remove(cur_wall_point)

                heap_map_scope_length = len(cur_wall_point.heat_map_scope_pixels)
                duplicated_points = []

                saved_point = cur_wall_point
                # if the distance is smaller than 10
                for tmp_point in all_wall_points:
                    dist = np.sqrt((tmp_point.x - cur_wall_point.x) ** 2 + (tmp_point.y - cur_wall_point.y) ** 2)
                    inclined_dim = np.abs(
                        tmp_point.x - cur_wall_point.x) > self.same_point_distance_threshold and np.abs(
                        tmp_point.y - cur_wall_point.y) > self.same_point_distance_threshold
                    # 不对斜墙的点做去重，后面通过斜墙剪枝算法来剔除无效数据
                    if is_wall and inclined_dim and tmp_point.type_category == 0 and cur_wall_point.type_category == 0:
                        continue
                    else:
                        # 同类别小碎点加入重复的点集合
                        if dist < self.same_point_distance_threshold and \
                                (tmp_point.type_category == cur_wall_point.type_category and \
                                 tmp_point.type_sub_category == cur_wall_point.type_sub_category):
                            duplicated_points.append(tmp_point)
                            continue

                        # 某些情况下，一个点会认为是属于两种不同的type。
                        # 0.3属于经验值, 这个值不能设太大。
                        if dist < 0.35 * self.same_point_distance_threshold:
                            duplicated_points.append(tmp_point)
                            continue
                        elif (dist < 0.5 * self.same_point_distance_threshold) and heap_map_scope_length < 10:
                            # heat_map所占据的范围很小，是duplicated points
                            duplicated_points.append(tmp_point)
                            continue
                        elif self.is_same_points_by_other_condition(cur_wall_point, tmp_point):  # 是否是其他类型的相同点。
                            duplicated_points.append(tmp_point)
                            continue

                duplicated_points_type_category = [[saved_point.type_category, saved_point.type_sub_category]]

                # find same points which has the same type category and type sub category.
                if len(duplicated_points) > 0:
                    sum_x = cur_wall_point.x
                    sum_y = cur_wall_point.y

                    # calculate the x, y position.
                    for tmp_point in duplicated_points:
                        sum_x += tmp_point.x
                        sum_y += tmp_point.y
                        # remove the duplicated points from the array.
                        all_wall_points.remove(tmp_point)

                        # 看谁的heatmap够大，可以分析出谁的类型够强。
                        if len(saved_point.heat_map_scope_pixels) < len(tmp_point.heat_map_scope_pixels):
                            saved_point = tmp_point
                        duplicated_points_type_category.append([tmp_point.type_category, tmp_point.type_sub_category])

                    cur_wall_point.x = int(float(sum_x) / (len(duplicated_points) + 1))
                    cur_wall_point.y = int(float(sum_y) / (len(duplicated_points) + 1))
                    cur_wall_point.type_category = saved_point.type_category
                    cur_wall_point.type_sub_category = saved_point.type_sub_category

                    if [cur_wall_point.type_category,
                        cur_wall_point.type_sub_category] in duplicated_points_type_category:
                        duplicated_points_type_category.remove(
                            [cur_wall_point.type_category, cur_wall_point.type_sub_category])
                    cur_wall_point.candidate_categories = duplicated_points_type_category

                new_all_wall_points.append(cur_wall_point)

            return new_all_wall_points
        except Exception as err:
            print(err)
            return new_all_wall_points

    def _remove_alone_wall_lines(self, wall_lines):
        removed_wall_lines = []
        for wall_line in wall_lines:
            if self._is_alone_wall_line(wall_line):
                removed_wall_lines.append(wall_line)

        for wall_line in removed_wall_lines:
            wall_lines.remove(wall_line)
        return wall_lines

    def _is_alone_wall_line(self, wall_line):
        if len(wall_line.start_point.wall_lines) == 1 and len(wall_line.end_point.wall_lines) == 1:
            return True
        else:
            return False

    # 只判断水平｜垂直墙，gap来矫正角度较小的斜墙，斜墙｜弧形墙不做判断
    def _is_line_parallel_direction(self, line_point_1, line_point_2, target_direction):
        if line_point_1.x == line_point_2.x:
            return True if target_direction == 1 else False
        if line_point_1.y == line_point_2.y:
            return True if target_direction == 0 else False

        # for x direction.
        if target_direction == 0:
            if np.abs(line_point_1.y - line_point_2.y) < self.gap:
                return True
            else:
                return False
        else:
            if np.abs(line_point_1.x - line_point_2.x) < self.gap:
                return True
            else:
                return False

    # find the points which direct to the target direction.
    # target_direction = 0: x
    # target_direction = 1: y
    def _find_points_on_same_direction(self, all_wall_points, target_point, target_direction):
        found_points = []

        for i in range(len(all_wall_points)):
            tmp_point = all_wall_points[i]
            # TODO 是否要考虑墙厚的误差？
            if self._is_line_parallel_direction(target_point, tmp_point, target_direction):
                found_points.append(tmp_point)

        return found_points

    # 找到和这个点有关的其他点。其他点和这个点的连线必须平行于x/y轴
    def _build_point_wall_lines(self, all_wall_points, target_point, target_direction):
        same_dir_points = self._find_points_on_same_direction(all_wall_points, target_point, target_direction)
        if len(same_dir_points) == 0:
            return []

        wall_lines = []

        # 加上它本身，然后排序。两两之间形成wall line
        same_dir_points.append(target_point)
        sorted_points = []
        if target_direction == 0:
            sorted_points = sorted(same_dir_points, key=lambda obj: obj.x)
        else:
            sorted_points = sorted(same_dir_points, key=lambda obj: obj.y)

        # self._align_wall_line_points(same_dir_points, target_direction)
        cur_wall_section = []
        for i in range(len(sorted_points) - 1):
            p_1 = sorted_points[i]
            p_2 = sorted_points[i + 1]
            cur_wall_section.append(p_1)
            if not self._is_feasible_wall_line_points(p_1, p_2, target_direction):
                # 判断当前点是否可以和i+2这个点相连，当然i+1和i+2这两个点隔的很近，由于误差原因会导致他们的位置混乱。
                # 出现这种情况是因为两堵墙，一堵墙朝上，一堵墙朝下。这两个点在竖直方向是不需要相连的。
                # 当然，p_2 和p_3之间一般情况下，不会相连。
                if i + 2 < len(sorted_points) - 1:
                    p_3 = sorted_points[i + 2]
                    if p_2.is_close_enough_point(p_3, target_direction, self.gap):
                        # 交换他们的位置
                        if self._is_feasible_wall_line_points(p_1, p_3, target_direction):
                            sorted_points[i + 1] = p_3
                            sorted_points[i + 2] = p_2
                            cur_wall_section.append(p_3)
                            # 不需要判断p_2和p_3之前是否能相连。
                            i += 1

                # align the wall line points position.
                if len(cur_wall_section) > 1:
                    self._align_wall_line_points(cur_wall_section, target_direction)
                    cur_wall_section = []
            else:
                if i == len(sorted_points) - 2:
                    cur_wall_section.append(p_2)

        # 墙都是可以两两相连。
        if len(cur_wall_section) > 1:
            self._align_wall_line_points(cur_wall_section, target_direction)

        # 两两之间建立联系。一定要注意，由于排序了，p_1的值比p_2的值小。
        for i in range(len(sorted_points) - 1):
            p_1 = sorted_points[i]
            p_2 = sorted_points[i + 1]

            if target_direction == 0:
                p_1.x_visited_flag = True
                p_2.x_visited_flag = True
            else:
                p_1.y_visited_flag = True
                p_2.y_visited_flag = True

            # 中间断了。
            if not self._is_feasible_wall_line_points(p_1, p_2, target_direction):
                continue

            cur_wall_line = WallLine(p_1, p_2)
            p_1.wall_lines.append(cur_wall_line)
            p_2.wall_lines.append(cur_wall_line)
            p_1.add_connect_point(p_2)
            p_2.add_connect_point(p_1)
            wall_lines.append(cur_wall_line)

        return wall_lines

    def _is_infeasible_potential_wall_line_connection(self, point_1_forbidden_direction,
                                                      point_2_forbidden_direction,
                                                      target_direction):

        if point_1_forbidden_direction == 0 or point_2_forbidden_direction == 2:
            return True
        elif point_1_forbidden_direction == 3 or point_2_forbidden_direction == 1:
            return True
        else:
            return False

    # 一定保证point_1在point_2左边或者上边.
    # 连线是从point_1出发，连接point_2
    def _is_feasible_wall_line_points(self, point_1, point_2, target_direction):
        if point_1.type_category == 1 and point_2.type_category == 1:
            if point_1.type_sub_category == point_2.type_sub_category:
                return False

        point_1_forbidden_direction = point_1.get_forbidden_direction()
        point_2_forbidden_direction = point_2.get_forbidden_direction()

        # 去掉干扰direction。point_1_forbidden_direction 最多有两个, 共有9种情况。
        if target_direction == 0:
            point_1_forbidden_direction = list(set(point_1_forbidden_direction).difference(set([0, 2])))
            # point_2_forbidden_direction = list(set(point_2_forbidden_direction).difference(set([0, 2])))
        else:
            point_1_forbidden_direction = list(set(point_1_forbidden_direction).difference(set([1, 3])))
            # point_2_forbidden_direction = list(set(point_2_forbidden_direction).difference(set([1, 3])))

        if len(point_1_forbidden_direction) == 2:
            # 斜墙可以任意方向连接，有点🐶 by henry.hao 2023.06.29
            return True
        elif len(point_1_forbidden_direction) == 1:
            # 被禁只有方向只有向右[3]以及向下方向[0], 如果恰好point_2需要这两个方向[1](从左边来)以及[2]从上面来，那么
            # 这两个点不能相连。
            forbidden_direction = point_1_forbidden_direction[0]
            # point1 如果不允许向下[0]或者向右[3]的方向。但是，point1是在point2的上方或者左侧，因此这种情况肯定不是
            # feasible wall line.
            if forbidden_direction in [0, 3]:
                return False
            else:
                # 如果point1被禁的方向是左[1]和上[2], 不能认为point_1与point_2一定能相连。
                # 必须保证point_2允许从左和上方相连。
                # allowed_direction = 3 if forbidden_direction == 1 else 0
                # needed_direction = 1 if allowed_direction == 3 else 2
                # 如果point2需要的direction被禁止了，这两个point不能相连。
                # 这里不太好理解，注意。
                if point_2.is_feasible_direction(forbidden_direction):
                    return True
                else:
                    return False
        else:
            needed_direction = 1 if target_direction == 0 else 2
            if point_2.is_feasible_direction(needed_direction):
                return True
            else:
                return False

    def _align_wall_line_points(self, same_dir_points, direction):
        sum_coord = 0
        for cur_point in same_dir_points:
            sum_coord += cur_point.x if direction == 1 else cur_point.y
        avg_coord = int(float(sum_coord) / len(same_dir_points))
        for cur_point in same_dir_points:
            if direction == 1:
                cur_point.x = avg_coord
            else:
                cur_point.y = avg_coord

    def _build_wall_lines(self, all_wall_points):
        # 1. build the relationship of all wall points.
        all_wall_points_copied = all_wall_points.copy()

        i = 0
        inclined_wall_points = []
        all_wall_lines = []
        while len(all_wall_points_copied) > 0:
            cur_wall_point = all_wall_points_copied[0]
            # remove current wall point.
            all_wall_points_copied.remove(cur_wall_point)
            cur_wall_lines = []
            # find the points which are on the same x, y direction.
            if not cur_wall_point.x_visited_flag:
                cur_wall_lines.extend(self._build_point_wall_lines(all_wall_points_copied, cur_wall_point, 0))
            if not cur_wall_point.y_visited_flag:
                cur_wall_lines.extend(self._build_point_wall_lines(all_wall_points_copied, cur_wall_point, 1))
            # 专门处理斜墙的corner，比较烧脑
            if cur_wall_point.type_category == 0 and cur_wall_point.type_sub_category < 4:
                inclined_wall_points.append(cur_wall_point)
            all_wall_lines.extend(cur_wall_lines)
            i += 1

        inclined_wall_lines = self._inclined_wall_line(inclined_wall_points)

        all_wall_lines.extend(inclined_wall_lines)

        all_wall_lines = self._remove_alone_wall_lines(all_wall_lines)
        return all_wall_lines

    def _inclined_wall_line(self, inclined_wall_points):
        # need_remove_index = []
        inclined_wall_lines = []
        inclined_wall_points_copied = inclined_wall_points.copy()
        while len(inclined_wall_points_copied) > 0:
            cur_wall_point = inclined_wall_points_copied[0]
            inclined_wall_points_copied.remove(cur_wall_point)
            for tmp_point in inclined_wall_points_copied:
                dist = np.sqrt((tmp_point.x - cur_wall_point.x) ** 2 + (tmp_point.y - cur_wall_point.y) ** 2)
                if dist < 3:
                    if cur_wall_point in inclined_wall_points:
                        inclined_wall_points.remove(cur_wall_point)
                elif dist < 150:  # 119是临时拍的值，假设斜墙的线段距离不超过119
                    cur_wall_line = WallLine(tmp_point, cur_wall_point)
                    cur_wall_line.is_inclined_wall = True
                    tmp_point.wall_lines.append(cur_wall_line)
                    cur_wall_point.wall_lines.append(cur_wall_line)
                    tmp_point.add_connect_point(cur_wall_point)
                    cur_wall_point.add_connect_point(tmp_point)
                    inclined_wall_lines.append(cur_wall_line)
        return inclined_wall_lines

    # 通过加一些额外的wall line来达到点的完备性。
    def _repair_incompleteness_points(self, wall_points, wall_lines):
        new_wall_lines = []
        for cur_wall_point in wall_points:
            if cur_wall_point.is_completeness_point():
                continue

            cur_new_wall_lines, cur_new_points = cur_wall_point.repair_completeness(wall_lines, self.all_opening_points)
            new_wall_lines.extend(cur_new_wall_lines)

        return new_wall_lines

    # junction_class = 0:   wall junction
    # junction_class = 2:   door junction
    def _extract_junctions(self, heat_map_img, max_target_points_number, junction_type_category,
                           junction_type_sub_category, heat_map_threshold=0.5, line_width=5, junction_class=0):
        try:
            # copy the data since the heat map img will be changed here.
            heat_map_img_mask = copy.deepcopy(heat_map_img)
            wall_points = []

            for pointIndex in range(max_target_points_number):
                index = np.argmax(heat_map_img_mask)
                # get the coordinate of the x, y
                y, x = np.unravel_index(index, heat_map_img_mask.shape)
                # filter the noise.
                cur_max_value = heat_map_img_mask[y, x]
                if  np.abs(x-127)<6:
                    a = 1
                if junction_class ==0 and junction_type_category==0:
                    # a=1
                    if cur_max_value <= 1.0e-02:
                        break
                elif cur_max_value <= heat_map_threshold:
                    break

                # the pixels.
                self.tmp_pixel_list = []
                self._suppress_wall_point_neighbors(heat_map_img_mask, x, y, self.heat_map_wall_connect_threshold)

                # calculate the heatmap center as the point position.
                x_index = [value[0] for value in self.tmp_pixel_list]
                y_index = [value[1] for value in self.tmp_pixel_list]
                x = int(0.5 * (np.max(x_index) + np.min(x_index)))
                y = int(0.5 * (np.max(y_index) + np.min(y_index)))

                if junction_class == 2:
                    cur_wall_junction = DoorPoint(junction_type_category, junction_type_sub_category, x, y,
                                                  cur_max_value)
                else:
                    cur_wall_junction = WallPoint(junction_type_category, junction_type_sub_category, x, y,
                                                  cur_max_value)
                cur_wall_junction.heat_map_scope_pixels = self.tmp_pixel_list
                wall_points.append(cur_wall_junction)

            return wall_points
        except Exception as err:
            print(err)
            return []

    # set all neighbors heatmap value as -1.
    def _suppress_wall_point_neighbors(self, heat_map_img_mask, x, y, heat_map_threshold):
        try:
            # value = heat_map_img_mask[y][x]
            heat_map_img_mask[y][x] = -1
            self.tmp_pixel_list.append([x, y])

            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for delta in deltas:
                neighbor_x = x + delta[0]
                neighbor_y = y + delta[1]
                if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= self.options.width or neighbor_y >= self.options.height:
                    continue
                neighbor_value = heat_map_img_mask[neighbor_y][neighbor_x]
                if neighbor_value > heat_map_threshold:
                    self._suppress_wall_point_neighbors(heat_map_img_mask, neighbor_x, neighbor_y, heat_map_threshold)
        except Exception as err:
            print(err)

    # only consider about x, y direction.
    def calc_line_dim(self, point_1, point_2):
        if np.abs(point_2.x - point_1.x) > np.abs(point_2.y - point_1.y):
            return 0
        else:
            return 1
