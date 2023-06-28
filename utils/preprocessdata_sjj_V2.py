# coding=utf-8
import os
import json
import numpy as np
import cv2
from PIL import Image
import shutil
import random
from tqdm import tqdm

global_index = -1000
global_index_p = 0


class PreprocessDataSJJ(object):
    def __init__(self, folder_path, item_folder_name, res_folder_path):
        self.folder_path = folder_path
        self.res_folder_path = res_folder_path
        self.item_file_name = item_folder_name

        self.sub_folder_path = os.path.join(self.folder_path, item_folder_name)
        self.res_sub_folder_path = os.path.join(self.res_folder_path, item_folder_name)
        if not os.path.exists(self.res_sub_folder_path):
            os.makedirs(self.res_sub_folder_path)

        # all holes.
        self.all_holes = []
        self.all_hole_points = []

        self.all_wall_points = {}
        self.all_wall_segments = {}

        self.all_opening_points = []
        self.all_opening_lines = []

        self.all_door_points = []
        self.all_door_lines = []

        self.img_file_path = os.path.join(self.sub_folder_path, "{0}.png".format(item_folder_name))

        self.training_data_lines = []
        # It's

        self.cad_image_resized_file_path = ""

        self.id_2_items_dict = {}
        self.id_2_opening_items_dict = {}

        self.all_merged_wall_points_dict = {}

        self.scene_item = None
        self.underlay_item = None

        self.all_walls = []

        self.floor_plan_img_height = 512
        self.floor_plan_img_width = 512

        self.opening_v2_enum_list = ["Door",
                                     "Window",
                                     "Hole",
                                     "CornerWindow",
                                     "POrdinaryWindow",
                                     "CornerFlatWindow",
                                     "BayWindow"]

        super(PreprocessDataSJJ, self).__init__()

        self.json_file_path = ""
        self._init_data()

    def _init_data(self):
        list_files = os.listdir(self.sub_folder_path)
        for file_name in list_files:
            if file_name.endswith(".json"):
                self.json_file_path = os.path.join(self.sub_folder_path, file_name)
                break

        target_file_path = os.path.join(self.res_sub_folder_path, "{0}.jpg".format(self.item_file_name))
        if os.path.exists(self.img_file_path):
            # copy the img to self.res_sub_folder_path.
            shutil.copyfile(self.img_file_path, target_file_path)

    def _get_door_type2(self, door_seek_id, products):
        try:
            door_content_type = self.get_product_content_type(door_seek_id, products)
            if len(door_content_type) == 0:
                door_content_type = self._get_door_type(door_seek_id)

            # print(door_content_type)
            if "single swing" in door_content_type:
                return 1  # 单开门
            elif "door window" in door_content_type:
                return 3  # 门窗一体
            elif "double swing" in door_content_type:
                return 2  # 双开门
            else:
                return 4  # double sliding 双移门
        except Exception as err:
            print(err)
            return 1
        return 1

    # 门弧形在第几象限.  0：第四象限， 1：第一象限 2: 第二象限, 3:第三象限
    # 水平门和竖直门各有4个。
    def _get_door_direction(self, door_item):
        if 'swing' not in door_item.keys():
            swing = 1
        else:
            swing = door_item['swing']
        if 'ZRotation' not in door_item.keys():
            z_rotation = 0
        else:
            z_rotation = door_item['ZRotation']

        z_rotation_changes = int(z_rotation / 90)

        # 旋转时逆时针为负，顺时针为正
        # 象限分布逆时针增加
        swing_changes = swing - z_rotation_changes

        if swing_changes < 0:
            swing_changes += 4
        elif swing_changes >= 4:
            swing_changes -= 4

        return swing_changes

    def get_window_type2(self, window_seek_id, products):
        try:
            window_content_type = self.get_product_content_type(window_seek_id, products)
            if len(window_content_type) == 0:
                window_content_type = self.get_window_type(window_seek_id)

            if "bay" in window_content_type:
                return 1
            else:
                return 0
        except Exception as err:
            print(err)
            return 0

    def get_product_content_type(self, seek_id, products):
        for prod in products:
            if 'seekId' not in prod.keys() or 'contentType' not in prod.keys():
                continue

            if prod['seekId'] == seek_id:
                return prod['contentType']

        return "single swing"

    # point_1_dict 是直接从JSON解释的数据。
    def _calc_points_distance(self, point_1_dict, point_2_dict):
        x_1 = point_1_dict['x']
        y_1 = point_1_dict['y']

        #
        x_2 = point_2_dict['x']
        y_2 = point_2_dict['y']

        return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

    def _parse_wall_line(self, wall_line):
        wall_verties = []
        for cur_children_id in wall_line["children"]:
            cur_children_entity = self.id_2_items_dict[cur_children_id]
            if cur_children_entity["l"] != "Vertex":
                continue

            wall_verties.append(cur_children_entity)

        if len(wall_verties) != 2:
            print("Error Wall paring: {0}".format(wall_line["id"]))
            return None

        width = wall_line["width"]
        vertex_1 = wall_verties[0]
        vertex_2 = wall_verties[1]

        # Create WallPoint.
        if vertex_1["id"] in self.all_wall_points.keys():
            cur_point_1 = self.all_wall_points[vertex_1["id"]]
        else:
            cur_point_1 = Point(vertex_1["id"], vertex_1["x"], vertex_1["y"])
            self.all_wall_points[vertex_1["id"]] = cur_point_1

        if vertex_2["id"] in self.all_wall_points.keys():
            cur_point_2 = self.all_wall_points[vertex_2["id"]]
        else:
            cur_point_2 = Point(vertex_2["id"], vertex_2["x"], vertex_2["y"])
            self.all_wall_points[vertex_2["id"]] = cur_point_2

        # Create Edge.
        cur_wall_line = WallLine(wall_line["id"], cur_point_1, cur_point_2)
        cur_wall_line.width = width

        cur_point_1.connect_walls.append(cur_wall_line)
        cur_point_2.connect_walls.append(cur_wall_line)
        self.all_wall_segments[wall_line["id"]] = cur_wall_line

    # point是否已经存在？ 因为一些碎墙的原因，两个点可能非常接近。
    # 真实空间的点。
    def _is_point_exists(point, point_list):
        for cur_point in point_list:
            cur_dist = np.sqrt((point["x"] - cur_point.x) ** 2 + (point["y"] - cur_point.y) ** 2)
            # 距离小于50mm
            if cur_dist < 0.05:
                return True, cur_point
        return False, None

    def _parse_layer_wall_lines(self, active_layer):
        try:
            layer_children = active_layer["children"]
            for cur_children_id in layer_children:
                if cur_children_id not in self.id_2_items_dict.keys():
                    continue

                cur_children_entity = self.id_2_items_dict[cur_children_id]

                if cur_children_entity["l"] != "Wall":
                    continue
                # 解析墙
                self._parse_wall_line(cur_children_entity)
        except Exception as err:
            print(err)

    def _get_wall_direction(self, wall_id):
        try:
            wall_item = self.id_2_items_dict[wall_id]
            top_face_id = wall_item["faces"]["top"][0]
            face_item = self.id_2_items_dict[top_face_id]
            face_outer_loop_id = face_item["outerLoop"]
            loop = self.id_2_items_dict[face_outer_loop_id]

            min_x = 1000.0
            max_x = -1000.0
            min_y = 1000.0
            max_y = -1000.0
            # get the range box of loop.
            for child_id in loop["children"]:
                cur_co_edge = self.id_2_items_dict[child_id]
                edge_id = cur_co_edge["edge"]
                edge = self.id_2_items_dict[edge_id]
                for vertex_id in edge["children"]:
                    vertex = self.id_2_items_dict[vertex_id]

                    x = vertex["x"]
                    y = vertex["y"]
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

            if max_x - min_x > max_y - min_y:
                return 0, max_y - min_y
            else:
                return 1, max_x - min_x

        except Exception as err:
            print(err)
            return -1, 0

    def resize_image(self):
        try:
            img_pil = Image.open(self.img_file_path)
            w, h = img_pil.size

            width_scale = w / 512
            height_scale = h / 512
            if width_scale < height_scale:
                scale = width_scale
            else:
                scale = height_scale

            new_img = img_pil.resize((int(w / scale), int(h / scale)), Image.Resampling.LANCZOS)

            file_name = os.path.split(self.img_file_path)[1]
            file_name_without_ext = os.path.splitext(file_name)[0]
            if file_name_without_ext == "1":
                file_name_without_ext = os.path.split(self.res_sub_folder_path)[-1]
            resize_img_file_path = os.path.join(self.res_sub_folder_path,
                                                "{0}_resized.png".format(file_name_without_ext))
            self.cad_image_resized_file_path = resize_img_file_path
            new_img.save(resize_img_file_path)
            new_img.close()

            # copy the image.
            target_img_file_path = os.path.join(self.res_sub_folder_path, "{0}.png".format(file_name_without_ext))
            shutil.copyfile(self.img_file_path, target_img_file_path)
        except Exception as err:
            print(err)

    def format_point_info(self, jason_data):
        try:
            img_data = cv2.imread(self.cad_image_resized_file_path)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)

            floor_plan_img_height = img_data.shape[0]
            floor_plan_img_width = img_data.shape[1]

            x_offset, y_offset, self.factor = self._calc_factor(jason_data, floor_plan_img_height,
                                                                floor_plan_img_width)

            half_height = 0.5 * floor_plan_img_height
            half_width = 0.5 * floor_plan_img_width

            for cur_wall_point in self.all_wall_points.values():
                x = int(cur_wall_point.x * self.factor + half_width)
                y = -int(cur_wall_point.y * self.factor - half_height)

                x = max(x, 0)
                x = min(x, floor_plan_img_width)
                y = max(y, 0)
                y = min(y, floor_plan_img_height)

                cur_wall_point.x = x
                cur_wall_point.y = y

            for cur_wall_point in self.all_opening_points:
                x = int(cur_wall_point.x * self.factor + half_width)
                y = -int(cur_wall_point.y * self.factor - half_height)

                x = max(x, 0)
                x = min(x, floor_plan_img_width)
                y = max(y, 0)
                y = min(y, floor_plan_img_height)

                cur_wall_point.x = x
                cur_wall_point.y = y

            for cur_wall_point in self.all_door_points:
                x = int(cur_wall_point.x * self.factor + half_width)
                y = -int(cur_wall_point.y * self.factor - half_height)

                x = max(x, 0)
                x = min(x, floor_plan_img_width)
                y = max(y, 0)
                y = min(y, floor_plan_img_height)

                cur_wall_point.x = x
                cur_wall_point.y = y

            for cur_hole_point in self.all_hole_points:
                x = int(cur_hole_point.x * self.factor + half_width)
                y = -int(cur_hole_point.y * self.factor - half_height)

                x = max(x, 0)
                x = min(x, floor_plan_img_width)
                y = max(y, 0)
                y = min(y, floor_plan_img_height)

                cur_hole_point.x = x
                cur_hole_point.y = y

        except Exception as err:
            print(err)

    def _parse_corner_flat_window2(self, window_item):
        window_part_info = window_item["partsInfo"]
        # json info.
        part_json = json.loads(window_part_info)
        # Side A
        if "B" in part_json.keys():
            self._parse_side_corner_window(part_json["B"], add_wall_flag=False)
        if "C" in part_json.keys():
            self._parse_side_corner_window(part_json["C"], add_wall_flag=False)

    # 转角窗的两条线。因为转角窗是必定有墙的，所以不必特殊处理墙。
    def _parse_corner_flat_window(self, window_item):
        host_wall_id = window_item["host"]
        host_wall = None
        if host_wall_id in self.all_wall_segments.keys():
            host_wall = self.all_wall_segments[host_wall_id]

        x = window_item["x"]
        y = window_item["y"]
        x_length = np.abs(window_item["XLength"])
        y_length = np.abs(window_item["YLength"])

        # host wall thickness.
        half_host_wall_thickness = 0.5 * host_wall.width  # thickness.
        host_wall_thickness = host_wall.width
        z_rotation = window_item["ZRotation"]
        if z_rotation == 0:
            x_c = x  # - half_host_wall_thickness
            y_c = y  # + half_host_wall_thickness

            x_1 = x_c + (x_length - host_wall_thickness)
            y_1 = y_c

            x_2 = x_c
            y_2 = y_c - (y_length - host_wall_thickness)

        elif z_rotation == 90:
            x_c = x  # + half_host_wall_thickness
            y_c = y  # + half_host_wall_thickness

            x_1 = x_c - (x_length - host_wall_thickness)
            y_1 = y_c

            x_2 = x_c
            y_2 = y_c - (y_length - host_wall_thickness)

        elif z_rotation == -90:
            x_c = x  # - half_host_wall_thickness
            y_c = y  # - half_host_wall_thickness

            x_1 = x_c + (x_length - host_wall_thickness)
            y_1 = y_c

            x_2 = x_c
            y_2 = y_c + (y_length - host_wall_thickness)

        elif z_rotation == -180:
            x_c = x  # + half_host_wall_thickness
            y_c = y  # - half_host_wall_thickness

            x_1 = x_c - (x_length - host_wall_thickness)
            y_1 = y_c

            x_2 = x_c
            y_2 = y_c + (y_length - host_wall_thickness)

        else:
            print("Z_Rotation error for window Item.")
            return

        ############################################################
        cent_point = Point("-1", x_c, y_c)
        h_point = Point("-1", x_1, y_1)
        v_point = Point("-1", x_2, y_2)
        cent_point.window_type = 0
        h_point.window_type = 0
        v_point.window_type = 0
        self.all_opening_points.append(cent_point)
        self.all_opening_points.append(h_point)
        self.all_opening_points.append(v_point)

        # 窗户。
        opening_line_1 = WallLine("-1", cent_point, h_point)
        self.all_opening_lines.append(opening_line_1)

        # 窗户
        opening_line_2 = WallLine("-1", cent_point, v_point)
        self.all_opening_lines.append(opening_line_2)

    # 普通窗户。
    def _parse_normal_window_item(self, window_item):
        host_wall_id = window_item["host"]
        host_wall = None
        if host_wall_id in self.all_wall_segments.keys():
            host_wall = self.all_wall_segments[host_wall_id]

        direction, wall_thickness = self._get_wall_direction(host_wall_id)
        if direction == -1 or wall_thickness == 0.0:
            return

        x = window_item["x"]
        y = window_item["y"]
        x_length = np.abs(window_item["XLength"])
        y_length = np.abs(window_item["YLength"])
        x_scale = np.abs(window_item["XScale"])
        y_scale = np.abs(window_item["YScale"])
        z_rotation_value = window_item["ZRotation"]
        length = x_length * x_scale if x_length > y_length else y_length * y_scale
        z_rotation_value = 3.1415926535 * z_rotation_value / 180.0
        start_point = Point("-1", x - 0.5 * length * np.cos(z_rotation_value),
                            y + 0.5 * length * np.sin(z_rotation_value))
        end_point = Point("-1", x + 0.5 * length * np.cos(z_rotation_value),
                          y - 0.5 * length * np.sin(z_rotation_value))

        host_wall_direction = host_wall.calc_wall_direction()
        if host_wall_direction == 0:  # 水平墙
            start_point.y = host_wall.start_point.y
            end_point.y = host_wall.start_point.y
        else:  # 竖直墙
            start_point.x = host_wall.start_point.x
            end_point.x = host_wall.start_point.x

        start_point.window_type = 0
        end_point.window_type = 0

        opening_line = WallLine(window_item["id"], start_point, end_point)

        self.all_opening_points.append(start_point)
        self.all_opening_points.append(end_point)
        self.all_opening_lines.append(opening_line)

    def _parse_ordinary_window(self, window_item):
        window_part_info = window_item["partsInfo"]
        # json info.
        part_json = json.loads(window_part_info)
        # Side A
        x = window_item["x"]
        y = window_item["y"]

        if "host" not in window_item.keys():
            return

        host_wall_id = window_item["host"]
        if host_wall_id not in self.all_wall_segments.keys():
            return

        host_wall = self.all_wall_segments[host_wall_id]
        if "B" in part_json.keys():
            self._parse_side_corner_window(part_json["B"], offset_x=x, offset_y=y, host_wall=host_wall,
                                           window_type="POrdinaryWindow",
                                           add_wall_flag=False)

    # 添加3堵墙， 并且将飘窗转换成3个普通窗户。
    def _parse_bay_window(self, window_item):
        window_part_info = window_item["partsInfo"]
        # json info.
        part_json = json.loads(window_part_info)
        # Side A
        x = window_item["x"]
        y = window_item["y"]

        if "host" not in window_item.keys():
            return

        host_wall_id = window_item["host"]
        if host_wall_id not in self.all_wall_segments.keys():
            return

        host_wall = self.all_wall_segments[host_wall_id]
        if "A" in part_json.keys():
            self._parse_side_corner_window(part_json["A"], offset_x=x, offset_y=y, host_wall=host_wall,
                                           window_type="BayWindow",
                                           part_type="A")
        if "B" in part_json.keys():
            self._parse_side_corner_window(part_json["B"], offset_x=x, offset_y=y,
                                           host_wall=host_wall,
                                           window_type="BayWindow",
                                           part_type="B"
                                           )
        if "C" in part_json.keys():
            self._parse_side_corner_window(part_json["C"], offset_x=x, offset_y=y, host_wall=host_wall,
                                           window_type="BayWindow",
                                           part_type="C"
                                           )

    def _parse_side_corner_window(self, side_part_info, offset_x=0.0, offset_y=0.0,
                                  host_wall=None,
                                  window_type="Window",
                                  part_type="A",
                                  add_wall_flag=True):

        start_point_x = 0.5 * (side_part_info["outerFrom"]["x"] + side_part_info["innerFrom"]["x"]) + offset_x
        start_point_y = 0.5 * (side_part_info["outerFrom"]["y"] + side_part_info["innerFrom"]["y"]) + offset_y

        end_point_x = 0.5 * (side_part_info["outerTo"]["x"] + side_part_info["innerTo"]["x"]) + offset_x
        end_point_y = 0.5 * (side_part_info["outerTo"]["y"] + side_part_info["innerTo"]["y"]) + offset_y

        if window_type == "BayWindow":
            if host_wall is None:
                return
            if part_type == "A" or part_type == "C":
                direction = host_wall.calc_wall_direction()
                if direction == 0:
                    # 竖直的。
                    wall_center_pos = host_wall.start_point.y
                    if 0.5 * (start_point_y + end_point_y) < wall_center_pos:  # 竖直墙的左边。
                        start_point_y = min(start_point_y, end_point_y)
                        end_point_y = wall_center_pos
                    else:
                        max_y = max(start_point_y, end_point_y)
                        start_point_y = wall_center_pos
                        end_point_y = max_y
                else:
                    # 竖直的。
                    wall_center_pos = host_wall.start_point.x
                    if 0.5 * (start_point_x + end_point_x) < wall_center_pos:  # 竖直墙的左边。
                        start_point_x = min(start_point_x, end_point_x)
                        end_point_x = wall_center_pos
                    else:
                        max_x = max(start_point_x, end_point_x)
                        start_point_x = wall_center_pos
                        end_point_x = max_x
        elif window_type == "POrdinaryWindow":
            if host_wall is None:
                return
            direction = host_wall.calc_wall_direction()
            if direction == 0:
                start_point_y = host_wall.start_point.y
                end_point_y = host_wall.start_point.y
            else:
                start_point_x = host_wall.start_point.x
                end_point_x = host_wall.start_point.x

        # else:
        #     # 对于一般的Window：POrdinaryWindow
        #     pass

        window_start_point = Point("-1", start_point_x, start_point_y)
        window_start_point.window_type = 0
        window_end_point = Point("-1", end_point_x, end_point_y)
        window_end_point.window_type = 0
        window_line = WallLine("-1", window_start_point, window_end_point)

        self.all_opening_points.append(window_start_point)
        self.all_opening_points.append(window_end_point)
        self.all_opening_lines.append(window_line)

        if add_wall_flag:
            wall_start_point = Point("-1", start_point_x, start_point_y)
            wall_end_point = Point("-1", end_point_x, end_point_y)
            wall_line = WallLine("-1", window_start_point, window_end_point)

            wall_start_point.connect_walls.append(wall_line)
            wall_end_point.connect_walls.append(wall_line)

            wall_line.window_on_wall_flag = True

            self.all_wall_points[wall_start_point.id] = wall_start_point
            self.all_wall_points[wall_end_point.id] = wall_end_point
            self.all_wall_segments[wall_line.id] = wall_line

    # 转角飘窗
    def _parse_corner_window(self, window_item):
        window_part_info = window_item["partsInfo"]
        # json info.
        part_json = json.loads(window_part_info)

        host_wall_id = window_item[""]

        # Side A
        if "A" in part_json.keys():
            self._parse_side_corner_window(part_json["A"], window_item["l"])
        if "B" in part_json.keys():
            self._parse_side_corner_window(part_json["B"], window_item["l"])
        if "C" in part_json.keys():
            self._parse_side_corner_window(part_json["C"], window_item["l"])
        if "D" in part_json.keys():
            self._parse_side_corner_window(part_json["D"], window_item["l"])

    def _parse_window_item(self, window_item):
        class_type_name = window_item["l"]

        # 是否存在Host.
        if "host" not in window_item.keys():
            return

        host_wall_id = window_item["host"]
        if host_wall_id not in self.all_wall_segments.keys():
            return

        # 对于转角窗，可以是当成两个普通窗户来处理。
        if class_type_name == "CornerFlatWindow":  # 转角窗
            # 转角Window，不需要处理墙，只需要把其分成两各普通窗户就可。
            self._parse_corner_flat_window2(window_item)
        elif class_type_name == "CornerWindow":  # 转角飘窗
            self._parse_corner_window(window_item)
        elif class_type_name == "BayWindow":  # 飘窗
            self._parse_bay_window(window_item)
        elif class_type_name == "POrdinaryWindow":
            self._parse_ordinary_window(window_item)
        else:  # 一般的窗户。
            self._parse_normal_window_item(window_item)

    # 处理门窗
    def _parse_opening_data(self, products_data):
        # opening ID.
        for opening_id in self.id_2_opening_items_dict.keys():
            opening_item = self.id_2_opening_items_dict[opening_id]
            class_type_name = opening_item["l"]

            x = opening_item["x"]
            y = opening_item["y"]

            # 获得Wall
            if "host" not in opening_item.keys():
                continue

            host_wall_id = opening_item["host"]
            host_wall = None
            if host_wall_id in self.all_wall_segments.keys():
                host_wall = self.all_wall_segments[host_wall_id]
            if 'ZRotation' not in opening_item.keys():
                z_rotation_value = 360
            else:
                z_rotation_value = opening_item["ZRotation"] + 360
            relative_rotation_value = z_rotation_value / 90.0
            if relative_rotation_value - np.floor(relative_rotation_value) > 0.2:
                continue

            # 将CornerFlatWindow和CornerWindow处理成两个窗户。
            if class_type_name == "Window" or \
                    class_type_name == "CornerFlatWindow" or \
                    class_type_name == "CornerWindow" or \
                    class_type_name == "BayWindow" or \
                    class_type_name == "POrdinaryWindow":
                self._parse_window_item(opening_item)
            else:
                x_length = np.abs(opening_item["XLength"])
                y_length = np.abs(opening_item["YLength"])
                x_scale = np.abs(opening_item["XScale"])
                y_scale = 1

                # z_rotation_value = opening_item["ZRotation"]
                length = x_length * x_scale if x_length > y_length else y_length * y_scale
                z_rotation_value = 3.1415926535 * z_rotation_value / 180.0
                start_point = Point("-1", x - 0.5 * length * np.cos(z_rotation_value),
                                    y + 0.5 * length * np.sin(z_rotation_value))
                end_point = Point("-1", x + 0.5 * length * np.cos(z_rotation_value),
                                  y - 0.5 * length * np.sin(z_rotation_value))

                # 门
                if class_type_name == "Door":

                    door_line = WallLine(opening_id, start_point, end_point)
                    door_type = self._get_door_type2(opening_item["seekId"], products_data)
                    door_direction = self._get_door_direction(opening_item)

                    # set the door type.
                    start_point.door_type = door_type
                    end_point.door_type = door_type

                    start_point.door_direction = door_direction
                    end_point.door_direction = door_direction

                    self.all_door_points.append(start_point)
                    self.all_door_points.append(end_point)
                    self.all_door_lines.append(door_line)
                else:
                    host_id = opening_item["host"]  # 洞
                    hole = Hole(opening_id, start_point, end_point, host_id)
                    self.all_hole_points.append(start_point)
                    self.all_hole_points.append(end_point)
                    self.all_holes.append(hole)

    # parse item data.
    def _parse_items_data_dict(self, json_data):
        all_data_items = json_data["data"]

        for cur_item in all_data_items:
            if cur_item["l"] == "Scene":
                self.scene_item = cur_item
                continue

            if cur_item["l"] == "Underlay":
                self.underlay_item = cur_item
                continue

            # DOOR and Window.
            if cur_item["l"] in self.opening_v2_enum_list:
                self.id_2_opening_items_dict[cur_item["id"]] = cur_item
            else:
                # Wall、Vertex、Edge
                self.id_2_items_dict[cur_item["id"]] = cur_item

    # 处于同一条直线的wall，要合并成一个wall
    def merge_walls(self):
        try:
            new_wall_list = []

            all_wall_segments = list(self.all_wall_segments.values())
            for wall_line in all_wall_segments:
                wall_line.switch_points()
                wall_line.mark_x = False
                wall_line.mark_y = False

            for wall_line in all_wall_segments:
                if (not wall_line.mark_x) and (self.calc_line_dim(wall_line.start_point, wall_line.end_point) == 0):
                    x_aligned_wall_list = self._merge_aligned_walls(wall_line, all_wall_segments, 0)
                    wall_line.mark_x = True
                    new_wall_list.extend(x_aligned_wall_list)

                if not wall_line.mark_y and (self.calc_line_dim(wall_line.start_point, wall_line.end_point) == 1):
                    y_aligned_wall_list = self._merge_aligned_walls(wall_line, all_wall_segments, 1)
                    wall_line.mark_y = True
                    new_wall_list.extend(y_aligned_wall_list)

            self.all_wall_segments = new_wall_list
        except Exception as err:
            print(err)

    def calc_wall_length(self):
        return np.sqrt((self.start_point.x - self.end_point.x) * (self.start_point.x - self.end_point.x) +
                       (self.start_point.y - self.end_point.y) * (self.start_point.y - self.end_point.y))

    def calc_line_dim(self, point_1, point_2, threshold=5, space_flag=False):
        # space_flag.
        if not space_flag:
            if np.abs(point_2.x - point_1.x) > threshold and np.abs(point_2.y - point_1.y) > threshold:
                return -1

        if np.abs(point_2.x - point_1.x) > np.abs(point_2.y - point_1.y):
            return 0
        else:
            return 1

    def _save_line_points(self, wall_line_list, type_name="door"):
        for wall_line in wall_line_list:
            start_point = wall_line.start_point
            end_point = wall_line.end_point

            start_point, end_point = self._switch_start_end_points(start_point, end_point)

            if type_name == "door":
                str_line = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(start_point.x, start_point.y, end_point.x,
                                                                        end_point.y, type_name, start_point.door_type,
                                                                        start_point.door_direction)
            elif type_name == "opening":
                str_line = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t1\n".format(start_point.x, start_point.y, end_point.x,
                                                                      end_point.y, type_name, start_point.window_type)
            else:
                str_line = "{0}\t{1}\t{2}\t{3}\t{4}\t1\t1\n".format(start_point.x, start_point.y, end_point.x,
                                                                    end_point.y, type_name)

            self.training_data_lines.append(str_line)

    def _switch_start_end_points(self, from_point, to_point):
        start_point, end_point = None, None
        if from_point.x == to_point.x:
            if from_point.y > to_point.y:
                start_point = to_point
                end_point = from_point
            else:
                start_point = from_point
                end_point = to_point
        elif from_point.x < to_point.x:
            start_point = from_point
            end_point = to_point
        else:
            start_point = to_point
            end_point = from_point

        return start_point, end_point

    def _find_aligned_wall_lines(self, wall_line, all_wall_lists, direction, threshold=5):
        aligned_walls = []
        start_pt = wall_line.start_point
        end_pt = wall_line.end_point

        fix_value = 0.5 * (start_pt.y + end_pt.y) if direction == 0 else 0.5 * (start_pt.x + end_pt.x)

        threshold = 8
        tuned_threshold = threshold

        for tmp_wall in all_wall_lists:
            if wall_line.id == tmp_wall.id:
                continue

            # 该直线以及被处理过。
            if direction == 0 and tmp_wall.mark_x:
                continue
            if direction == 1 and tmp_wall.mark_y:
                continue

            if self.calc_line_dim(tmp_wall.start_point, tmp_wall.end_point) != direction:
                continue

            tmp_fix_value = 0.5 * (tmp_wall.start_point.y + tmp_wall.end_point.y) if direction == 0 else 0.5 * (
                    tmp_wall.start_point.x + tmp_wall.end_point.x)
            if np.abs(fix_value - tmp_fix_value) < tuned_threshold:
                aligned_walls.append(tmp_wall)

                if direction == 0:
                    tmp_wall.mark_x = True
                else:
                    tmp_wall.mark_y = True

        return aligned_walls

    # WallLine 是否在all_wall_lines中存在。
    # 通过比较对应的两个端点的位置。
    def _is_wall_line_exist(self, all_wall_lines_dict, wall_line, same_threshold):
        #
        for cur_wall in all_wall_lines_dict:
            # 是相同的Wall.
            if self._is_same_wall_line(cur_wall, wall_line, same_threshold):
                return True

        return False

    # 两个wall line是否相同。
    def _is_same_wall_line(self, wall_line_1, wall_line_2, same_threshold):
        if wall_line_1.id == wall_line_2.id:
            return True

        if self._is_same_wall_point(wall_line_1.start_point, wall_line_2.start_point, same_threshold) and \
                self._is_same_wall_point(wall_line_1.end_point, wall_line_2.end_point, same_threshold):
            return True

        return False

    def _is_same_wall_point(self, point_1, point_2, same_threshold=5):
        if point_1.id == point_2.id:
            return True

        diff_x = point_1.x - point_2.x
        diff_y = point_1.y - point_2.y
        diff_length = np.sqrt(diff_x * diff_x + diff_y * diff_y)

        if diff_length < same_threshold:
            return True

        return False

    def _get_same_point(self, point_1, point_list, same_threshold=5):
        for cur_point in point_list:
            if self._is_same_wall_point(point_1, cur_point, same_threshold):
                return cur_point
        return None

    def _update_wall_points(self, wall, update_point, same_point):
        direction = self.calc_line_dim(wall.start_point, wall.end_point)

        if wall.start_point == update_point:
            wall.start_point = same_point
            if direction == 0:
                wall.end_point.y = same_point.y
            else:
                wall.end_point.x = same_point.x
        if wall.end_point == update_point:
            wall.end_point = same_point
            if direction == 0:
                wall.start_point.y = same_point.y
            else:
                wall.start_point.x = same_point.x

    # merge the points.
    def merge_points(self):
        all_diff_points_dict = {}

        # 对所有的墙点进行合并。
        for cur_point_id in self.all_wall_points.keys():
            cur_point = self.all_wall_points[cur_point_id]
            same_point = self._get_same_point(cur_point, all_diff_points_dict.values(), 5)
            if same_point is None:
                all_diff_points_dict[cur_point_id] = cur_point
            else:
                # 找到相同的point, cur_point
                for cur_wall in cur_point.connect_walls:
                    self._update_wall_points(cur_wall, cur_point, same_point)

        self.all_wall_points = all_diff_points_dict

    # 和这个点最近的WALL。 当然，
    def _calc_close_wall_of_point(self, point, point_wall):
        point_wall_direction = point_wall.calc_line_dim(point_wall.start_point, point_wall.end_point)

        min_distance = 100000.0
        min_wall = None
        for cur_wall in self.all_wall_segments:
            if cur_wall == point_wall:
                continue
            cur_wall_direction = cur_wall.calc_line_dim(cur_wall.start_point, cur_wall.end_point)
            if point_wall_direction == cur_wall_direction:
                continue

            if point_wall_direction == 0 and cur_wall_direction == 1:
                # 水平的墙.
                distance = abs(cur_wall.start_point.x - point.x)
                if min_distance > distance:
                    min_wall = cur_wall
                min_distance = min(distance, min_distance)

            else:
                distance = abs(cur_wall.start_point.y - point.y)
                if min_distance > distance:
                    min_wall = cur_wall
                min_distance = min(distance, min_distance)
        return min_wall, min_distance

    # 第一种Case：多余的墙。这个墙在是角落，比较短，且有个端点只有一个墙相连。和最近的比较长的墙平行。并且距离比较小。
    # 第二种case: 如果墙很短，直接拿掉。

    def remove_smash_wall(self):
        smash_walls = []
        for wall_1 in self.all_wall_segments:
            length_1 = wall_1.calc_wall_length()
            if length_1 < 8:
                smash_walls.append(wall_1)
            direction_1 = wall_1.calc_wall_direction()
            if direction_1 < 0:
                continue
            for wall_2 in self.all_wall_segments:
                if wall_1.p_id == wall_2.p_id:
                    continue
                if wall_2 in smash_walls:
                    continue
                direction_2 = wall_2.calc_wall_direction()
                # 去除
                if direction_1 != direction_2:
                    continue

                dist = wall_1.calc_distance(wall_2)
                # 经验值. 两者考的非常近。
                if dist > 0.6 * (wall_1.width + wall_2.width) * self.factor:
                    continue

                # 是否这条线被包含了。
                if direction_1 == 0:
                    # wall_1 被Wall_2包含了。
                    if wall_1.start_point.x >= wall_2.start_point.x and wall_1.end_point.x <= wall_2.end_point.x:
                        smash_walls.append(wall_1)
                    if wall_2.start_point.x >= wall_1.start_point.x and wall_2.end_point.x <= wall_1.end_point.x:
                        smash_walls.append(wall_2)
                else:
                    if wall_1.start_point.y >= wall_2.start_point.y and wall_1.end_point.y <= wall_2.end_point.y:
                        smash_walls.append(wall_1)
                    if wall_2.start_point.y >= wall_1.start_point.y and wall_2.end_point.y <= wall_1.end_point.y:
                        smash_walls.append(wall_2)

        for cur_wall in smash_walls:
            if cur_wall in self.all_wall_segments:
                self.all_wall_segments.remove(cur_wall)

    def update_walls_points_position(self):
        for cur_wall_line in self.all_wall_segments:
            if cur_wall_line.start_point.p_id not in self.all_merged_wall_points_dict.keys():
                cur_wall_line.start_point.connect_walls = []
                self.all_merged_wall_points_dict[cur_wall_line.start_point.p_id] = cur_wall_line.start_point
            cur_wall_line.start_point.connect_walls.append(cur_wall_line)

            if cur_wall_line.end_point.p_id not in self.all_merged_wall_points_dict.keys():
                cur_wall_line.end_point.connect_walls = []
                self.all_merged_wall_points_dict[cur_wall_line.end_point.p_id] = cur_wall_line.end_point
            cur_wall_line.end_point.connect_walls.append(cur_wall_line)

    def remove_small_room_walls(self):
        for point_id in self.all_merged_wall_points_dict.keys():
            point = self.all_merged_wall_points_dict[point_id]
            # 如果这个点的连接墙都是很短的墙，去掉他们。
            keep_flag = False
            for tmp_wall in point.connect_walls:
                tmp_wall_length = tmp_wall.calc_wall_length()

                # 20个像素
                if tmp_wall_length > 20 or tmp_wall.window_on_wall_flag:
                    keep_flag = True
                    break

            if not keep_flag:
                for tmp_wall in point.connect_walls:
                    if tmp_wall in self.all_wall_segments:
                        self.all_wall_segments.remove(tmp_wall)

    # 会存在一些点很靠近WALL，但是却没有和这些WALL相连。
    # 实际上，这些点是要和墙相连的。一般存在于：
    def reset_wall_close_points(self):
        for cur_wall_line in self.all_wall_segments:
            # Line的方向。
            line_direction = cur_wall_line.calc_line_dim(cur_wall_line.start_point, cur_wall_line.end_point)
            if len(cur_wall_line.start_point.connect_walls) < 2:
                start_point_min_wall, start_point_min_wall_distance = self._calc_close_wall_of_point(
                    cur_wall_line.start_point,
                    cur_wall_line)
                # 距离很小，并且相邻的wall不是另外一个点的connect——wall
                if start_point_min_wall_distance < 15 and \
                        start_point_min_wall not in cur_wall_line.end_point.connect_walls:
                    # 微调这个点的坐标。
                    if line_direction == 0:
                        cur_wall_line.start_point.x = start_point_min_wall.start_point.x
                    else:
                        cur_wall_line.start_point.y = start_point_min_wall.start_point.y

            if len(cur_wall_line.end_point.connect_walls) < 2:
                end_point_min_wall, end_point_min_wall_distance = self._calc_close_wall_of_point(
                    cur_wall_line.end_point, cur_wall_line)
                # 距离很小，并且相邻的wall不是另外一个点的connect——wall
                if end_point_min_wall_distance < 15 and \
                        end_point_min_wall not in cur_wall_line.start_point.connect_walls:
                    # 微调这个点的坐标。
                    if line_direction == 0:
                        cur_wall_line.end_point.x = end_point_min_wall.start_point.x
                    else:
                        cur_wall_line.end_point.y = end_point_min_wall.start_point.y

    def _reset_avg_position(self, wall_lines, direction):
        if direction == 0:
            sum_y = sum([obj.start_point.y + obj.end_point.y for obj in wall_lines])
            avg_y = int(sum_y / (2 * float(len(wall_lines))))
            for obj in wall_lines:
                obj.start_point.y = avg_y
                obj.end_point.y = avg_y
        else:
            sum_x = sum([obj.start_point.x + obj.end_point.x for obj in wall_lines])
            avg_x = int(sum_x / (2 * float(len(wall_lines))))
            for obj in wall_lines:
                obj.start_point.x = avg_x
                obj.end_point.x = avg_x

    def _is_walls_contained_by_other_wall(self, wall, all_walls, direction):
        min_value = min(wall.start_point.x, wall.end_point.x) if direction == 0 else min(wall.start_point.y,
                                                                                         wall.end_point.y)
        max_value = max(wall.start_point.x, wall.end_point.x) if direction == 0 else max(wall.start_point.y,
                                                                                         wall.end_point.y)
        for cur_wall in all_walls:
            if cur_wall == wall:
                continue

            # 水平墙。
            if direction == 0:
                if min_value >= cur_wall.start_point.x and max_value <= cur_wall.end_point.x:
                    return True
            else:
                if min_value >= cur_wall.start_point.y and max_value <= cur_wall.end_point.y:
                    return True

        return False

    # 由于碎墙的原因，很有可能一堵墙在另外一堵墙内部。
    def _remove_sorted_dumplicate_walls(self, sorted_walls, direction):
        new_sorted_walls = []

        for cur_wall in sorted_walls:
            if not self._is_walls_contained_by_other_wall(cur_wall, sorted_walls, direction):
                new_sorted_walls.append(cur_wall)

        return new_sorted_walls

    # 将在一条直线上的墙合并在一起。
    def _merge_aligned_walls(self, wall_line, all_wall_list, direction, threshold=5):
        try:

            # 它自己本身。
            aligned_walls = self._find_aligned_wall_lines(wall_line, all_wall_list, direction)
            if len(aligned_walls) == 0:
                return [wall_line]

            # 重新设置这些aligned_walls的某些坐标值。
            self._reset_avg_position(aligned_walls, direction)

            aligned_walls.append(wall_line)

            avg_width = sum([a.width for a in aligned_walls])
            avg_width /= len(aligned_walls)

            threshold = 8
            v_threshold = threshold
            h_threshold = threshold
            same_threshold = h_threshold if direction == 0 else v_threshold

            non_duplicated_aligned_walls = []
            for cur_wall in aligned_walls:

                # 碎墙。。。。。。会导致墙的start_point和end_point是一样的点。
                if cur_wall.start_point.id == cur_wall.end_point.id:
                    continue

                if not self._is_wall_line_exist(non_duplicated_aligned_walls, cur_wall, same_threshold):
                    non_duplicated_aligned_walls.append(cur_wall)

            # reset the data.
            aligned_walls = non_duplicated_aligned_walls

            if len(aligned_walls) == 0:
                return []

            # sort the aligned_walls.
            if direction == 0:
                aligned_walls = sorted(aligned_walls, key=lambda obj: obj.start_point.x)
            else:
                aligned_walls = sorted(aligned_walls, key=lambda obj: obj.start_point.y)

            aligned_walls = self._remove_sorted_dumplicate_walls(aligned_walls, direction)

            new_walls = []
            start_pt = None
            end_pt = None

            for i in range(len(aligned_walls)):
                wall_1 = aligned_walls[i]
                if start_pt is None:
                    start_pt = wall_1.start_point
                    end_pt = wall_1.end_point
                else:
                    # 是否相连
                    if (direction == 0 and np.abs(wall_1.start_point.x - end_pt.x) < h_threshold) or \
                            (direction == 1 and np.abs(wall_1.start_point.y - end_pt.y) < v_threshold):
                        end_pt = wall_1.end_point
                    else:
                        # 不相连
                        wall_obj = WallLine(-1, start_pt, end_pt)
                        start_pt.connect_walls.append(wall_obj)
                        end_pt.connect_walls.append(wall_obj)
                        wall_obj.width = avg_width
                        new_walls.append(wall_obj)
                        start_pt = wall_1.start_point
                        end_pt = wall_1.end_point

            if direction == 0:
                avg_y = int(0.5 * (start_pt.y + end_pt.y))
                start_pt.y = avg_y
                end_pt.y = avg_y
            else:
                avg_x = int(0.5 * (start_pt.x + end_pt.x))
                start_pt.x = avg_x
                end_pt.x = avg_x

            new_wall_line = WallLine(-1, start_pt, end_pt)
            start_pt.connect_walls.append(new_wall_line)
            end_pt.connect_walls.append(new_wall_line)
            new_wall_line.width = avg_width
            new_walls.append(new_wall_line)
            return new_walls
        except Exception as err:
            print(err)
            return []

    def draw_points(self, all_wall_points, line_width=5, file_name="Node", background_img_data=None, save_flag=True,
                    rgb_color=[0, 0, 255]):
        try:
            if background_img_data is None:
                img_data = cv2.imread(self.img_file_path)
                img_data = shutil.copy.deepcopy(img_data)
            else:
                img_data = background_img_data

            line_color = np.random.rand(3) * 255
            line_color[0] = rgb_color[0]
            line_color[1] = rgb_color[1]
            line_color[2] = rgb_color[2]

            floor_plan_img_height = img_data.shape[0]
            floor_plan_img_width = img_data.shape[1]

            for cur_wall_point in all_wall_points:
                x = cur_wall_point.x
                y = cur_wall_point.y

                img_data[max(y - line_width, 0):min(y + line_width, floor_plan_img_height - 1),
                max(x - line_width, 0):min(x + line_width, floor_plan_img_width - 1)] = line_color
                # #
                if cur_wall_point.p_id == 43:
                    cv2.putText(img_data, str(cur_wall_point.p_id), (x, y),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 255, 0))

            if save_flag:
                wall_point_res_file_path = os.path.join(self.res_sub_folder_path, "{0}.jpg".format(file_name))
                cv2.imwrite(wall_point_res_file_path, img_data)
            else:
                return img_data
        except Exception as err:
            print(err)

    def draw_lines(self, all_wall_lines, line_width=3, file_name="WallLines", rgb_color=None,
                   background_img_data=None, save_flag=True):
        try:
            if background_img_data is None:
                img_data = cv2.imread(self.img_file_path)
                image = shutil.copy.deepcopy(img_data)
            else:
                image = background_img_data

            self.floor_plan_img_height = image.shape[0]
            self.floor_plan_img_width = image.shape[1]

            i = 0
            for wall_line in all_wall_lines:
                line_color = np.random.rand(3) * 255
                if rgb_color is not None:
                    line_color[0] = rgb_color[0]
                    line_color[1] = rgb_color[1]
                    line_color[2] = rgb_color[2]

                point_1 = wall_line.start_point
                point_2 = wall_line.end_point

                i += 1

                line_dim = self.calc_line_dim(point_1, point_2)

                fixedValue = int(round((point_1.y + point_2.y) / 2)) if line_dim == 0 else int(
                    round((point_1.x + point_2.x) / 2))
                minValue = int(round(min(point_1.x, point_2.x))) if line_dim == 0 else int(
                    round(min(point_1.y, point_2.y)))
                maxValue = int(round(max(point_1.x, point_2.x))) if line_dim == 0 else int(
                    round(max(point_1.y, point_2.y)))

                if line_dim == 0:
                    image[max(fixedValue - line_width, 0):min(fixedValue + line_width, self.floor_plan_img_height),
                    minValue:maxValue + 1, :] = line_color

                    cv2.putText(image, str(wall_line.p_id), (int(0.5 * (maxValue + minValue)), fixedValue),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 255, 0))

                else:
                    image[minValue:maxValue + 1,
                    max(fixedValue - line_width, 0):min(fixedValue + line_width, self.floor_plan_img_width),
                    :] = line_color
                    # #
                    cv2.putText(image, str(wall_line.p_id), (fixedValue, int(0.5 * (maxValue + minValue))),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 255, 0))

            if save_flag:
                wall_lines_res_file_path = os.path.join(self.res_sub_folder_path, "{0}.jpg".format(file_name))
                cv2.imwrite(wall_lines_res_file_path, image)
            else:
                return image

        except Exception as err:
            print(err)

    def extract_primitive_data(self):
        # try:
            if not os.path.exists(self.img_file_path) or not os.path.exists(self.json_file_path):
                return None

            # Load Image Data.
            img_data = cv2.imread(self.img_file_path)
            self.floor_plan_img_height = img_data.shape[0]
            self.floor_plan_img_width = img_data.shape[1]

            # Load Json Data.
            with open(self.json_file_path, 'r', encoding=r'UTF-8') as load_f:
                floorplan_json_data = json.load(load_f)

            # Parse the Json Data.
            self._parse_items_data_dict(floorplan_json_data)

            # 场景数据。
            if self.scene_item is None:
                return None

            # active layer就是其布局界面
            active_layer_id = self.scene_item["activeLayer"]
            active_layer = self.id_2_items_dict[active_layer_id]

            # 解析 wall 数据
            self._parse_layer_wall_lines(active_layer)

            # 解析 opening 数据
            self._parse_opening_data(floorplan_json_data['products'])

            # 图片大小改为为512*512
            self.resize_image()

            # mapping points from 3D space to pixel space.
            self.format_point_info(floorplan_json_data)

            # 把相同点进行合并。
            self.merge_points()

            # 墙进行合并
            self.merge_walls()

            # 去掉多余的墙,比如，用一堵墙来标识一个管道等等。
            self.remove_smash_wall()

            # update wall point connects.
            self.update_walls_points_position()

            self.reset_wall_close_points()

            all_wall_points = list(self.all_wall_points.values())
            img_data = cv2.imread(self.cad_image_resized_file_path)
            back_groud_img_data = img_data.copy()
            self.draw_points(all_wall_points, background_img_data=back_groud_img_data, file_name="WallPoint")
            back_groud_img_data = img_data.copy()
            self.draw_points(self.all_door_points, background_img_data=back_groud_img_data, file_name="DoorPoint")
            back_groud_img_data = img_data.copy()
            self.draw_points(self.all_opening_points, background_img_data=back_groud_img_data, file_name="OpeningPoint")
            back_groud_img_data = img_data.copy()
            self.draw_points(self.all_hole_points, line_width=2, background_img_data=back_groud_img_data,
                             file_name="HolePoint")
            back_groud_img_data = img_data.copy()
            self.draw_lines(self.all_wall_segments, line_width=2, background_img_data=back_groud_img_data,
                            file_name="WallLines")
            back_groud_img_data = img_data.copy()
            self.draw_lines(self.all_opening_lines, line_width=2, background_img_data=back_groud_img_data,
                            file_name="OpeningLines")
            back_groud_img_data = img_data.copy()
            self.draw_lines(self.all_door_lines, line_width=2, background_img_data=back_groud_img_data,
                            file_name="DoorLines")
            back_groud_img_data = img_data.copy()
            self.draw_lines(self.all_holes, line_width=2, background_img_data=back_groud_img_data, file_name="Holes")

            self._save_training_data()
        # except Exception as err:
        #     print(err)


    def _calc_factor(self, json_data, floor_plan_img_height, floor_plan_img_width):
        if self.underlay_item is None:
            return None
        width = 10 #self.underlay_item["width"]
        height = 15.123 #self.underlay_item["height"]

        ratio = height / (floor_plan_img_height)
        return abs(width), abs(height), abs(1.0 / ratio)

    def _save_training_data(self):
        self._save_line_points(self.all_wall_segments, type_name="wall")
        self._save_line_points(self.all_opening_lines, type_name="opening")
        self._save_line_points(self.all_door_lines, type_name="door")
        # self._save_line_points(self.all_holes, type_name="hole")

        annotation_data_file_path = os.path.join(self.res_sub_folder_path,
                                                 "{0}.txt".format(self.item_file_name))
        with open(annotation_data_file_path, "w") as f:
            # for data_line in self.training_data_lines:
            f.writelines(self.training_data_lines)

        # delete all txt in self.sub_folder_path
        for txt_file in os.listdir(self.sub_folder_path):
            if txt_file.endswith(".txt"):
                os.remove(os.path.join(self.sub_folder_path, txt_file))

        with open(os.path.join(self.sub_folder_path, "{0}.txt".format(self.factor)), "w") as f:
            f.write(str(self.factor))


class Entity(object):
    def __init__(self, id):
        if int(id) < 0:
            global global_index
            global_index -= 1
            self.id = global_index
        else:
            self.id = id

        global global_index_p
        global_index_p += 1
        self.p_id = global_index_p

        self.mark_flag = False

        super(Entity, self).__init__()


class Point(Entity):
    def __init__(self, id, x, y):
        self.x = x
        self.y = y

        self.connect_walls = []

        super(Point, self).__init__(id)

    def calc_distance(self, point):
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)


class WallLine(Entity):
    def __init__(self, id, from_point, to_point):

        self.start_point = from_point
        self.end_point = to_point

        self.wall_length = np.sqrt(
            (self.end_point.x - self.start_point.x) ** 2 + (self.end_point.y - self.start_point.y) ** 2)

        # one wall related two co wall.
        self.co_wall_id_list = []

        self.mark_x = False
        self.mark_y = False

        # 墙的宽度。
        self.width = 0

        super(WallLine, self).__init__(id)

        # start point 和 end point换个位置。
        direction = self.calc_line_dim(self.start_point, self.end_point)
        if direction == 0:  # 水平
            if self.start_point.x > self.end_point.x:
                tmp = self.start_point
                self.start_point = self.end_point
                self.end_point = tmp
        else:
            if self.start_point.y > self.end_point.y:
                tmp = self.start_point
                self.start_point = self.end_point
                self.end_point = tmp

    def calc_line_dim(self, point_1, point_2, threshold=5, space_flag=False):
        # space_flag.
        if not space_flag:
            if np.abs(point_2.x - point_1.x) > threshold and np.abs(point_2.y - point_1.y) > threshold:
                return -1

        if np.abs(point_2.x - point_1.x) > np.abs(point_2.y - point_1.y):
            return 0
        else:
            return 1

    def calc_wall_length(self):
        return np.sqrt((self.start_point.x - self.end_point.x) * (self.start_point.x - self.end_point.x) +
                       (self.start_point.y - self.end_point.y) * (self.start_point.y - self.end_point.y))

    def calc_wall_direction(self, space_flag=False):
        return self.calc_line_dim(self.start_point, self.end_point, space_flag=space_flag)

    def calc_distance(self, src_wall, space_flag=False):
        direction = self.calc_wall_direction(space_flag=space_flag)
        src_direction = src_wall.calc_wall_direction(space_flag=space_flag)
        if direction != src_direction:
            return -1

        if direction == 0:
            return abs(src_wall.start_point.y - self.start_point.y)
        else:
            return abs(src_wall.start_point.x - self.start_point.x)

    def switch_points(self):
        direction = self.calc_line_dim(self.start_point, self.end_point)
        if direction < 0:
            return

        if direction == 0 and self.start_point.x > self.end_point.x:
            tmp_pt = self.start_point
            self.start_point = self.end_point
            self.end_point = tmp_pt
        elif direction == 1 and self.start_point.y > self.end_point.y:
            tmp_pt = self.start_point
            self.start_point = self.end_point
            self.end_point = tmp_pt

    def _is_terminal_point(self, point, threshold=5):
        if np.sqrt((self.start_point.x - point.x) ** 2 + (self.start_point.y - point.y) ** 2) < threshold:
            return True
        if np.sqrt((self.end_point.x - point.x) ** 2 + (self.end_point.y - point.y) ** 2) < threshold:
            return True
        return False

    # 从数据上判断。
    def is_walls_connected(self, another_wall):
        if self._is_terminal_point(another_wall.start_point) or self._is_terminal_point(another_wall.end_point):
            return True
        else:
            return False


class Hole(WallLine):
    def __init__(self, id, from_point, to_point, wall_host_id):
        self.wall_host_id = wall_host_id
        self.room_host_list = []

        super(Hole, self).__init__(id, from_point, to_point)

    def add_room_host(self, room_id):
        self.room_host_list.append(room_id)


def process_design_data(folder_path, sub_folder_name, res_folder_path):
    # try:
        pro_data = PreprocessDataSJJ(folder_path, sub_folder_name, res_folder_path)
        pro_data.extract_primitive_data()
    # except Exception as err:
    #     print(err)


def create_training_data(folder_path, res_folder_path):
    if not os.path.exists(res_folder_path):
        os.makedirs(res_folder_path)

    list_dir = os.listdir(folder_path)

    img_list_files = []
    annotation_list_files = []

    res_data_folder_name = "Data"
    res_data_folder_path = os.path.join(res_folder_path, res_data_folder_name)
    if not os.path.exists(res_data_folder_path):
        os.makedirs(res_data_folder_path)

    test_sample_index_list = random.sample(range(len(list_dir)), len(list_dir))

    for i in range(len(test_sample_index_list)):
        index = test_sample_index_list[i]
        sub_folder_name = list_dir[index]

        print(sub_folder_name)

        sub_folder_path = os.path.join(folder_path, sub_folder_name)
        if "_resized" in sub_folder_name:
            img_file_name = "{0}.jpg".format(sub_folder_name)
            annotation_file_name = "{0}.txt".format(sub_folder_name)

            target_img_file_name = "{0}_2.jpg".format(sub_folder_name)
            target_annotation_file_name = "{0}_2.txt".format(sub_folder_name)
        else:
            img_file_name = "{0}_resized.png".format(sub_folder_name)  # 需要Copy这张图片
            annotation_file_name = "{0}.txt".format(sub_folder_name)  # 但是txt却没有变化。

            target_img_file_name = img_file_name
            target_annotation_file_name = annotation_file_name

        img_file_path = os.path.join(sub_folder_path, img_file_name)
        if not os.path.exists(img_file_path):
            continue
        annotation_file_path = os.path.join(sub_folder_path, annotation_file_name)
        if not os.path.exists(annotation_file_path):
            continue

        img_list_files.append("{0}/{1}".format(res_data_folder_name, target_img_file_name))
        annotation_list_files.append("{0}/{1}".format(res_data_folder_name, target_annotation_file_name))

        shutil.copyfile(img_file_path, os.path.join(res_data_folder_path, target_img_file_name))
        shutil.copyfile(annotation_file_path, os.path.join(res_data_folder_path, target_annotation_file_name))

    # save the results.
    with open(os.path.join(res_folder_path, "train.txt"), "w") as f:
        for i in range(len(img_list_files)):
            f.write("{0}\t{1}\n".format(img_list_files[i], annotation_list_files[i]))


if __name__ == "__main__":
    # 待处理的图片&json文件目录
    folder_path = "sjj_v2/data"
    # 解析结果可视化目录
    res_folder_path = "sjj_v2/result"

    train_folder_path = "sjj_v2/traindata"
    if not os.path.exists(res_folder_path):
        os.makedirs(res_folder_path)
    all_folders = os.listdir(folder_path)
    for sub_folder_name in tqdm(all_folders):
        process_design_data(folder_path, sub_folder_name, res_folder_path)

    create_training_data(res_folder_path, train_folder_path)