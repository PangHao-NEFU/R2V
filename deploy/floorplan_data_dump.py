# coding=utf-8

import math


class FloorplanDataDump(object):
    def __init__(self, wall_builder_obj):

        self.wall_builder_obj = wall_builder_obj

        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

        self.offset_x = 0
        self.offset_y = 0

        self.max_wall_thickness = 0

        self.need_offset_to_room_center = False

        super(FloorplanDataDump, self).__init__()

    def _calc_default_measure_scale_ratio(self, single_swing_door_list, max_wall_thickness):
        door_num = len(single_swing_door_list)
        default_door_width = 0.8
        if door_num >= 1:
            single_swing_door_list.sort(key=lambda x: x[4])
            middle_door_index = int(door_num / 2)
            door_width = single_swing_door_list[middle_door_index][4]
            ratio = default_door_width / door_width
        else:
            ratio = 0.24 / max_wall_thickness
        return ratio

    def _init_data_(self):
        offset_x = 0.5 * self.wall_builder_obj.floor_plan_img_width
        offset_y = 0.5 * self.wall_builder_obj.floor_plan_img_height
        self.min_y = 0
        self.max_y = self.wall_builder_obj.floor_plan_img_height - 1
        if self.need_offset_to_room_center:
            min_x = 10000
            min_y = 10000
            max_x = -10000
            max_y = -10000

            # construct pixel information
            for i in range(len(self.wall_builder_obj.all_wall_lines)):
                cur_wall = self.wall_builder_obj.all_wall_lines[i]
                if cur_wall.line_dim() == 0:
                    positions = [cur_wall.start_point.x, cur_wall.boundary_range_box[1], cur_wall.end_point.x,
                                 cur_wall.boundary_range_box[3]]
                else:
                    positions = [cur_wall.boundary_range_box[0], cur_wall.start_point.y,
                                 cur_wall.boundary_range_box[2],
                                 cur_wall.end_point.y]

                # find the min and max value in wall list
                min_x = min(min(positions[0], positions[2]), min_x)
                max_x = max(max(positions[0], positions[2]), max_x)
                min_y = min(min(positions[1], positions[3]), min_y)
                max_y = max(max(positions[1], positions[3]), max_y)

            offset_x = (min_x + max_x) * 0.5
            offset_y = (min_y + max_y) * 0.5

            self.min_x = min_x
            self.max_x = max_x
            self.min_y = min_y
            self.min_y = min_y

        # normalize the position, make the room center at(0, 0)
        self.offset_x = offset_x
        self.offset_y = offset_y

        max_wall_thickness = 0.01
        for cur_wall in self.wall_builder_obj.all_wall_lines:
            cur_wall_thickness = cur_wall.get_wall_thickness()
            if max_wall_thickness < cur_wall_thickness:
                max_wall_thickness = cur_wall_thickness
        self.max_wall_thickness = max_wall_thickness

    def _normal_position(self, positions):
        # flip y
        positions[1] = (self.min_y + self.max_y) - positions[1]
        positions[3] = (self.min_y + self.max_y) - positions[3]

        # normalize the point
        positions[0] -= self.offset_x
        positions[1] -= self.offset_y
        positions[2] -= self.offset_x
        positions[3] -= self.offset_y

        return positions

    def _convert_wall_lines(self, wall_lines):
        wall_list = []
        wall_id_positions_map = {}
        for cur_wall in wall_lines:
            # format: start_x, start_y, end_x, end_y, wall_thickness
            wall_positions = [cur_wall.start_point.x, cur_wall.start_point.y, cur_wall.end_point.x,
                              cur_wall.end_point.y]
            positions = wall_positions
            positions.append(cur_wall.get_wall_thickness())

            # normalize position.
            positions = self._normal_position(positions)
            wall_id_positions_map[cur_wall.id] = positions
            wall_list.append(positions)
        return wall_list, wall_id_positions_map

    def _convert_opening_lines(self, opening_lines):
        window_list = []
        for cur_window in opening_lines:
            positions = [0, 0, 0, 0, 0, 0]
            positions[0] = cur_window.start_point.x
            positions[1] = cur_window.start_point.y
            positions[2] = cur_window.end_point.x
            positions[3] = cur_window.end_point.y
            positions[4] = cur_window.get_wall_thickness()

            positions = self._normal_position(positions)

            # add the by window type
            if cur_window.is_bay_window():
                positions[5] = 1

            # 最后添加window
            positions.append(cur_window)
            window_list.append(positions)
        return window_list

    def _convert_doors(self, door_lines):
        door_list = []
        single_swing_door_list = []
        for cur_door in door_lines:
            positions = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            positions[0] = cur_door.start_point.x
            positions[1] = cur_door.start_point.y
            positions[2] = cur_door.end_point.x
            positions[3] = cur_door.end_point.y

            # check the swing value
            door_type = cur_door.get_door_type()
            swing = 0
            is_entrance_door = 0
            if door_type < 3:  # 单开门，双开门，门窗一体
                # find the parent wall line
                parent_wall_line = None
                for w_index in range(len(self.wall_builder_obj.all_wall_lines)):
                    cur_wall = self.wall_builder_obj.all_wall_lines[w_index]
                    for o_index in range(len(cur_wall.openings)):
                        if cur_door == cur_wall.openings[o_index]:
                            parent_wall_line = cur_wall
                            break

                # parent_wall_line is valid
                if parent_wall_line is not None:
                    # check the door is entrance door
                    if parent_wall_line.is_outer_wall:
                        is_entrance_door = 1

                    wall_start_x = parent_wall_line.start_point.x
                    wall_start_y = parent_wall_line.start_point.y
                    wall_end_x = parent_wall_line.end_point.x
                    wall_end_y = parent_wall_line.end_point.y
                    # fpLog.info("door start and end", positions[0], positions[1], positions[2], positions[3])
                    # fpLog.info("wall start and end", wall_start_x, wall_start_y, wall_end_x, wall_end_y)

                    start_side = math.sqrt((wall_start_x - positions[0]) ** 2 + (wall_start_y - positions[1]) ** 2)
                    end_side = math.sqrt((wall_end_x - positions[2]) ** 2 + (wall_end_y - positions[3]) ** 2)

                    door_direction = cur_door.get_door_direction()
                    # 水平门
                    if wall_start_y == wall_end_y:
                        if door_direction == 0:  # 图片中，门朝上
                            if start_side < end_side:
                                swing = 1
                            else:
                                swing = 2
                        else:
                            if start_side < end_side:
                                swing = 0
                            else:
                                swing = 3
                    else:  # 垂直门
                        if door_direction == 0:  # 图片中，门朝右
                            if start_side < end_side:
                                swing = 1
                            else:
                                swing = 2
                        else:
                            if start_side < end_side:
                                swing = 0
                            else:
                                swing = 3
            positions[7] = swing
            positions[8] = is_entrance_door

            positions = self._normal_position(positions)

            # 计算门的宽度
            positions[4] = math.sqrt((positions[0] - positions[2]) ** 2 + (positions[1] - positions[3]) ** 2)
            positions[5] = cur_door.get_wall_thickness()
            positions[6] = cur_door.get_door_type()
            # 对单开门单独处理,用来决定图片的比例尺
            if positions[6] == 0:
                single_swing_door_list.append(positions)

            # 最后添加door.
            positions.append(cur_door)
            door_list.append(positions)

        return door_list, single_swing_door_list

    def _dump_room_info_header(self):
        room_json_str = {
            # 中间格式版本号，当前版本号1.0
            "version": "1.0",
            "meta": {
                "unit": {
                    # 长度单位 m，cm，mm, ft 默认 m
                    "length": "m"
                }
            },
            # 户型信息
            "floorPlanInfo": {
                "walls": [],
                "doors": [],
                "windows": []
            }
        }

        return room_json_str

    def _dump_wall_info(self, room_json_str, wall_list, wall_id_positions_map, ratio):
        wallIds = list(wall_id_positions_map.keys())
        # add wall info
        for wall_index in range(len(wall_list)):
            thickness = 0.24
            is_bearing = True
            if wall_list[wall_index][4] < self.max_wall_thickness * ratio:
                thickness = 0.12
                is_bearing = False
            else:
                thickness = wall_list[wall_index][4]
            wall_info = {
                # 起点坐标[x, y, z]
                "from": {
                    "x": wall_list[wall_index][0],
                    "y": wall_list[wall_index][1],
                    "z": 0,
                },
                # 终点坐标[x, y, z]
                "to": {
                    "x": wall_list[wall_index][2],
                    "y": wall_list[wall_index][3],
                    "z": 0,
                },
                # 墙厚
                "thickness": thickness,
                # 墙高（从楼板到墙顶的距离z）
                "height": 2.60,
                "type": "generic",
                # 是否为承重墙，true / false，默认为true
                "bearing": is_bearing,
                "id": wallIds[wall_index]
            }
            room_json_str['floorPlanInfo']['walls'].append(wall_info)

        return room_json_str

    def _dump_window_info(self, room_json_str, window_list):
        # add window info
        for window_index in range(len(window_list)):
            x0 = window_list[window_index][0]
            y0 = window_list[window_index][1]
            x1 = window_list[window_index][2]
            y1 = window_list[window_index][3]

            window_width = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
            middle_x = 0.5 * (x0 + x1)
            middle_y = 0.5 * (y0 + y1)
            thickness = 0.24
            if window_list[window_index][4] < self.max_wall_thickness:
                thickness = 0.12

            window_info = {
                # 窗
                "type": "window",
                # 长a，宽b，高c (NOTE: 宽，高用的默认值）
                "size": {
                    "l": window_width,
                    "w": thickness,
                    "h": 1.5,
                },
                # 底面中心点坐标[x, y, z]
                "position": {
                    "x": middle_x,
                    "y": middle_y,
                    "z": 0.9,
                },

            }
            # set window type
            if window_list[window_index][5] != 1:
                if window_width < 0.8:
                    window_info["type"] = "normal_window"
                    window_info["size"]["h"] = 1.2
                elif window_width < 2.4:
                    window_info["type"] = "large_window"
                else:
                    window_info["type"] = "floor_based_window"
                    window_info["size"]["h"] = 2.1
                    window_info["position"]["z"] = 0.0  # 离地高度

            cur_window = window_list[window_index][-1]
            window_info["host_wall_indices"] = [wall.id for wall in cur_window.host_walls]  # host wall
            room_json_str['floorPlanInfo']['windows'].append(window_info)

        return room_json_str

    def _dump_door_info(self, room_json_str, door_list):
        # add door info
        for door_index in range(len(door_list)):
            x0 = door_list[door_index][0]
            y0 = door_list[door_index][1]
            x1 = door_list[door_index][2]
            y1 = door_list[door_index][3]
            door_width = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
            middle_x = 0.5 * (x0 + x1)
            middle_y = 0.5 * (y0 + y1)
            thickness = 0.24
            if door_list[door_index][5] < self.max_wall_thickness:
                thickness = 0.12

            door_info = {
                # 门
                "type": "door",
                # 长a，宽b，高c (NOTE: 宽，高用的默认值）
                "size": {
                    "l": door_width,
                    "w": thickness,
                    "h": 2.1,
                },
                # 底面中心点坐标[x, y, z]
                "position": {
                    "x": middle_x,
                    "y": middle_y,
                    "z": 0,
                }  # , "swing": door_list[door_index][7],
            }

            # check door type and door info
            if door_list[door_index][6] == 1:
                door_info["type"] = "double_swing_door"  # 双开门
            elif door_list[door_index][6] == 3:  # 移门
                door_info["type"] = "double_sliding_door"  # 滑门
            else:
                door_info["type"] = "single_door"  # 单开门

            cur_door = door_list[door_index][-1]
            door_info["host_wall_indices"] = [wall.id for wall in cur_door.host_walls]
            room_json_str['floorPlanInfo']['doors'].append(door_info)
        return room_json_str

    # dump the results.
    def dump_room_design_json(self, measuring_scale_ratio=-1.0):
        if self.wall_builder_obj is None:
            return False

        # 初始化一些数据
        self._init_data_()

        # 数据转换
        wall_list, wall_id_positions_map = self._convert_wall_lines(self.wall_builder_obj.all_wall_lines)
        window_list = self._convert_opening_lines(self.wall_builder_obj.all_opening_lines)
        door_list, single_swing_door_list = self._convert_doors(self.wall_builder_obj.all_door_lines)

        # 计算
        ratio = measuring_scale_ratio
        if ratio < 0:
            ratio = self._calc_default_measure_scale_ratio(single_swing_door_list, self.max_wall_thickness)

        # update the door list with ratio
        for door_index in range(len(door_list)):
            for i in range(0, 4):
                door_list[door_index][i] = door_list[door_index][i] * ratio

        # update the window list with ratio
        for window_index in range(len(window_list)):
            for i in range(0, 4):
                window_list[window_index][i] = window_list[window_index][i] * ratio

        # update the wall list with ratio
        for wall_index in range(len(wall_list)):
            for i in range(0, 5):
                wall_list[wall_index][i] = wall_list[wall_index][i] * ratio

        # header information.
        room_json_str = self._dump_room_info_header()
        room_json_str = self._dump_wall_info(room_json_str, wall_list, wall_id_positions_map, ratio)
        room_json_str = self._dump_window_info(room_json_str, window_list)
        room_json_str = self._dump_door_info(room_json_str, door_list)
        return room_json_str
