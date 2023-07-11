import numpy as np

global_id = 0


class WallPoint(object):
    def __init__(self, type_category, type_sub_category, x, y, heat_map_value=0.4):

        global global_id
        self.id = global_id
        global_id += 1

        # the type include I, L, T, X
        #                  0, 1, 2, 3
        # http://art-programmer.github.io/floorplan-transformation/paper.pdf
        self.type_category = type_category
        # each type_category has some sub categories.
        self.type_sub_category = type_sub_category

        # 有可能一个点会被判别成不同的category, 需要记录下来。
        self.candidate_categories = []

        # the coordinate of the wall point of the image.
        self.x = x
        self.y = y

        # 每个Junction都有不同的方向，在不同的方向上有不同的宽度。
        # [0, 1, 2, 3] 这位置顺序和点的方向相同。 下/左/上/右顺时针。
        # 0,2对应X值，1,3对应Y值
        self.up_boundary = [-1, -1, -1, -1]
        self.down_boundary = [-1, -1, -1, -1]

        # the heat map value of this wall point.
        self.heat_map_value = heat_map_value
        self.heat_map_scope_pixels = []

        self.x_visited_flag = False
        self.y_visited_flag = False

        self.wall_lines = []
        self.connect_points = []

        super(WallPoint, self).__init__()

    def get_connect_wall(self, other_point):
        for cur_wall_line in self.wall_lines:
            if other_point == cur_wall_line.start_point or other_point == cur_wall_line.end_point:
                return cur_wall_line
        return None

    def is_close_enough_point(self, other_point, target_direction, threshold):
        distance = np.abs(self.y - other_point.y) if target_direction == 1 else np.abs(self.x - other_point.x)
        return True if distance < threshold else False

    def set_boundary(self, boundary_value, direction, up_flag=False):
        if up_flag:
            self.up_boundary[direction] = boundary_value
        else:
            self.down_boundary[direction] = boundary_value

    # line_dim is 0 or 1
    def get_boundary(self, line_dim):
        if line_dim == 1:
            if self.up_boundary[0] > 0:
                return self.up_boundary[0], self.down_boundary[0]
            elif self.down_boundary[2] > 0:
                return self.up_boundary[2], self.down_boundary[2]
        else:
            if self.up_boundary[1] > 0:
                return self.up_boundary[1], self.down_boundary[1]
            elif self.down_boundary[3] > 0:
                return self.up_boundary[3], self.down_boundary[3]

        return -1, -1

    def fit_coordinate_by_boundary(self):
        try:
            up_boundary, bottom_boundary = self.get_boundary(0)
            if up_boundary > 0 and bottom_boundary > 0:
                self.y = 0.5 * (up_boundary + bottom_boundary)

            up_boundary, bottom_boundary = self.get_boundary(1)
            if up_boundary > 0 and bottom_boundary > 0:
                self.x = 0.5 * (up_boundary + bottom_boundary)
        except Exception as err:
            print(err)

    def add_connect_point(self, point):
        if point not in self.connect_points:
            self.connect_points.append(point)

    # 是否具备完备性，比如点的category类型为T型，它就需要和3个wall_line或者opening 相连。
    def is_completeness_point(self):
        if (self.type_category + 1) == len(self.wall_lines):
            return True
        else:
            return False

    # 从self出发，是以何种方式与connect point相连。
    def get_connect_type(self, connect_point):
        # 水平方向。
        if np.abs(self.x - connect_point.x) > np.abs(self.y - connect_point.y):
            # self在左， connect_point在右。相对于self point, 连接方式是3.
            if self.x - connect_point.x < 0:
                return 3
            else:
                return 1
        else:
            # 竖直方向。
            if self.y - connect_point.y < 0:
                # self在上，connect_point在下
                return 0
            else:
                return 2

    def repair_completeness(self, all_wall_lines, all_opening_points):
        new_points = []
        new_wall_lines = []
        must_direction = list(set([0, 1, 2, 3]).difference(set(self.get_forbidden_direction())))
        for cur_direction in must_direction:
            exist_flag = False
            for cur_wall_line in self.wall_lines:
                wall_line_relative_direction = cur_wall_line.calc_relative_direction(self)
                if wall_line_relative_direction == cur_direction:
                    exist_flag = True
                    break

            # 如果不存在，缺失了。需要采用一定的方法，额外的添加一条wall line来满足。
            # 当然，这一步是在算法的最后一步完成，采用的策略应该是保守的策略。
            # 该策略：找最近相交的wall line. 如果他们之间存在一个opening point. 如果存在，添加额外的一条wall line.
            if not exist_flag:
                dist = 10000000.0
                nearest_wall_line = None

                for wall_line in all_wall_lines:
                    wall_line_dim = wall_line.line_dim()

                    # 平行
                    if (wall_line_dim == 0 and (cur_direction == 1 or cur_direction == 3)) \
                            or (wall_line_dim == 1 and (cur_direction == 0 or cur_direction == 2)):
                        continue

                    # 是否有交点
                    if cur_direction == 0 or cur_direction == 2:
                        if (self.x > np.maximum(wall_line.start_point.x, wall_line.end_point.x)) \
                                or (self.x < np.minimum(wall_line.start_point.x, wall_line.end_point.x)):
                            continue

                        new_wall_point_y = wall_line.start_point.y
                        new_wall_point_x = self.x

                        # 不包括它自己
                        cur_dist = np.abs(self.y - new_wall_point_y)
                        if cur_dist == 0:
                            continue

                        if dist > cur_dist:
                            dist = cur_dist
                            nearest_wall_line = wall_line
                    else:
                        if (self.y > np.maximum(wall_line.start_point.y, wall_line.end_point.y)) \
                                or (self.y < np.minimum(wall_line.start_point.y, wall_line.end_point.y)):
                            continue

                        new_wall_point_y = self.y
                        new_wall_point_x = wall_line.start_point.x

                        cur_dist = np.abs(self.x - new_wall_point_x)
                        if cur_dist == 0:
                            continue

                        if dist > cur_dist:
                            dist = cur_dist
                            nearest_wall_line = wall_line

                if nearest_wall_line is None:
                    continue

                for cur_opening_point in all_opening_points:
                    if cur_direction == 0 or cur_direction == 2:

                        new_wall_point_y = nearest_wall_line.start_point.y
                        new_wall_point_x = self.x

                        if (cur_direction == 0) and (self.y < cur_opening_point.y < new_wall_point_y and (
                                np.abs(cur_opening_point.x - self.x) < 10)):
                            # find one opening point
                            new_wall_point = WallPoint(-1, -1, new_wall_point_x, new_wall_point_y)
                            new_points.append(new_wall_point)
                            new_wall_line = WallLine(self, new_wall_point)
                            new_wall_lines.append(new_wall_line)
                            break
                        elif (cur_direction == 2) and (self.y > cur_opening_point.y > new_wall_point_y and (
                                np.abs(cur_opening_point.x - self.x) < 10)):
                            # find one opening point
                            new_wall_point = WallPoint(-1, -1, new_wall_point_x, new_wall_point_y)
                            new_points.append(new_wall_point)
                            new_wall_line = WallLine(new_wall_point, self)
                            new_wall_lines.append(new_wall_line)
                            break
                    else:
                        new_wall_point_y = self.y
                        new_wall_point_x = nearest_wall_line.start_point.x
                        if (cur_direction == 1) and (new_wall_point_x < cur_opening_point.x < self.x and (
                                np.abs(cur_opening_point.y - self.y) < 10)):

                            # find one opening point
                            new_wall_point = WallPoint(-1, -1, new_wall_point_x, new_wall_point_y)
                            new_points.append(new_wall_point)
                            new_wall_line = WallLine(self, new_wall_point)
                            new_wall_lines.append(new_wall_line)
                            break
                        elif (cur_direction == 3) and (new_wall_point_x > cur_opening_point.x > self.x and (
                                np.abs(cur_opening_point.y - self.y) < 10)):
                            # find one opening point
                            new_wall_point = WallPoint(-1, -1, new_wall_point_x, new_wall_point_y)
                            new_points.append(new_wall_point)
                            new_wall_line = WallLine(new_wall_point, self)
                            new_wall_lines.append(new_wall_line)
                            break

        return new_wall_lines, new_points

    def is_feasible_direction(self, direction):
        all_directions = [0, 1, 2, 3]
        feasible_direction = list(set(all_directions).difference(self.get_forbidden_direction()))
        if direction in feasible_direction:
            return True
        else:
            return False

    def get_feasible_direction(self):
        all_directions = [0, 1, 2, 3]
        feasible_direction = list(set(all_directions).difference(self.get_forbidden_direction()))
        return feasible_direction

    # 是否存在不完备的方向。
    # 每个方向都有一天wall line相连。
    def get_noncomplete_direction(self):
        # 必须要满足的方向,[0, 1, 2, 3].difference(forbidden_direction)
        feasible_direction = self.get_feasible_direction()

        # 已经相连的points.
        connect_points = self.connect_points
        connect_directions = []
        for tmp_connect_point in connect_points:
            # tmp_wall_point与tmp_connect_point是以何种方式相连。
            cur_dir = self.get_connect_type(tmp_connect_point)
            connect_directions.append(cur_dir)

        # 某个方向上没有完备
        un_fit_directions = list(set(feasible_direction).difference(set(connect_directions)))
        return un_fit_directions

    # 当连接墙的时候，point不能和某些point相连，因为point是有方向的。
    def get_forbidden_direction(self):
        if self.type_category == 0:
            # 竖直方向的点，禁止横向的连线
            '''
                   0             2
                   。            |
                   |             。
            '''
            if self.type_sub_category == 0 or self.type_sub_category == 2:
                return [1, 3]
            else:
                # 水平方向的点，禁止竖直方向的连线。
                '''
                      1           3
                     ---。        。---
                '''
                return [0, 2]
        elif self.type_category == 1:
            if self.type_sub_category == 0:
                '''
                           ||
                           ||
                       ====== 
                '''
                return [0, 3]
            elif self.type_sub_category == 1:
                '''
                           ||
                           ||
                           ======                        
                '''
                return [0, 1]
            elif self.type_sub_category == 2:
                '''
                           ======        
                           ||
                           ||                
                '''
                return [1, 2]
            else:
                '''
                       ======        
                           ||
                           ||                
                '''
                return [2, 3]
        elif self.type_category == 2:
            if self.type_sub_category == 0:
                '''
                       ==========        
                           ||
                           ||                
                '''
                return [2]
            elif self.type_sub_category == 1:
                '''
                           ||
                           ||                 
                       ======      
                           ||
                           ||                
                '''
                return [3]
            elif self.type_sub_category == 2:
                '''
                           ||
                           ||                 
                       ==========                   
                '''
                return [0]
            else:
                '''
                           ||
                           ||                 
                           ========      
                           ||
                           ||                
                '''
                return [1]
        else:
            return []

    def get_direction(self):
        if self.type_sub_category in [0, 2]:
            direction = 1
        else:
            direction = 0

        return direction

    def is_line_start_point(self):
        if self.type_sub_category in [0, 3]:
            return True
        else:
            return False

    def is_line_end_point(self):
        if self.type_sub_category in [1, 2]:
            return True
        else:
            return False

    def get_heat_map_centroid(self):
        x_index = [value[0] for value in self.heat_map_scope_pixels]
        y_index = [value[1] for value in self.heat_map_scope_pixels]
        x_center = 0.5 * (np.max(x_index) + np.min(x_index))
        y_center = 0.5 * (np.max(y_index) + np.min(y_index))

        return x_center, y_center

    def get_heat_map_boundary(self):
        x_index = [value[0] for value in self.heat_map_scope_pixels]
        y_index = [value[1] for value in self.heat_map_scope_pixels]
        return np.max(x_index)-np.min(x_index),np.max(y_index)-np.min(y_index)


# DoorPoint
# type_category = 0: 单开门   type_sub_category: [0, 1], [2, 3], [4, 5], [6, 7]
# type_category = 1: 双开门
# type_category = 2: 门窗一体
# type_cateogyr = 3: 移门     type_sub_cateogyr: 0, 1, 2, 3
class DoorPoint(WallPoint):
    def __init__(self, type_category, type_sub_category, x, y, heat_map_value=0.4):

        super(DoorPoint, self).__init__(type_category, type_sub_category, x, y, heat_map_value=heat_map_value)

    def get_direction(self):
        # 对于移门
        if self.type_category == 3:
            return super(DoorPoint, self).get_direction()
        else:
            if self.type_sub_category in [0, 1, 4, 5]:
                direction = 1
            else:
                direction = 0

            return direction

    def is_line_start_point(self):
        if self.type_category == 3:
            return super(DoorPoint, self).is_line_start_point()
        else:
            if self.type_sub_category in [0, 1, 6, 7]:
                return True
            else:
                return False

    def is_line_end_point(self):
        if self.type_category == 3:
            return super(DoorPoint, self).is_line_end_point()
        else:
            if self.type_sub_category in [2, 3, 4, 5]:
                return True
            else:
                return False


class WallLine(object):
    def __init__(self, start_point, end_point, is_opening=False):

        global global_id
        self.id = global_id
        global_id += 1

        # is wall line or opening(door or windows)?
        self.is_opening = is_opening

        # the primitive contains two wall points.
        self.start_point = start_point
        self.end_point = end_point

        self.visit_flag = False

        # 最后需要判断Wall Line是否有相交。因为有可能会判断错误。
        self.intersect_wall_lines = []

        # Wall Line Boundary Position. [min_x, min_y, max_x, max_y]
        self.boundary_range_box = [-1, -1, -1, -1]

        # openings include windows and doors.
        self.openings = []
        # the host wall of an opening.
        self.host_walls = []

        self.connect_rooms = []

        self.is_outer_wall = False

        self.is_inclined_wall = False

        # 实际的长度， 考虑了比例尺的问题。
        # 这个不是像素级别的长度。
        self.actual_space_length = -1.0

        # 这里包含了每堵墙的端点位置。
        self.sub_sections = []

        super(WallLine, self).__init__()

    # 伪斜墙的边界计算
    def fine_tun_boundary_by_points(self, direction):
        if direction == 0:
            # 更新水平放上的厚度，获得point的top, bottom boundary.
            top_boundary, bottom_boundary = self.start_point.get_boundary(1)
            if bottom_boundary > 0:
                self.boundary_range_box[0] = bottom_boundary
            top_boundary, bottom_boundary = self.end_point.get_boundary(1)
            if top_boundary > 0:
                self.boundary_range_box[2] = top_boundary
        elif direction == 1:
            top_boundary, bottom_boundary = self.start_point.get_boundary(0)
            if bottom_boundary > 0:
                self.boundary_range_box[1] = bottom_boundary
            top_boundary, bottom_boundary = self.end_point.get_boundary(0)
            if top_boundary > 0:
                self.boundary_range_box[3] = top_boundary
        # elif direction == -1:

    def get_line_slope(self):
        return np.abs((self.end_point.y - self.start_point.y) / (self.end_point.x - self.start_point.x))

    def set_boundary_by_points(self, direction):
        if direction == 0:
            # 更新水平放上的厚度，获得point的top, bottom boundary.
            top_boundary, bottom_boundary = self.start_point.get_boundary(1)
            if bottom_boundary > 0:
                self.boundary_range_box[0] = bottom_boundary
            top_boundary, bottom_boundary = self.end_point.get_boundary(1)
            if top_boundary > 0:
                self.boundary_range_box[2] = top_boundary
        else:
            top_boundary, bottom_boundary = self.start_point.get_boundary(0)
            if bottom_boundary > 0:
                self.boundary_range_box[1] = bottom_boundary
            top_boundary, bottom_boundary = self.end_point.get_boundary(0)
            if top_boundary > 0:
                self.boundary_range_box[3] = top_boundary
    def set_thickness_boundary_by_neighbor_wall(self, direction, neighbor_wall):
        neighbor_wall_thickness = neighbor_wall.get_wall_thickness()
        if neighbor_wall_thickness < 0:
            return False

        if direction == 1:
            self.boundary_range_box[0] = self.start_point.x - int(neighbor_wall_thickness / 2)
            self.boundary_range_box[2] = self.start_point.x + int(neighbor_wall_thickness / 2)
        else:
            self.boundary_range_box[1] = self.start_point.y - int(neighbor_wall_thickness / 2)
            self.boundary_range_box[3] = self.start_point.y + int(neighbor_wall_thickness / 2)
        return True

    # 由于特殊原因，需要根据相邻墙设置墙的boundary。
    # direction: 要更新那个方向的boundary
    # neighbor_wall_thickness_direction：取那个方向的thickness去更新。
    def set_thickness_boundary_by_connect_wall(self, inner_wall_thickness, outer_wall_thickness):
        wall_dim = self.line_dim()

        diff_direction_walls = []

        same_direction_and_types_walls = []
        same_types_walls = []
        same_direction_walls = []

        connect_walls = [wall_line for wall_line in self.start_point.wall_lines]
        connect_walls.extend(self.end_point.wall_lines)
        for neighbor_wall in connect_walls:
            if neighbor_wall.id == self.id:
                continue

            if (neighbor_wall.is_outer_wall == self.is_outer_wall) and (neighbor_wall.line_dim() == wall_dim):
                same_direction_and_types_walls.append(neighbor_wall)
            elif neighbor_wall.is_outer_wall == self.is_outer_wall:
                same_types_walls.append(neighbor_wall)
            elif neighbor_wall.line_dim() == wall_dim:
                same_direction_walls.append(neighbor_wall)
            else:
                diff_direction_walls.append(neighbor_wall)

        for neighbor_wall in same_direction_and_types_walls:
            if neighbor_wall.get_wall_thickness() > 0:
                if wall_dim == 0:
                    self.boundary_range_box[1] = neighbor_wall.boundary_range_box[1]
                    self.boundary_range_box[3] = neighbor_wall.boundary_range_box[3]
                else:
                    self.boundary_range_box[0] = neighbor_wall.boundary_range_box[0]
                    self.boundary_range_box[2] = neighbor_wall.boundary_range_box[2]
                return True

        for neighbor_wall in same_types_walls:
            if self.set_thickness_boundary_by_neighbor_wall(wall_dim, neighbor_wall):
                return True

        for neighbor_wall in same_direction_walls:
            if neighbor_wall.get_wall_thickness() > 0:
                if wall_dim == 0:
                    self.boundary_range_box[1] = neighbor_wall.boundary_range_box[1]
                    self.boundary_range_box[3] = neighbor_wall.boundary_range_box[3]
                else:
                    self.boundary_range_box[0] = neighbor_wall.boundary_range_box[0]
                    self.boundary_range_box[2] = neighbor_wall.boundary_range_box[2]
                return True

        for neighbor_wall in diff_direction_walls:
            if self.set_thickness_boundary_by_neighbor_wall(wall_dim, neighbor_wall):
                return True
        return False

    def is_aligned(self):
        if -1 in self.boundary_range_box:
            return False
        else:
            return True

    def to_json_dict(self):
        data = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(self.boundary_range_box[0], self.boundary_range_box[1],
                                                               self.boundary_range_box[2], self.boundary_range_box[3],
                                                               "wall", 1, 1, self.get_wall_thickness())
        return data

    def fine_tune_wall_thickness(self, inner_wall_thickness, outer_wall_thickness):
        direction = self.line_dim()
        wall_thickness_limit = outer_wall_thickness if self.is_outer_wall else inner_wall_thickness
        if direction == 0:
            if self.boundary_range_box[1] > 0 and self.boundary_range_box[3] > 0:
                wall_thickness = np.abs(self.boundary_range_box[1] - self.boundary_range_box[3])
                diff_wall_thickness = wall_thickness - wall_thickness_limit
                self.boundary_range_box[1] += 0.5 * diff_wall_thickness
                self.boundary_range_box[3] -= 0.5 * diff_wall_thickness
            else:
                self.boundary_range_box[1] = self.start_point.y - 0.5 * wall_thickness_limit
                self.boundary_range_box[3] = self.start_point.y + 0.5 * wall_thickness_limit
        elif direction == 1:
            if self.boundary_range_box[0] > 0 and self.boundary_range_box[2] > 0:
                wall_thickness = np.abs(self.boundary_range_box[0] - self.boundary_range_box[2])
                diff_wall_thickness = wall_thickness - wall_thickness_limit
                self.boundary_range_box[0] += 0.5 * diff_wall_thickness
                self.boundary_range_box[2] -= 0.5 * diff_wall_thickness
            else:
                self.boundary_range_box[0] = self.start_point.x - 0.5 * wall_thickness_limit
                self.boundary_range_box[2] = self.start_point.x + 0.5 * wall_thickness_limit

    def get_wall_thickness(self):
        direction = self.line_dim()
        # 水平方向的厚度。
        if direction == 1:
            # max_y - max_y
            if self.boundary_range_box[0] < 0 or self.boundary_range_box[2] < 0:
                return -1
            else:
                return np.abs(self.boundary_range_box[0] - self.boundary_range_box[2])
        else:
            # max_x - min_x
            if self.boundary_range_box[1] < 0 or self.boundary_range_box[3] < 0:
                return -1
            else:
                return np.abs(self.boundary_range_box[1] - self.boundary_range_box[3])

    def get_wall_length(self):
        if self.line_dim() == 0:
            return np.abs(self.end_point.x - self.start_point.x)
        elif self.line_dim() == 1:
            return np.abs(self.end_point.y - self.start_point.y)
        elif self.line_dim() == -1:
            # 计算斜墙的length
            return np.sqrt((self.end_point.x - self.start_point.x) ** 2 + (self.end_point.y - self.start_point.y) ** 2)

    # 调节opening 的坐标。
    def fine_tune_openings_coordinate(self):
        direction = self.line_dim()
        if direction == 0:
            wall_left_limit = 0.0
            wall_right_limit = 0.0
            if len(self.openings) > 0:
                for tmp_wall in self.start_point.wall_lines:
                    if tmp_wall.id == self.id or tmp_wall.line_dim() != 1:
                        continue
                    wall_left_limit = tmp_wall.boundary_range_box[2]
                    break
                for tmp_wall in self.end_point.wall_lines:
                    if tmp_wall.id == self.id or tmp_wall.line_dim() != 1:
                        continue
                    wall_right_limit = tmp_wall.boundary_range_box[0]
                    break

            for opening in self.openings:
                opening.start_point.y = self.start_point.y
                opening.end_point.y = self.start_point.y

                opening.boundary_range_box[1] = self.boundary_range_box[1]
                opening.boundary_range_box[3] = self.boundary_range_box[3]

                opening.boundary_range_box[0] = opening.start_point.x
                opening.boundary_range_box[2] = opening.end_point.x

                # 限制Opening.start_point.x and Opening.end_point.x
                if opening.start_point.x < wall_left_limit:
                    opening.start_point.x = wall_left_limit
                if 0 < wall_right_limit < opening.end_point.x:
                    opening.end_point.x = wall_right_limit
        elif direction == 1:
            wall_down_limit = 0.0
            wall_up_limit = 0.0
            if len(self.openings) > 0:
                for tmp_wall in self.start_point.wall_lines:
                    if tmp_wall.id == self.id or tmp_wall.line_dim() != 0:
                        continue
                    wall_down_limit = tmp_wall.boundary_range_box[3]
                    break
                for tmp_wall in self.end_point.wall_lines:
                    if tmp_wall.id == self.id or tmp_wall.line_dim() != 0:
                        continue
                    wall_up_limit = tmp_wall.boundary_range_box[1]
                    break

            for opening in self.openings:
                opening.start_point.x = self.start_point.x
                opening.end_point.x = self.start_point.x

                opening.boundary_range_box[0] = self.boundary_range_box[0]
                opening.boundary_range_box[2] = self.boundary_range_box[2]

                opening.boundary_range_box[1] = opening.start_point.y
                opening.boundary_range_box[3] = opening.end_point.y

                if opening.start_point.y < wall_down_limit:
                    opening.start_point.y = wall_down_limit
                if 0 < wall_up_limit < opening.end_point.y:
                    opening.end_point.y = wall_up_limit
        elif direction == -1:
            wall_left_limit = 0.0
            wall_right_limit = 0.0
            wall_down_limit = 0.0
            wall_up_limit = 0.0
            if len(self.openings) > 0:
                for tmp_wall in self.start_point.wall_lines:
                    wall_left_limit = tmp_wall.boundary_range_box[2]
                    wall_down_limit = tmp_wall.boundary_range_box[3]
                    break
                for tmp_wall in self.end_point.wall_lines:
                    wall_right_limit = tmp_wall.boundary_range_box[0]
                    wall_up_limit = tmp_wall.boundary_range_box[1]
                    break

            for opening in self.openings:
                opening.start_point.x = self.start_point.x
                opening.end_point.x = self.start_point.x
                opening.start_point.y = self.start_point.y
                opening.end_point.y = self.start_point.y

                opening.boundary_range_box[1] = self.boundary_range_box[1]
                opening.boundary_range_box[3] = self.boundary_range_box[3]

                opening.boundary_range_box[0] = self.boundary_range_box[0]
                opening.boundary_range_box[2] = self.boundary_range_box[2]

                # 限制Opening.start_point.x and Opening.end_point.x
                if opening.start_point.x < wall_left_limit:
                    opening.start_point.x = wall_left_limit
                if 0 < wall_right_limit < opening.end_point.x:
                    opening.end_point.x = wall_right_limit
                # if opening.start_point.y < wall_down_limit:
                #     opening.start_point.y = wall_down_limit
                # if 0 < wall_up_limit < opening.end_point.y:
                #     opening.end_point.y = wall_up_limit

    def line_dim(self):
        threshold = 5  # 纯个人经验值 by henry.hao 2023.06.29
        if np.abs(self.end_point.x - self.start_point.x) > threshold and np.abs(
                self.end_point.y - self.start_point.y) > threshold:
            return -1
        # 平行X
        if np.abs(self.end_point.x - self.start_point.x) > np.abs(self.end_point.y - self.start_point.y):
            return 0
        else:
            return 1

    # 判断门/窗是否属于当前这条wall,支持斜墙上的门/窗计算
    def is_opening_on_wall(self, opening):
        direction = self.line_dim()
        if direction != opening.line_dim():
            return False

        if direction == 0:
            if np.abs(opening.start_point.y - self.start_point.y) > 10:
                return False

            if opening.start_point.x >= self.start_point.x and opening.end_point.x <= self.end_point.x:
                return True
            else:
                return False
        elif direction == 1:
            if np.abs(opening.start_point.x - self.start_point.x) > 10:
                return False

            if opening.start_point.y >= self.start_point.y and opening.end_point.y <= self.end_point.y:
                return True
            else:
                return False
        elif direction == -1:
            # 两个点的斜率和截距,没有对point做严格的排序，所以先忽略方向
            if self.start_point.x <= self.end_point.x:
                line_slope = (self.end_point.y - self.start_point.y) / (self.end_point.x - self.start_point.x)
                opening_slope = (opening.end_point.y - self.start_point.y) / (opening.end_point.x - self.start_point.x)
            else:
                line_slope = (self.start_point.y - self.end_point.y) / (self.start_point.x - self.end_point.x)
                opening_slope = (opening.start_point.y - self.end_point.y) / (opening.start_point.x - self.end_point.x)

            if np.abs(line_slope - opening_slope) > 0.5:
                return False
            if opening.get_wall_length() < self.get_wall_length():
                return True
            else:
                return False


    # wall lines 是否相交。
    def is_interest(self, other_wall_line):
        # 直线平行。
        cur_line_dim = self.line_dim()
        other_line_dim = other_wall_line.line_dim()
        if cur_line_dim == other_line_dim:
            return False

        # 不平行:
        if cur_line_dim == 0:
            if min(other_wall_line.start_point.y, other_wall_line.end_point.y) < self.start_point.y < max(
                    other_wall_line.start_point.y, other_wall_line.end_point.y):
                return True
            else:
                return False
        else:
            if min(other_wall_line.start_point.x, other_wall_line.end_point.x) < self.start_point.x < max(
                    other_wall_line.start_point.x, other_wall_line.end_point.x):
                return True
            else:
                return False

    # 计算相对的方向。从point出发，
    def calc_relative_direction(self, point):
        other_point = self.end_point if self.start_point == point else self.start_point
        line_dim = self.line_dim()
        # 平行X轴
        if line_dim == 0:
            if other_point.x > point.x:
                return 3
            else:
                return 1
        else:
            # 平行Y轴
            if other_point.y > point.y:
                return 0
            else:
                return 2


class DoorLine(WallLine):
    def __init__(self, start_point, end_point):
        super(DoorLine, self).__init__(start_point, end_point, is_opening=False)

    def get_door_type(self):
        if self.start_point is None:
            return -1

        return self.start_point.type_category

    def get_door_direction(self):
        if self.start_point is None:
            return -2

        line_dim = self.line_dim()
        if line_dim == -1:
            return -1
        elif line_dim == 0:
            if self.start_point.type_category < 3:
                if self.start_point.type_sub_category in [3, 7]:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            if self.start_point.type_category < 3:
                if self.start_point.type_sub_category in [0, 7]:
                    return 0
                else:
                    return 1
            else:
                return 0

    def to_json_dict(self):

        door_type = self.start_point.type_category
        door_direction = self.start_point.get_direction()
        data = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(self.boundary_range_box[0], self.boundary_range_box[1],
                                                               self.boundary_range_box[2], self.boundary_range_box[3],
                                                               "door", door_type, door_direction,
                                                               self.get_wall_thickness())

        # data = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(self.start_point.x, self.start_point.y,
        #                                                   self.end_point.x, self.end_point.y,
        #                                                   "door", door_type, door_direction)

        return data


class OpeningLine(WallLine):
    def __init__(self, start_point, end_point):
        super(OpeningLine, self).__init__(start_point, end_point, is_opening=True)

    def is_bay_window(self):
        if self.start_point.type_category == 1:
            return True
        else:
            return False

    def to_json_dict(self):
        data = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(self.boundary_range_box[0], self.boundary_range_box[1],
                                                               self.boundary_range_box[2], self.boundary_range_box[3],
                                                               "opening", 1, 1, self.get_wall_thickness())

        # data = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(self.start_point.x, self.start_point.y,
        #                                                   self.end_point.x, self.end_point.y,
        #                                                   "opening", 1, 1)
        return data
