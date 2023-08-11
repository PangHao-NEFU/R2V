import os
import numpy as np
import cv2
import copy
import torch

from utils import *
from Primitive import *

# from WallBuilderFloorplan import *

class DebugInfo(object):
    def __init__(self, options):

        self.options = options

        super(DebugInfo, self).__init__()

    def save_floorplan_imag(self, image, res_folder_path):
        if not os.path.exists(res_folder_path):
            os.makedirs(res_folder_path)

        cv2.imwrite(os.path.join(res_folder_path, "floorplan.png"), image)

    def save_floorplan_imag_with_name(self, image, res_folder_path,name):
        if not os.path.exists(res_folder_path):
            os.makedirs(res_folder_path)

        cv2.imwrite(os.path.join(res_folder_path, name+".png"), image)

    def save_corner_heatmaps_img(self, corner_tensor, img_transfor_obj = None, res_folder_path=r"D:\tmp", sample_index = 0,corner_base_pred=None):
        try:
            if not os.path.exists(res_folder_path):
                os.makedirs(res_folder_path)

            res_folder_path = os.path.join(res_folder_path, "heatmaps")
            if not os.path.exists(res_folder_path):
                os.makedirs(res_folder_path)

            corner_heatmaps = corner_tensor.detach().cpu().numpy()
            corner_heatmaps = copy.copy(corner_heatmaps)

            shape = corner_heatmaps.shape
            all_cur_heatmaps = []
            for batchIndex in range(shape[0]):
                cur_img_heatmaps = corner_heatmaps[batchIndex]
                for i in range(cur_img_heatmaps.shape[2]):
                    cur_heatmap = cur_img_heatmaps[:,:,i]
                    if i < 13:
                        cur_heatmap_file_path = os.path.join(res_folder_path, "junction_heatmap_{0}.png".format(i+1))
                        cur_heatmap *= 255
                    elif i < 8 + 13:
                        cur_heatmap_file_path = os.path.join(res_folder_path, "opening_heatmap_{0}.png".format(i + 1 - 13))
                        cur_heatmap *= 100
                    elif i < 8 + 13 + 28 + 4:
                        cur_heatmap_file_path = os.path.join(res_folder_path, "door_heatmap_{0}.png".format(i + 1 - 13 - 8))
                        cur_heatmap *= 170
                    else:
                        cur_heatmap_file_path = os.path.join(res_folder_path, "icon_heatmap_{0}.png".format(i + 1 - 8 - 13 - 28 - 4))
                        cur_heatmap *= 255
                    cur_heatmap.astype(np.int8)

                    if img_transfor_obj is not None:
                        cur_heatmap = img_transfor_obj.mapping_2_original_image_size(cur_heatmap)
                    cv2_write_image_light(cur_heatmap, cur_heatmap_file_path)
                    if 13>i:
                     all_cur_heatmaps.append(cur_heatmap)
            temp = all_cur_heatmaps[len(all_cur_heatmaps)-1]
            for i in range(len(all_cur_heatmaps)-1):
                temp+=all_cur_heatmaps[i]
            cv2_write_image_light(temp, os.path.join(res_folder_path, "all_cur_heatmaps.png"))
        except Exception as err:
            print(err)


# 用来保存WallBuilder中的data。
class WallBuilderDataDump(object):
    def __init__(self, wall_builder_obj):

        self.wall_builder_obj = wall_builder_obj

        super(WallBuilderDataDump, self).__init__()

    def dump_data(self):
        if self.wall_builder_obj is None:
            return False

        file_path = os.path.join(self.wall_builder_obj.options.res_folder_path, "Floorplan.png")
        if not os.path.exists(self.wall_builder_obj.options.res_folder_path):
            os.makedirs(self.wall_builder_obj.options.res_folder_path)
        cv2_write_image(self.wall_builder_obj.floor_plan_img_data, file_path)
        # self.wall_builder_obj.floor_plan_img_data = np.ones(
        #     (self.wall_builder_obj.floor_plan_img_height, self.wall_builder_obj.floor_plan_img_width, 3),
        #     np.uint8) * 255

        back_ground_img = self.wall_builder_obj.floor_plan_img_data.copy()
        back_ground_img = (back_ground_img.astype(np.float32) - 100)
        self._draw_wall_lines(self.wall_builder_obj.all_wall_lines, line_width=2,
                              back_ground_img=back_ground_img,
                              rgb_color=[255, 0, 0])
        # back_ground_img = self.wall_builder_obj.floor_plan_img_data.copy()
        self._draw_wall_lines_range_box(self.wall_builder_obj.all_wall_lines,
                                        back_ground_img=back_ground_img,
                                        rgb_color=[255, 0, 0])
        # back_ground_img = self.wall_builder_obj.floor_plan_img_data.copy()
        self._draw_wall_lines(self.wall_builder_obj.all_opening_lines,
                              back_ground_img=back_ground_img,
                              rgb_color=[0, 255, 0])

        # back_ground_img = self.wall_builder_obj.floor_plan_img_data.copy()
        self._draw_door_lines(self.wall_builder_obj.all_door_lines,
                              back_ground_img=back_ground_img)

        # back_ground_img = self.wall_builder_obj.floor_plan_img_data.copy()
        self._draw_wall_points(self.wall_builder_obj.all_wall_points, file_name="WallPoints.png",
                               back_ground_img=back_ground_img,
                               rgb_color=[0, 255, 0])

        # back_ground_img = self.wall_builder_obj.floor_plan_img_data.copy()
        self._draw_wall_points(self.wall_builder_obj.all_opening_points, file_name="OpeningPoints.png",
                               # line_width=2,
                               back_ground_img=back_ground_img,
                               rgb_color=[0, 0, 255])

        # back_ground_img = self.wall_builder_obj.floor_plan_img_data.copy()
        self._draw_wall_points(self.wall_builder_obj.all_door_points, file_name="DoorPoints.png",
                               # line_width=2,
                               back_ground_img=back_ground_img,
                               rgb_color=[255, 0, 0])

        self._create_summary_results()

    def _draw_wall_points(self, all_wall_points, line_width=8,
                          back_ground_img=None, rgb_color=[0, 0, 255],
                          file_name="wall_point_selected.png"):

        if not os.path.exists(self.wall_builder_obj.options.res_folder_path):
            os.makedirs(self.wall_builder_obj.options.res_folder_path)
        img_mask = None
        if back_ground_img is None:
            img_mask = np.ones((self.wall_builder_obj.floor_plan_img_height, self.wall_builder_obj.floor_plan_img_width, 3), np.uint8) * 0
        else:
            img_mask = back_ground_img

        line_color = np.random.rand(3) * 255
        line_color[0] = rgb_color[0]
        line_color[1] = rgb_color[1]
        line_color[2] = rgb_color[2]

        for cur_wall_point in all_wall_points:
            color = (0, 255, 0)

            un_fit_directions = cur_wall_point.get_noncomplete_direction()
            if len(un_fit_directions) > 0:
                color = (0, 255, 255)

            x, y = cur_wall_point.x, cur_wall_point.y
            img_mask[max(y - line_width, 0):min(y + line_width, self.wall_builder_obj.floor_plan_img_height - 1),
            max(x - line_width, 0):min(x + line_width, self.wall_builder_obj.floor_plan_img_width - 1)] = line_color
            # cv2.putText(img_mask, (str(x)+','+str(y)), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color)

        file_path = self._get_file_path(file_name)
        cv2_write_image(img_mask, file_path)

    def _get_file_path(self, file_name):
        return os.path.join(self.wall_builder_obj.options.res_folder_path, "{0}".format(file_name))

    def _draw_wall_lines_range_box(self, wall_lines, back_ground_img=None, rgb_color=[255, 0, 0]):
        if not os.path.exists(self.wall_builder_obj.options.res_folder_path):
            os.makedirs(self.wall_builder_obj.options.res_folder_path)

        image = None
        if back_ground_img is None:
            image = np.ones((self.wall_builder_obj.floor_plan_img_height, self.wall_builder_obj.floor_plan_img_width, 3), np.uint8) * 0
        else:
            image = back_ground_img

        line_color = np.random.rand(3) * 255
        line_color[0] = rgb_color[1]
        line_color[1] = rgb_color[2]
        line_color[2] = rgb_color[0]
        for i in range(len(wall_lines)):
            wall_line = wall_lines[i]
            if wall_line.line_dim() == 0:  # 水平
                boundary_range_box = [wall_line.start_point.x, wall_line.boundary_range_box[1], wall_line.end_point.x,
                                      wall_line.boundary_range_box[3]]
            elif wall_line.line_dim() == 1:
                boundary_range_box = [wall_line.boundary_range_box[0], wall_line.start_point.y,
                                      wall_line.boundary_range_box[2],
                                      wall_line.end_point.y]
            self._draw_range_box(boundary_range_box, image, line_color)
            if wall_line.line_dim() == -1:
                cv2.line(image, (wall_line.start_point.x, wall_line.start_point.y), (wall_line.end_point.x, wall_line.end_point.y), line_color, 2)


    def _draw_wall_lines(self, wall_lines, line_width=2, back_ground_img=None,
                         file_name="wall_lines_selected.png", rgb_color=[255, 0, 0], bay_window_color=[127, 127, 0]):
        if not os.path.exists(self.wall_builder_obj.options.res_folder_path):
            os.makedirs(self.wall_builder_obj.options.res_folder_path)

        image = None
        if back_ground_img is None:
            image = np.ones((self.wall_builder_obj.floor_plan_img_height, self.wall_builder_obj.floor_plan_img_width, 3), np.uint8) * 0
        else:
            image = back_ground_img

        # line_color = np.array([rgb_color[1], rgb_color[2], rgb_color[0]])

        bay_window_color = np.array([bay_window_color[1], bay_window_color[2], bay_window_color[0]])

        for i in range(len(wall_lines)):
            line_color = np.random.rand(3) * 255
            wall_line = wall_lines[i]
            point_1 = wall_line.start_point
            point_2 = wall_line.end_point

            line_dim = self.wall_builder_obj.calc_line_dim(point_1, point_2)

            fixedValue = int(round((point_1.y + point_2.y) / 2)) if line_dim == 0 else int(
                round((point_1.x + point_2.x) / 2))
            minValue = int(round(min(point_1.x, point_2.x))) if line_dim == 0 else int(round(min(point_1.y, point_2.y)))
            maxValue = int(round(max(point_1.x, point_2.x))) if line_dim == 0 else int(round(max(point_1.y, point_2.y)))

            draw_color = line_color
            # for bay window color
            if isinstance(wall_line, OpeningLine) and point_1.type_category > 0:
                draw_color = bay_window_color

            if line_dim == 0:
                image[max(fixedValue - line_width, 0):min(fixedValue + line_width, self.wall_builder_obj.floor_plan_img_height),
                max(minValue, 0):min(maxValue + 1, self.wall_builder_obj.floor_plan_img_width), :] = draw_color
            elif line_dim == 1:
                image[max(minValue, 0):min(maxValue + 1, self.wall_builder_obj.floor_plan_img_height),
                max(fixedValue - line_width, 0):min(fixedValue + line_width, self.wall_builder_obj.floor_plan_img_width), :] = draw_color
            else:
                cv2.line(image, (point_1.x, point_1.y), (point_2.x, point_2.y), (int(draw_color[1]), int(draw_color[2]), int(draw_color[0])), 2)
                # cv2.putText(image, str(wall_line.id),(point_1.x, point_1.y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (int(draw_color[1]), int(draw_color[2]), int(draw_color[0])))

            if wall_line.actual_space_length > 0:
                x = int(0.5 * (point_1.x + point_2.x))
                y = int(0.5 * (point_1.y + point_2.y))
                # cv2.putText(image, str(int(wall_line.actual_space_length * 1000.0)), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))

        file_path = self._get_file_path(file_name)
        cv2_write_image(image, file_path)
        return image

    def _draw_range_box(self, range_box, image_data, color):
        min_x = max(range_box[0], 0)
        max_x = min(range_box[2], self.wall_builder_obj.floor_plan_img_width - 1)
        min_y = max(range_box[1], 0)
        max_y = min(range_box[3], self.wall_builder_obj.floor_plan_img_height - 1)

        image_data[min_y, min_x:max_x, :] = color
        image_data[min_y:max_y, min_x, :] = color
        image_data[max_y, min_x:max_x, :] = color
        image_data[min_y:max_y, max_x, :] = color

    def _draw_door_lines(self, door_lines,
                         line_width=2,
                         back_ground_img=None,
                         file_name="wall_lines_selected.png"):

        if not os.path.exists(self.wall_builder_obj.options.res_folder_path):
            os.makedirs(self.wall_builder_obj.options.res_folder_path)

        image = None
        if back_ground_img is None:
            image = np.ones((self.wall_builder_obj.floor_plan_img_height, self.wall_builder_obj.floor_plan_img_width, 3), np.uint8) * 0
        else:
            image = back_ground_img

        for i in range(len(door_lines)):
            door_line = door_lines[i]

            point_1 = door_line.start_point
            point_2 = door_line.end_point

            line_color = np.random.rand(3) * 255
            # 单开门
            if point_1.type_category == 0:
                line_color[0] = 0
                line_color[1] = 127
                line_color[2] = 255
            elif point_1.type_category == 1:  # 双开门
                line_color[0] = 255
                line_color[1] = 0
                line_color[2] = 127
            elif point_1.type_category == 2:  # 门窗一体
                line_color[0] = 64
                line_color[1] = 0
                line_color[2] = 128
            else:
                line_color[0] = 255
                line_color[1] = 127
                line_color[2] = 0

            line_dim = self.wall_builder_obj.calc_line_dim(point_1, point_2)

            fixedValue = int(round((point_1.y + point_2.y) / 2)) if line_dim == 0 else int(
                round((point_1.x + point_2.x) / 2))
            minValue = int(round(min(point_1.x, point_2.x))) if line_dim == 0 else int(round(min(point_1.y, point_2.y)))
            maxValue = int(round(max(point_1.x, point_2.x))) if line_dim == 0 else int(round(max(point_1.y, point_2.y)))

            # 平行X轴
            if line_dim == 0:
                image[max(fixedValue - line_width, 0):min(fixedValue + line_width, self.wall_builder_obj.floor_plan_img_height),
                minValue:maxValue + 1, :] = line_color

                if point_1.type_category < 3:
                    # 在下面
                    if point_1.type_sub_category in [3, 7]:
                        image[max(fixedValue, 0):min(fixedValue + 15, self.wall_builder_obj.floor_plan_img_height),
                        minValue + 5:minValue + 5 + line_width, :] = line_color
                    else:
                        # 在上面
                        image[max(fixedValue - 15, 0):max(fixedValue, 0),
                        minValue + 5:minValue + 5 + line_width, :] = line_color
            elif line_dim == 1:
                image[minValue:maxValue + 1,
                max(fixedValue - line_width, 0):min(fixedValue + line_width, self.wall_builder_obj.floor_plan_img_width), :] = line_color

                if point_1.type_category < 3:
                    # 在右边
                    if point_1.type_sub_category in [0, 7]:
                        image[minValue + 5:minValue + 5 + line_width,
                        max(fixedValue, 0):min(fixedValue + 15, self.wall_builder_obj.floor_plan_img_width), :] = line_color
                    else:
                        # 在左边
                        image[minValue + 5:minValue + 5 + line_width, max(fixedValue - 15, 0):max(fixedValue, 0),
                        :] = line_color
            elif line_dim == -1:
                cv2.line(image, (point_1.x, point_1.y), (point_2.x, point_2.y),
                         (255, 127,0 ), 2)
                cv2.putText(image, str(point_1.x)+","+str(point_1.y), (point_1.x, point_1.y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 127,0))

        file_path = self._get_file_path(file_name)
        cv2_write_image(image, file_path)
        return image

    def _create_summary_results(self):
        import pandas as pd
        df = pd.DataFrame()
        df["Wall"] = [len(self.wall_builder_obj.all_wall_lines)]

        bay_window_number = 0
        for cur_window in self.wall_builder_obj.all_opening_lines:
            if cur_window.start_point.type_category > 0:
                bay_window_number += 1
        df["Windows"] = [len(self.wall_builder_obj.all_opening_lines) - bay_window_number]
        df["BayWindows"] = [bay_window_number]

        df["Door"] = [len(self.wall_builder_obj.all_door_lines)]

        single_door_number = 0
        double_door_number = 0
        slide_door_number = 0
        door_window_number = 0
        for cur_door in self.wall_builder_obj.all_door_lines:
            door_type = cur_door.get_door_type()
            if door_type == 0:
                single_door_number += 1
            elif door_type == 1:
                double_door_number += 1
            elif door_type == 2:
                door_window_number += 1
            else:
                slide_door_number += 1

        df["SingleDoor"] = [single_door_number]
        df["DoubleDoor"] = [double_door_number]
        df["DoorWindow"] = [door_window_number]
        df["SlideDoor"] = [slide_door_number]
        file_path = os.path.join(self.wall_builder_obj.options.res_folder_path, "summary.csv")
        df.to_csv(file_path, index=False)


class WallBuilderFloorplanDataDump(WallBuilderDataDump):
    def __init__(self, wall_builder_obj):

        super(WallBuilderFloorplanDataDump, self).__init__(wall_builder_obj)

    def _get_file_path(self, file_name):

        completeness_flag = self.wall_builder_obj.floorplan_status_flag
        connectivity_status_flag = self.wall_builder_obj.connectivity_status_flag
        scale_status_flag = self.wall_builder_obj.dimension_status_flag
        pipeline_issue_flag = self.wall_builder_obj.has_pipeline_flag

        if completeness_flag and connectivity_status_flag and scale_status_flag and not pipeline_issue_flag:
            file_path = os.path.join(self.wall_builder_obj.options.res_folder_path, "Good_{0}".format(file_name))
        else:
            file_name_format = "Bad_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}".format("complete", completeness_flag,
                                                                            "connect", connectivity_status_flag,
                                                                            "scale", scale_status_flag,
                                                                            "pipeline", pipeline_issue_flag, file_name)
            file_path = os.path.join(self.wall_builder_obj.options.res_folder_path, file_name_format)

        return file_path