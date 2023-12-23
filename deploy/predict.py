import os

from utils import cv2_read_image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
cur_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(cur_folder_path)

from OFA_OCR.ikea_ocr import IKEA_OCR_Detect
from OFA_OCR.ocr_utils import get_boundaryDirection_yValue_ocrPoints, get_corner_points, get_ocr_ratio
from WallBuilder import *

from options import parse_args
from model import Model

from debug_utils import *

from imagetransform import *


class Predict(object):
    def __init__(self, options, model_config_path='', model_base_config_path='', ocr_model_path="", yolo_model_path=""):
        
        self.model_config_path = model_config_path
        self.model_base_config_path = model_base_config_path
        self.ocr_model_path = ocr_model_path
        self.yolo_model_path = yolo_model_path
        
        self.options = options
        
        self.model_base = None
        
        self.model = None
        
        self.scale_model = None
        
        self.all_heatmap_data = None
        
        self.image_transform = ImageTransform(self.options)
        
        self.room_type_predict_obj = None
        
        super(Predict, self).__init__()
        
        self._init_model()
    
    def _init_model(self):
        print("cuda.is_available", torch.cuda.is_available())
        if self.options.gpuFlag == 1:
            if not torch.cuda.is_available():
                self.options.gpuFlag = 0
        self.model = Model(self.options)
        self.model_base = Model(self.options)
        if self.options.gpuFlag == 1:
            self.model.load_state_dict(torch.load(self.model_config_path))
            self.model = self.model.cuda()
            self.model_base.load_state_dict(torch.load(self.model_base_config_path))
            self.model_base = self.model_base.cuda()
        else:
            self.model.load_state_dict(torch.load(self.model_config_path, map_location='cpu'))
            self.model_base.load_state_dict(torch.load(self.model_base_config_path, map_location='cpu'))
        
        print("~~~~~load scale model~~~~~")
        # self.scale_model = IKEA_OCR_Detect(self.ocr_model_path, self.yolo_model_path)
        print("~~~~~scale model over~~~~~")
        self.debug_util_obj = DebugInfo(self.options)  # 一些debug函数封装的对象
    
    def pre_process_image(self, img_file_path, type):
        return self.image_transform.transform_image(img_file_path, type)
    
    def generate_heatmap_data(self, img_file_path, type):
        try:
            # 1.图片预处理
            start = time.time()
            org_image, image = self.pre_process_image(img_file_path, type)
            if org_image is None or image is None:
                return "Error occur when loading image file."
            
            if self.options.debugFlag == 1:
                img_file_name = os.path.split(img_file_path)[1]
                output_dir = self.options.outputDir
                res_folder_path = os.path.join(output_dir, os.path.splitext(img_file_name)[0])
                self.options.res_folder_path = res_folder_path
                if not os.path.exists(res_folder_path):
                    os.makedirs(res_folder_path)
                self.debug_util_obj.save_floorplan_imag(org_image, res_folder_path=res_folder_path)
            
            # 1.1 pre-process the image.
            image = (image.astype(np.float32) / 255 - 0.5)
            
            # debug
            if self.options.debugFlag == 1:
                self.debug_util_obj.save_floorplan_imag_with_name(
                    image * 255, res_folder_path=res_folder_path,
                    name="PreProcessFullImage_512*512"
                )
            image = image.transpose((2, 0, 1))
            image_tensor = torch.Tensor(image)
            image_tensor = image_tensor.view((1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]))
            
            if self.options.gpuFlag == 1:
                image_tensor = image_tensor.cuda()
            
            print("prepare data cost: {0}".format(time.time() - start))
            
            start = time.time()
            # 2. predict the results.
            corner_pred = self.model(image_tensor)  # 后训练
            corner_base_pred = self.model_base(image_tensor)  # 基本模型
            
            print("predict model cost {0}".format(time.time() - start))
            
            # debug
            if self.options.debugFlag == 1:
                self.debug_util_obj.save_corner_heatmaps_img(
                    corner_pred[0],
                    self.image_transform,
                    res_folder_path=self.options.res_folder_path,
                    corner_base_pred=corner_pred[0]
                )
            
            # 3. get the prediction data.
            start = time.time()
            # 3.1基模
            corner_base_heatmaps = corner_base_pred[0].detach().cpu().numpy()
            corner_base_heatmaps = corner_base_heatmaps[0]
            # 3.2 标模
            corner_heatmaps = corner_pred[0].detach().cpu().numpy()
            corner_heatmaps = corner_heatmaps[0]
            
            all_heatmap_data = []
            for i in range(corner_base_heatmaps.shape[2]):
                if i < 4:  # 前4个通道预测斜墙
                    cur_heatmap = corner_heatmaps[:, :, i]
                else:  # 后4个用基模
                    cur_heatmap = corner_base_heatmaps[:, :, i]
                if self.image_transform is not None:
                    # heatmap图片尺寸适配
                    cur_heatmap = self.image_transform.mapping_2_original_image_size(cur_heatmap)
                all_heatmap_data.append(cur_heatmap)
            self.all_heatmap_data = all_heatmap_data
            print("generate heatmap cost {0}".format(time.time() - start))
            return all_heatmap_data
        except:
            raise
    
    # if measuring_scale = -1.0, use default measuring scale ratio.
    def predict(self, img_file_path, measuring_scale=-1.0, type='url'):
        """
      
        Args:
          img_file_path: 图片地址
          measuring_scale:
          type: 当type是url时就从url下载,其他任意值时为本地图片
    
        Returns:
    
        """
        try:
            # 1. calculate the heatmap.在self.all_heatmap_data中有
            all_heatmap = self.generate_heatmap_data(img_file_path, type)
            
            start = time.time()
            
            wall_builder = Builder(self.options)
            org_image = cv2_read_image(img_file_path, type)
            
            # 2. build floorplan json data.
            res = wall_builder.build_floorplan_json(
                org_image, self.all_heatmap_data,
                measuring_scale_ratio=measuring_scale
            )
            
            # 3. 打印信息
            print("post process Builder function cost {0}".format(time.time() - start))
            return res
        except Exception as e:
            raise e
    
    def _dump_room_info_header(self, ratio, points, value):
        room_json_str = {
            # 中间格式版本号，当前版本号1.0
            "version": "1.0",
            "meta": {
                "unit": {
                    # 长度单位 m，cm，mm, ft 默认 m
                    "length": "m",
                    "ratio": ratio,
                    "points": points,
                    "value": value
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
    
    def ratio_predict(self, image):
        width, height = image.size
        json_res = self._dump_room_info_header(-1, [], 0)
        start = time.time()
        try:
            res = self.scale_model.ikea_ocr(image)  # ocr
            cropped_img, boundary_idx, y_value, ocr_point_list = get_boundaryDirection_yValue_ocrPoints(
                image,
                res
            )
            yolo_bboxs = self.scale_model.get_yolo_detect_bbox(cropped_img)
            corner_point_map = get_corner_points(cropped_img, yolo_bboxs)
            print("yolo_bboxs:", yolo_bboxs)
            up_ratios = get_ocr_ratio(boundary_idx, ocr_point_list, corner_point_map, y_value)
            print("up_ratios:", up_ratios)
            print("post process ratio predict function cost {0}".format(time.time() - start))
            if len(up_ratios) > 0:
                json_res = self._dump_room_info_header(
                    up_ratios[5],
                    [up_ratios[1] - int(width / 2), int(height / 2) - up_ratios[3],
                        up_ratios[2] - int(width / 2), int(height / 2) - up_ratios[3]],
                    up_ratios[4]
                )
            return 1, json_res
        except:
            print("post process ratio predict function cost {0}".format(time.time() - start))
            return 0, json_res


# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    ocr_model_path = 'checkpoints/ocr_general_clean.pt'
    yolo_model_path = 'checkpoints/floorplan_best.pt'
    folder_path = os.path.dirname(os.path.abspath(__file__))
    imagePath = "0a0eccef-2277-4da0-9fe1-7277299af870.png"  # 这里填图片名称
    img_file_path = os.path.join(folder_path, "OFA_OCR/check_data/" + imagePath)
    args = parse_args()
    args.outputDir = "check/detectResult"
    args.debugFlag = 1
    model_config_path = 'checkpoints/checkpoint_940.pth'
    model_base_config_path = 'checkpoints/checkpoint.pth'
    predictor = Predict(args, model_config_path, model_base_config_path, ocr_model_path, yolo_model_path)
    
    # image = Image.open(requests.get(
    #     "https://henry-search.oss-cn-hangzhou.aliyuncs.com/im2fp/deploy/test_a/0ad81247-8ac5-4287-ba1e-ea500e113298%20(1).jpg",
    #     stream=True).raw) # 从oss上下载图片测试
    
    # flag,ratio_res = predictor.ratio_predict(image)
    # print("up_ratios:", ratio_res)
    
    res = predictor.predict(img_file_path, type="111")  # type=url是从ossUrl读数据
    print("floorplan:", res)
    # if flag==1:
    #     res["meta"] = ratio_res["meta"]
    print("floorplan:", json.dumps(res))
