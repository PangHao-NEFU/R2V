import torch
import os
import zipfile
from PIL import Image
import requests
import json


# https://pytorch.org/serve/custom_service.html
class ModelHandler(object):
    
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.model_base = None
    
    def initialize(self, context):
        # 此model_dir配置由启动torchserve的命令行参数model-store配置决定
        model_dir = context.system_properties.get("model_dir")
        # 模型打包部署的代码都打包在model.bin文件夹下,需要解压到当前文件夹
        with zipfile.ZipFile(model_dir + '/model.bin', 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        print("cuda.is_available ", torch.cuda.is_available(), " device_count ", torch.cuda.device_count())
        
        model_config_path = os.path.join(model_dir, 'checkpoint/checkpoint_940.pth')
        model_base_config_path = os.path.join(model_dir, 'checkpoint/checkpoint.pth')
        ocr_model_path = "checkpoint/ocr_general_clean.pt"
        yolo_model_path = "checkpoint/floorplan_best.pt"
        print(
            "model_config_path:", model_config_path, "model_base_config_path:", model_base_config_path,
            "ocr_model_path:", ocr_model_path, "yolo_model_path:", yolo_model_path
            )
        
        from predict import Predict
        print("parse_args start~")
        from options import parse_args
        args_t = parse_args()
        print("model load start~")
        self.model = Predict(args_t, model_config_path, model_base_config_path, ocr_model_path, yolo_model_path)
        print("model load over~")
        self._context = context
        self.initialized = True
    
    def load_image(self, url, type, invoke):
        print("Image url:", url, ";type:", type, "invoke:", invoke)
        image = Image.open(requests.get(url, stream=True).raw)
        return image
    
    def get_data_from_body(self, body):
        invoke = "floorplan"
        if "invoke" in list(body.keys()):
            invoke = body.get('invoke')
        return self.load_image(body.get('content'), body.get('type'), invoke)
    
    def get_invoke_from_body(self, body):
        invoke = "floorplan"
        if "invoke" in list(body.keys()):
            invoke = body.get('invoke')
        return invoke
    
    def handle(self, requests, context):
        bodies = [request.get("body") for request in requests]
        datas = [self.get_data_from_body(body) for body in bodies]
        invokes = [self.get_invoke_from_body(body) for body in bodies]
        final_result = []
        for i in range(len(datas)):
            item = datas[i]
            invoke = invokes[i]
            flag, ratio_res = self.model.ratio_predict(item)
            if invoke == 'ratio':
                final_result.append(json.dumps(ratio_res))
            elif invoke == 'floorplan':
                results = self.model.predict(item, type="url")  # type=url是从ossUrl读数据
                if flag == 1:
                    results["meta"] = ratio_res["meta"]
                final_result.append(json.dumps(results))
        return final_result
