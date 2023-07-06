import torch
import os
import zipfile


class ModelHandler(object):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None

    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        with zipfile.ZipFile(model_dir + '/model.bin', 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        print("cuda.is_available ", torch.cuda.is_available(), " device_count ", torch.cuda.device_count())
        model_config_path = os.path.join(model_dir, 'checkpoint/checkpoint_100.pth')
        print("model_config_path:", model_config_path)
        from predict import Predict
        print("parse_args start~")
        from options import parse_args
        args_t = parse_args()
        print("model load start~")
        self.model = Predict(args_t, model_config_path)
        print("model load over~")
        self._context = context
        self.initialized = True

    def load_image(self, url, bbox):
        print("Image url:", url, ";type:", type)
        return url

    def get_data_from_body(self, body):
        return self.load_image(body.get('content'), body.get('bbox'))

    def handle(self, requests, context):
        bodies = [request.get("body") for request in requests]
        datas = [self.get_data_from_body(body) for body in bodies]
        final_result = []
        for item in datas:
            results = self.model.predict(item, type="url")
            final_result.append(results)
        return final_result
