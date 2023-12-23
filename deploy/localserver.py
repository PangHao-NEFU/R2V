from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
from PIL import Image
import torch
import os
from options import parse_args
from predict import Predict
from typing import Union
import uuid

app = FastAPI()
# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# print(app)

# 初始化模型文件
initialized = False
print("cuda.is_available ", torch.cuda.is_available(), " device_count ", torch.cuda.device_count())
model_config_path = 'checkpoints/checkpoint_940.pth'
model_base_config_path = 'checkpoints/checkpoint.pth'
ocr_model_path = "checkpoints/ocr_general_clean.pt"
yolo_model_path = "checkpoints/floorplan_best.pt"
print(
    "model_config_path:", model_config_path, "model_base_config_path:", model_base_config_path,
    "ocr_model_path:", ocr_model_path, "yolo_model_path:", yolo_model_path
)

print("model load start~")
args_t = parse_args()
model = Predict(args_t, model_config_path, model_base_config_path, ocr_model_path, yolo_model_path)
print("model load over~")
initialized = True


class FloorplanRequest(BaseModel):
    content: str  # 图片url
    type: str  # 类型"image"目前来说没啥用
    invoke: str  # "floorplan" | "ratio" = "floorplan"
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return None
    
    def get(self, attrname):
        if attrname in self.__dict__:
            return self.__dict__[attrname]
        return None


def load_image(url, type, invoke):
    print("Image url:", url, ";type:", type, "invoke:", invoke)
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def get_data_from_body(body):
    invoke = "floorplan"
    if body.get('invoke') == "ratio":
        invoke = "ratio"
    return load_image(body.get('content'), body.get('type'), invoke)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post('/floorplan')
async def floorplan(body: FloorplanRequest):
    """
    
    Args:
        body: 从网络url加载图片

    Returns:

    """
    global model
    image_data = get_data_from_body(body)
    invoke = body.get('invoke')
    
    final_result = None
    flag, ratio_res = model.ratio_predict(image_data)
    if invoke == 'ratio':
        final_result = json.dumps(ratio_res)
    elif invoke == 'floorplan':
        results = model.predict(image_data, type="url")  # type=url是从ossUrl读数据
        if flag == 1:
            results["meta"] = ratio_res["meta"]
        final_result = json.dumps(results)
    return {"code": 200, "data": final_result, "message": 'ok'}


@app.post('/file/upload')
async def uploadFile(file: Union[UploadFile, None] = None, scale: str = Form()):
    """
    
    Args:
        file: 上传图片到服务器
        scale:

    Returns:

    """
    if not os.path.exists('./uploadfiles'):
        os.makedirs('./uploadfiles')
    
    filename = file.filename + '_' + str(uuid.uuid4())
    filetype = file.content_type
    
    with open(f'./uploadfiles/{filename}', 'wb+') as f:
        f.write(await file.read())
    image_url = f"/uploadfiles/{filename}"
    return {"code": 200, "data": {"url": image_url, "filename": filename}, "message": 'ok'}


@app.post('/floorplan/local')
async def floorplan_local(body: FloorplanRequest):
    """
    本地图片推理
    Args:
        body:

    Returns:

    """
    filename = body.get('content')
    invoke = body.get('invoke')
    image_data = Image.open(f"./uploadfiles/{filename}")
    final_result = None
    flag, ratio_res = model.ratio_predict(image_data)
    if invoke == 'ratio':
        final_result = json.dumps(ratio_res)
    elif invoke == 'floorplan':
        results = model.predict(image_data, type="url")  # type=url是从ossUrl读数据
        if flag == 1:
            results["meta"] = ratio_res["meta"]
        final_result = json.dumps(results)
    return {"code": 200, "data": final_result, "message": 'ok'}


if __name__ == "__main__":
    uvicorn.run(app, port=8080, log_level='info')
