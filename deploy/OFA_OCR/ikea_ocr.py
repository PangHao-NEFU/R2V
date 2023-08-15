import os
import torch
import numpy as np

from OFA_OCR.fairseq import utils
from OFA_OCR.fairseq import checkpoint_utils
from typing import Tuple
from OFA_OCR.easyocrlite.reader import ReaderLite
from OFA_OCR.ocr_utils import get_boundaryDirection_yValue_ocrPoints, get_corner_points, get_ocr_ratio

from PIL import Image, ImageDraw
from torchvision import transforms
from OFA_OCR.data.mm_data.ocr_dataset import ocr_resize
from OFA_OCR.utils_.eval_utils import eval_step


class IKEA_OCR_Detect(object):
    def __init__(self, ocr_model_path='', yolo_model_path=''):
        self.ocr_model_path = ocr_model_path
        self.yolo_model_path = yolo_model_path
        self.ocr_model = None
        self.yolo_model = None
        self.ofa_models = None
        self.cfg = None
        self.use_fp16 = False
        self.task = None
        self.reader = None
        self.generator = None
        self.bos_item = None
        self.eos_item = None
        self.pad_idx = None
        self.use_cuda = False
        self.mean = None
        self.std = None
        self.Rect = None
        self.FourPoint = None
        super(IKEA_OCR_Detect, self).__init__()
        self._init_model()

    def _init_model(self):
        print("cuda.is_available", torch.cuda.is_available())
        # turn on cuda if GPU is available
        self.use_cuda = torch.cuda.is_available()
        # use fp16 only when GPU is available
        self.use_fp16 = self.use_cuda
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.Rect = Tuple[int, int, int, int]
        self.FourPoint = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]
        self.reader = ReaderLite(gpu=self.use_cuda)
        overrides = {"eval_cider": False, "beam": 5, "max_len_b": 64, "patch_image_size": 480,
                     "orig_patch_image_size": 224, "interpolate_position": True,
                     "no_repeat_ngram_size": 0, "seed": 42}
        print("~~~~~~load ocr model~~~~~~")
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(self.ocr_model_path),
            arg_overrides=overrides
        )
        self.ofa_models = models
        self.cfg = cfg
        self.task = task
        for model in models:
            model.eval()
            if self.use_fp16:
                model.half()
            if self.use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            self.model = model
            self.model.prepare_for_inference_(cfg)
        self.generator = task.build_generator(models, cfg.generation)
        self.bos_item = torch.LongTensor([task.src_dict.bos()])
        self.eos_item = torch.LongTensor([task.src_dict.eos()])
        self.pad_idx = task.src_dict.pad()
        print("~~~~~~load yolo model~~~~~~")
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.yolo_model_path)
        self.yolo_model.conf = 0.1
        self.yolo_model.iou = 0.1
        print("~~~~~~yolo model load over~~~~~~")

    def _get_images(self, image_path: str, reader: ReaderLite, **kwargs):
        results = self.reader.process(image_path, **kwargs)
        return results

    def _draw_boxes(image, bounds, color='red', width=2):
        draw = ImageDraw.Draw(image)
        for bound in bounds:
            p0, p1, p2, p3 = bound
            draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
        return image

    def _construct_sample(self, task, image: Image, patch_image_size=480):
        patch_image = self._patch_resize_transform(patch_image_size)(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = self._encode_text(task, "图片上的文字是什么?", append_bos=True, append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(self.pad_idx).long().sum() for s in src_text])
        sample = {
            "id": np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask,
            },
            "target": None
        }
        return sample

    def _patch_resize_transform(self, patch_image_size=480, is_document=False):
        _patch_resize_transform = transforms.Compose(
            [
                lambda image: ocr_resize(
                    image, patch_image_size, is_document=is_document
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        return _patch_resize_transform

    def _encode_text(self, task, text, length=None, append_bos=False, append_eos=False):
        bos_item = torch.LongTensor([task.src_dict.bos()])
        eos_item = torch.LongTensor([task.src_dict.eos()])
        pad_idx = task.src_dict.pad()

        s = task.tgt_dict.encode_line(
            line=task.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()

        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s

    def _apply_half(self, t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    def get_yolo_detect_bbox(self, image):
        results = self.yolo_model(image).pandas().xyxy[0]
        bboxs = []
        for idx, item in results.iterrows():
            x1 = item['xmin']
            y1 = item['ymin']
            x2 = item['xmax']
            y2 = item['ymax']
            bboxs.append([x1, y1, x2, y2, item['confidence']])
        return bboxs

    def ikea_ocr(self, img):
        results = self._get_images(img, self.reader)
        box_list, image_list = zip(*results)
        # self._draw_boxes(orig_image, box_list)
        ocr_result_bboxs = []
        for box, image in zip(box_list, image_list):
            image = Image.fromarray(image)
            sample = self._construct_sample(self.task, image, self.cfg.task.patch_image_size)
            sample = utils.move_to_cuda(sample) if self.use_cuda else sample
            sample = utils.apply_to_sample(self._apply_half, sample) if self.use_fp16 else sample

            with torch.no_grad():
                result, scores = eval_step(self.task, self.generator, self.ofa_models, sample)
            ocr_result_bboxs.append([box, result[0]['ocr']])

        return ocr_result_bboxs


if __name__ == "__main__":
    ocr_model_path = '../checkpoint/ocr_general_clean.pt'
    yolo_model_path = '../checkpoint/floorplan_best.pt'
    folder_path = os.path.dirname(os.path.abspath(__file__))
    imagePath = "2b2f621c-f6b0-48c7-b785-b425097df544.png"
    img_file_path = os.path.join(folder_path, "check_data/" + imagePath)
    predictor = IKEA_OCR_Detect(ocr_model_path, yolo_model_path)
    res = predictor.ikea_ocr(img_file_path)
    cropped_img, boundary_idx, y_value, ocr_point_list = get_boundaryDirection_yValue_ocrPoints(img_file_path, res)
    yolo_bboxs = predictor._get_yolo_detect_bbox(cropped_img)
    corner_point_map = get_corner_points(cropped_img, yolo_bboxs)
    print(boundary_idx, y_value, ocr_point_list, yolo_bboxs, corner_point_map)
    up_ratios = get_ocr_ratio(boundary_idx, ocr_point_list, corner_point_map, y_value)
    print("up_ratios:", up_ratios)
