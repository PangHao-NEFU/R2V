from __future__ import annotations

import logging
from typing import Union
import os
from pathlib import Path
from typing import Tuple

import PIL.Image
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance

from OFA_OCR.easyocrlite.model.craft import CRAFT

from OFA_OCR.easyocrlite.utils.download_utils import prepare_model
from OFA_OCR.easyocrlite.utils.image_utils import (
    adjust_result_coordinates,
    boxed_transform,
    normalize_mean_variance,
    resize_aspect_ratio,
)
from OFA_OCR.easyocrlite.utils.detect_utils import (
    extract_boxes,
    extract_regions_from_boxes,
    box_expand,
    greedy_merge,
)
from OFA_OCR.easyocrlite.types import BoxTuple, RegionTuple
import OFA_OCR.easyocrlite.utils.utils as utils

logger = logging.getLogger(__name__)

MODULE_PATH = (
    os.environ.get("EASYOCR_MODULE_PATH")
    or os.environ.get("MODULE_PATH")
    or os.path.expanduser("~/.EasyOCR/")
)


class ReaderLite(object):
    def __init__(
        self,
        gpu=True,
        model_storage_directory=None,
        download_enabled=True,
        verbose=True,
        quantize=True,
        cudnn_benchmark=False,
    ):

        self.verbose = verbose

        model_storage_directory = Path(
            model_storage_directory
            if model_storage_directory
            else MODULE_PATH + "/model"
        )
        self.detector_path = prepare_model(
            model_storage_directory, download_enabled, verbose
        )

        self.quantize = quantize
        self.cudnn_benchmark = cudnn_benchmark
        if gpu is False:
            self.device = "cpu"
            if verbose:
                logger.warning(
                    "Using CPU. Note: This module is much faster with a GPU."
                )
        elif not torch.cuda.is_available():
            self.device = "cpu"
            if verbose:
                logger.warning(
                    "CUDA not available - defaulting to CPU. Note: This module is much faster with a GPU."
                )
        elif gpu is True:
            self.device = "cuda"
        else:
            self.device = gpu

        self.detector = CRAFT()

        state_dict = torch.load(self.detector_path, map_location=self.device)
        if list(state_dict.keys())[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        self.detector.load_state_dict(state_dict)

        if self.device == "cpu":
            if self.quantize:
                try:
                    torch.quantization.quantize_dynamic(
                        self.detector, dtype=torch.qint8, inplace=True
                    )
                except:
                    pass
        else:
            self.detector = torch.nn.DataParallel(self.detector).to(self.device)
            import torch.backends.cudnn as cudnn

            cudnn.benchmark = self.cudnn_benchmark

        self.detector.eval()

    def process(
        self,
        image_path: Union[str, PIL.Image.Image],
        max_size: int = 960,
        expand_ratio: float = 1.0,
        sharp: float = 1.0,
        contrast: float = 1.0,
        text_confidence: float = 0.7,
        text_threshold: float = 0.4,
        link_threshold: float = 0.4,
        slope_ths: float = 0.1,
        ratio_ths: float = 0.5,
        center_ths: float = 0.5,
        dim_ths: float = 0.5,
        space_ths: float = 1.0,
        add_margin: float = 0.1,
        min_size: float = 0.01,
    ) -> Tuple[BoxTuple, list[np.ndarray]]:
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, PIL.Image.Image):
            image = image_path.convert('RGB')
        width, height = image.size
        height_relative = int(height / 3)
        tensor, inverse_ratio = self.preprocess(
            image, max_size, expand_ratio, sharp, contrast
        )

        scores = self.forward_net(tensor)

        boxes = self.detect(scores, text_confidence, text_threshold, link_threshold)
    

        image = np.array(image)
        region_list, box_list = self.postprocess(
            image,
            boxes,
            inverse_ratio,
            slope_ths,
            ratio_ths,
            center_ths,
            dim_ths,
            space_ths,
            add_margin,
            min_size,
        )
        # print(region_list,box_list)
        # get cropped image
        image_list = []
        for region in region_list:
            x_min, x_max, y_min, y_max = region
            x_min = x_min -8 if (x_min-10)>0 else x_min
            if y_min <= height_relative:
                crop_img = image[y_min:y_max, x_min:x_max, :]
                image_list.append(
                    (
                        ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)),
                        crop_img,
                    )
                )

        for box in box_list:
            transformed_img = boxed_transform(image, np.array(box, dtype="float32"))
            image_list.append((box, transformed_img))

        # sort by top left point
        image_list = sorted(image_list, key=lambda x: (x[0][0][1], x[0][0][0]))

        return image_list

    def preprocess(
        self,
        image: Image.Image,
        max_size: int,
        expand_ratio: float = 1.0,
        sharp: float = 1.0,
        contrast: float = 1.0,
    ) -> torch.Tensor:
        if sharp != 1:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharp)
        if contrast != 1:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)

        image = np.array(image)

        image, target_ratio = resize_aspect_ratio(
            image, max_size, interpolation=cv2.INTER_LINEAR, expand_ratio=expand_ratio
        )
        inverse_ratio = 1 / target_ratio

        x = np.transpose(normalize_mean_variance(image), (2, 0, 1))

        x = torch.tensor(np.array([x]), device=self.device)

        return x, inverse_ratio

    @torch.no_grad()
    def forward_net(self, tensor: torch.Tensor) -> torch.Tensor:
        scores, feature = self.detector(tensor)
        return scores[0]

    def detect(
        self,
        scores: torch.Tensor,
        text_confidence: float = 0.7,
        text_threshold: float = 0.4,
        link_threshold: float = 0.4,
    ) -> list[BoxTuple]:
        # make score and link map
        score_text = scores[:, :, 0].cpu().data.numpy()
        score_link = scores[:, :, 1].cpu().data.numpy()
        # extract box
        boxes, _ = extract_boxes(
            score_text, score_link, text_confidence, text_threshold, link_threshold
        )
        return boxes

    def postprocess(
        self,
        image: np.ndarray,
        boxes: list[BoxTuple],
        inverse_ratio: float,
        slope_ths: float = 0.1,
        ratio_ths: float = 0.5,
        center_ths: float = 0.5,
        dim_ths: float = 0.5,
        space_ths: float = 1.0,
        add_margin: float = 0.1,
        min_size: int = 0,
    ) -> Tuple[list[RegionTuple], list[BoxTuple]]:

        # coordinate adjustment
        boxes = adjust_result_coordinates(boxes, inverse_ratio)

        max_y, max_x, _ = image.shape

        # extract region and merge
        region_list, box_list = extract_regions_from_boxes(boxes, slope_ths)

        region_list = greedy_merge(
            region_list,
            ratio_ths=ratio_ths,
            center_ths=center_ths,
            dim_ths=dim_ths,
            space_ths=space_ths,
            verbose=0
        )

        # add margin
        region_list = [
            region.expand(add_margin, (max_x, max_y)).as_tuple()
            for region in region_list
        ]

        box_list = [box_expand(box, add_margin, (max_x, max_y)) for box in box_list]

        # filter by size
        if min_size:
            if min_size < 1:
                min_size = int(min(max_y, max_x) * min_size)

            region_list = [
                i for i in region_list if max(i[1] - i[0], i[3] - i[2]) > min_size
            ]
            box_list = [
                i
                for i in box_list
                if max(utils.diff([c[0] for c in i]), utils.diff([c[1] for c in i]))
                > min_size
            ]

        return region_list, box_list