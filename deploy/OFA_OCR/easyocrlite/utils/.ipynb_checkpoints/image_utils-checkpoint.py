from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from easyocrlite.types import BoxTuple


def resize_aspect_ratio(
    img: np.ndarray, max_size: int, interpolation: int, expand_ratio: float = 1.0
) -> Tuple[np.ndarray, float]:
    height, width, channel = img.shape

    # magnify image size
    target_size = expand_ratio * max(height, width)

    # set original image size
    if max_size and max_size > 0 and target_size > max_size:
        target_size = max_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)

    if target_h != height or target_w != width:
        proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
        # make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = proc
        target_h, target_w = target_h32, target_w32
    else:
        resized = img
    return resized, ratio


def adjust_result_coordinates(
    box: BoxTuple, inverse_ratio: int = 1, ratio_net: int = 2
) -> np.ndarray:
    if len(box) > 0:
        box = np.array(box)
        for k in range(len(box)):
            if box[k] is not None:
                box[k] *= (inverse_ratio * ratio_net, inverse_ratio * ratio_net)
    return box


def normalize_mean_variance(
    in_img: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    variance: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img

def boxed_transform(image: np.ndarray, box: BoxTuple) -> np.ndarray:
    (tl, tr, br, bl) = box

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped