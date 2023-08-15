from __future__ import annotations

import itertools
import logging
import math
import operator
from collections import namedtuple
from functools import cached_property
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from ..types import BoxTuple, RegionTuple
from ..utils.utils import grouped_by

logger = logging.getLogger(__name__)

class Region(namedtuple("Region", ["x_min", "x_max", "y_min", "y_max"])):
    @cached_property
    def ycenter(self):
        return 0.5 * (self.y_min + self.y_max)

    @cached_property
    def xcenter(self):
        return 0.5 * (self.x_min + self.x_max)

    @cached_property
    def height(self):
        return self.y_max - self.y_min

    @cached_property
    def width(self):
        return self.x_max - self.x_min

    @classmethod
    def from_box(cls, box: BoxTuple) -> Region:
        (xtl, ytl), (xtr, ytr), (xbr, ybr), (xbl, ybl) = box

        x_max = max(xtl, xtr, xbr, xbl)
        x_min = min(xtl, xtr, xbr, xbl)
        y_max = max(ytl, ytr, ybr, ybl)
        y_min = min(ytl, ytr, ybr, ybl)

        return cls(x_min, x_max, y_min, y_max)

    def as_tuple(self) -> RegionTuple:
        return self.x_min, self.x_max, self.y_min, self.y_max

    def expand(
        self, add_margin: float, size: Optional[Tuple[int, int] | int] = None
    ) -> Region:

        margin = int(add_margin * min(self.width, self.height))
        if isinstance(size, Iterable):
            max_x, max_y = size
        elif size is None:
            max_x = self.width * 2
            max_y = self.height * 2
        else:
            max_x = max_y = size

        return Region(
            max(0, self.x_min - margin),
            min(max_x, self.x_max + margin),
            max(0, self.y_min - margin),
            min(max_y, self.y_max + margin),
        )

    def __add__(self, region: Region) -> Region:
        return Region(
            min(self.x_min, region.x_min),
            max(self.x_max, region.x_max),
            min(self.y_min, region.y_min),
            max(self.y_max, region.y_max),
        )

def extract_boxes(
    textmap: np.ndarray,
    linkmap: np.ndarray,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
) -> Tuple[list[BoxTuple], list[int]]:
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    boxes = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255

        mapper.append(k)
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = (
            np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
            .transpose()
            .reshape(-1, 2)
        )
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
        boxes.append(box)

    return boxes, mapper


def extract_regions_from_boxes(
    boxes: list[BoxTuple], slope_ths: float
) -> Tuple[list[Region], list[BoxTuple]]:

    region_list: list[Region] = []
    box_list = []

    for box in boxes:
        box = np.array(box).astype(np.int32)
        (xtl, ytl), (xtr, ytr), (xbr, ybr), (xbl, ybl) = box

        # get the tan of top and bottom edge
        # why 10?
        slope_top = (ytr - ytl) / max(10, xtr - xtl)
        slope_bottom = (ybr - ybl) / max(10, xbr - xbl)
        if max(abs(slope_top), abs(slope_bottom)) < slope_ths:
            # not very tilted, rectangle box
            region_list.append(Region.from_box(box))
        else:
            # tilted
            box_list.append(box)
    return region_list, box_list


def box_expand(
    box: BoxTuple, add_margin: float, size: Optional[Tuple[int, int] | int] = None
) -> BoxTuple:

    (xtl, ytl), (xtr, ytr), (xbr, ybr), (xbl, ybl) = box
    height = np.linalg.norm([xbl - xtl, ybl - ytl])  # from top left to bottom left
    width = np.linalg.norm([xtr - xtl, ytr - ytl])  # from top left to top right

    # margin is added based on the diagonal
    margin = int(1.44 * add_margin * min(width, height))

    theta13 = abs(np.arctan((ytl - ybr) / max(10, (xtl - xbr))))
    theta24 = abs(np.arctan((ytr - ybl) / max(10, (xtr - xbl))))

    if isinstance(size, Iterable):
        max_x, max_y = size
    elif size is None:
        max_x = width * 2
        max_y = height * 2
    else:
        max_x = max_y = size

    new_box = (
        (
            max(0, int(xtl - np.cos(theta13) * margin)),
            max(0, int(ytl - np.sin(theta13) * margin)),
        ),
        (
            min(max_x, math.ceil(xtr + np.cos(theta24) * margin)),
            max(0, int(ytr - np.sin(theta24) * margin)),
        ),
        (
            min(max_x, math.ceil(xbr + np.cos(theta13) * margin)),
            min(max_y, math.ceil(ybr + np.sin(theta13) * margin)),
        ),
        (
            max(0, int(xbl - np.cos(theta24) * margin)),
            min(max_y, math.ceil(ybl + np.sin(theta24) * margin)),
        ),
    )
    return new_box


def greedy_merge(
    regions: list[Region],
    ratio_ths: float = 0.5,
    center_ths: float = 0.5,
    dim_ths: float = 0.5,
    space_ths: float = 1.0,
    verbose: int = 4,
) -> list[Region]:

    regions = sorted(regions, key=operator.attrgetter("ycenter"))

    # grouped by ycenter
    groups = grouped_by(
        regions,
        operator.attrgetter("ycenter"),
        center_ths,
        operator.attrgetter("height"),
    )
    for group in groups:
        group.sort(key=operator.attrgetter("x_min"))
        idx = 0
        while idx < len(group) - 1:
            region1, region2 = group[idx], group[idx + 1]
            # both are horizontal regions
            cond = (region1.width / region1.height) > ratio_ths and (
                region2.width / region2.height
            ) > ratio_ths
            # similar heights
            cond = cond and abs(region1.height - region2.height) < dim_ths * np.mean(
                [region1.height, region2.height]
            )
            # similar ycenters
            # cond = cond and abs(region1.ycenter - region2.ycenter) < center_ths * np.mean(
            #     [region1.height, region2.height]
            # )
            # horizontal space is small
            cond = cond and (region2.x_min - region1.x_max) < space_ths * np.mean(
                [region1.height, region2.height]
            )
            if cond:
                # merge regiona
                region = region1 + region2

                if verbose > 2:
                    logger.debug(f"horizontal merging {region1} {region2}")
                group.pop(idx)
                group.pop(idx)
                group.insert(idx, region)

            else:
                if verbose > 0:
                    logger.debug(f"not horizontal merging {region1} {region2}")
                idx += 1

    # flatten groups
    regions = list(itertools.chain.from_iterable(groups))

    # grouped by xcenter
    groups = grouped_by(
        regions,
        operator.attrgetter("xcenter"),
        center_ths,
        operator.attrgetter("width"),
    )

    for group in groups:
        group.sort(key=operator.attrgetter("y_min"))
        idx = 0
        while idx < len(group) - 1:
            region1, region2 = group[idx], group[idx + 1]
            # both are vertical regions
            cond = (region1.height / region1.width) > ratio_ths and (
                region2.height / region2.width
            ) > ratio_ths
            # similar widths
            cond = cond and abs(region1.width - region2.width) < dim_ths * np.mean(
                [region1.width, region2.width]
            )
            # # similar xcenters
            # cond = cond and abs(region1.xcenter - region2.xcenter) < center_ths * np.mean(
            #     [region1.width, region2.width]
            # )
            # vertical space is small
            cond = cond and (region2.y_min - region1.y_max) < space_ths * np.mean(
                [region1.width, region2.width]
            )
            if cond:
                # merge region
                region = region1 + region2
                if verbose > 2:
                    logger.debug(f"vertical merging {region1} {region2}")
                group.pop(idx)
                group.pop(idx)
                group.insert(idx, region)
            else:
                if verbose > 1:
                    logger.debug(f"not vertical merging {region1} {region2}")
                idx += 1

    # flatten groups
    regions = list(itertools.chain.from_iterable(groups))

    return regions