import cv2
import numpy as np
import copy
from utils import *
from PIL import Image


class ImageTransform(object):
    def __init__(self, option):
        self.option = option
        
        self.image_height = 0
        self.image_width = 0
        
        self.resized_image_height = 0
        self.resized_image_width = 0
        
        self.offset_x = 0
        self.offset_y = 0
        
        super(ImageTransform, self).__init__()
    
    def transform_image_data(self, img_data):
        if img_data is None:
            return None
        image = copy.copy(img_data)
        max_size = max(self.option.width, self.option.height)
        # X, Y 图片宽高
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]
        
        image_sizes = np.array(image.shape[:2]).astype(np.float32)
        image_sizes = (image_sizes / image_sizes.max() * max_size).astype(np.int32)
        
        self.resized_image_height = image_sizes[0]
        self.resized_image_width = image_sizes[1]
        
        # 放置在图像中间. x>y
        if image_sizes[1] > image_sizes[0]:
            self.offset_x = 0
            self.offset_y = int(0.5 * (self.option.height - image_sizes[0]))
        else:
            self.offset_x = int(0.5 * (self.option.width - image_sizes[1]))
            self.offset_y = 0
        
        # 512 * 512
        full_image = np.full((self.option.height, self.option.width, 3), fill_value=255)
        self.resized_image_height = image_sizes[0]
        self.resized_image_width = image_sizes[1]
        
        start_x = self.offset_x
        end_x = start_x + self.resized_image_width
        start_y = self.offset_y
        end_y = start_y + self.resized_image_height
        
        # (X, Y)
        resized_img = cv2.resize(image, (self.resized_image_width, self.resized_image_height))
        
        full_image[start_y:end_y, start_x:end_x] = resized_img
        
        return full_image
    
    def transform_image(self, img_file_path, type):
        org_image = cv2_read_image(img_file_path, type)
        full_image = self.transform_image_data(org_image)
        return org_image, full_image
    
    def mapping_2_original_image_size(self, img_data):
        start_x = self.offset_x
        end_x = start_x + self.resized_image_width
        start_y = self.offset_y
        end_y = start_y + self.resized_image_height
        
        new_image = img_data[start_y:end_y, start_x:end_x]
        new_image = cv2.resize(new_image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        
        return new_image
