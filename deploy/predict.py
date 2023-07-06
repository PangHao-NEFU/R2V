import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
cur_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(cur_folder_path)

from WallBuilder import *

from options import parse_args
from model import Model

from debug_utils import *

from imagetransform import *


class Predict(object):
    def __init__(self, options, model_config_path=''):

        self.model_config_path = model_config_path

        self.options = options

        self.model_base = None

        self.model = None

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
        if self.options.gpuFlag == 1:
            self.model.load_state_dict(torch.load(self.model_config_path))
            self.model = self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(self.model_config_path, map_location='cpu'))

        self.debug_util_obj = DebugInfo(self.options)

    def pre_process_image(self, img_file_path, type):
        return self.image_transform.transform_image(img_file_path, type)

    def generate_heatmap_data(self, img_file_path, type):
        try:
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

            # 1. pre-process the image.
            image = (image.astype(np.float32) / 255 - 0.5)
            # self.debug_util_obj.save_floorplan_imag_with_name(image * 255, res_folder_path=res_folder_path,
            #                                                   name="PreProcessFullImage")
            image = image.transpose((2, 0, 1))
            image_tensor = torch.Tensor(image)
            image_tensor = image_tensor.view((1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]))

            if self.options.gpuFlag == 1:
                image_tensor = image_tensor.cuda()

            print("prepare data cost {0}".format(time.time() - start))

            start = time.time()
            # 2. predict the results.
            corner_pred = self.model(image_tensor)
            print("predict model cost {0}".format(time.time() - start))
            if self.options.debugFlag == 1:
                self.debug_util_obj.save_corner_heatmaps_img(corner_pred[0], self.image_transform,
                                                             res_folder_path=self.options.res_folder_path)

            start = time.time()
            # 3. get the prediction data.
            corner_heatmaps = corner_pred[0].detach().cpu().numpy()
            corner_heatmaps = corner_heatmaps[0]

            all_heatmap_data = []
            for i in range(corner_heatmaps.shape[2]):
                cur_heatmap = corner_heatmaps[:, :, i]
                if self.image_transform is not None:
                    # heatmap图片尺寸适配
                    cur_heatmap = self.image_transform.mapping_2_original_image_size(cur_heatmap)
                all_heatmap_data.append(cur_heatmap)
            self.all_heatmap_data = all_heatmap_data
            print("generate heatmap cost {0}".format(time.time() - start))
        except:
            raise

    # if measuring_scale = -1.0, use default measuring scale ratio.
    def predict(self, img_file_path, measuring_scale=-1.0, type='url'):
        try:
            # 1. calculate the heatmap.
            self.generate_heatmap_data(img_file_path, type)

            start = time.time()

            wall_builder = Builder(self.options)
            org_image = cv2_read_image(img_file_path, type)
            # 2. build floorplan json data.
            res = wall_builder.build_floorplan_json(org_image, self.all_heatmap_data,
                                                    measuring_scale_ratio=measuring_scale)

            print("post process Builder function cost {0}".format(time.time() - start))
            return json.dumps(res)
        except:
            raise


if __name__ == "__main__":
    args = parse_args()
    args.outputDir = "./detectResult"
    args.debugFlag = 1
    folder_path = os.path.dirname(os.path.abspath(__file__))
    imgUrl = "https://henry-search.oss-cn-hangzhou.aliyuncs.com/im2fp/floorplan.png"
    imagePath1 = "0a0eccef-2277-4da0-9fe1-7277299af870.png"  # 单斜墙
    imagePath2 = "2caf3f3e-6225-4ee2-a74f-b98d7881e09c.jpg"  # 多斜墙
    imagePath3 = "0ad81247-8ac5-4287-ba1e-ea500e113298.jpg"  # 多斜墙
    img_file_path = os.path.join(folder_path, "../detectData/" + imagePath1)
    model_config_path = 'checkpoint/checkpoint_100.pth'
    predictor = Predict(args, model_config_path)
    res = predictor.predict(imagePath3,type="111")
    print(res)
