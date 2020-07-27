import os
import sys

from glob import glob
from models.Densenet_T2_ABK_auc_08.utils.helpers import *
import json
import SimpleITK as sitk
import models.settings as S
from keras.models import model_from_json
sys.path.append("../")


class Deploy:
    def __init__(self):
        self.current_dir = os.path.dirname(__file__)
        self.datagen_dict = read_json(self.current_dir + "/configs/datagen.json")['datagen']
        self.resampling_dict = read_json(self.current_dir + "/configs/preprocess.json")['preprocessing'][
            "resampling"]
        self.datagen_dict_specs = self.datagen_dict['specs']
        self.datagen_dict_prep = self.datagen_dict['preprocessing']

    def build(self):
        json_file = open(self.current_dir + "/model/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.current_dir + "/model/model_checkpoint.h5")
        return loaded_model

    def run(self, model, info):
        self.info = info
        self.case = info["case"]
        print("*" * 100)
        # print(self.info)
        print(self.case)
        print(self.info["lps"])
        t2_test, abk_test, zone_encoding = self.extract_patches()  # 提取图像patch
        t2_test, abk_test = self.mean_std_standarzation(t2_test, abk_test)
        x = [t2_test, abk_test]
        predicted_prob = model.predict(x, verbose=1)
        print("successss" * 6)
        scores = np.concatenate(predicted_prob).ravel()
        print("predictions: {} ".format(scores))
        description = "{:03.1f}% probability of Significant Prostate Cancer".format(scores[0] * 100)
        response_dict = {"case": self.info["case"],
                         "description": description,
                         "score": str(scores[0])}
        return json.dumps(response_dict)

    def resample_image(self, image):
        voxel_resampling_dict = {"t2_tse_tra": self.resampling_dict["spacing"]["t2"],
                                 "ADC": self.resampling_dict["spacing"]["dwi"],
                                 "BVAL": self.resampling_dict["spacing"]["dwi"],
                                 "Ktrans": self.resampling_dict["spacing"]["ktrans"]}
        return resample_new_spacing(image, target_spacing=voxel_resampling_dict["t2_tse_tra"])

    def read_image(self):
        image_paths = glob(os.path.join(S.nrrd_folder, self.case + '*.*'))  # 返回匹配的文件路径

        assert len(image_paths) == 1, print(self.case, "more than one image or zero")
        # print(image_paths)
        image = sitk.ReadImage(image_paths[0])
        image = self.resample_image(image)
        image_prep = preprocess(image=image,
                                window_intensity_dict=self.datagen_dict_prep["window_intensity"],
                                zero_scale_dict=self.datagen_dict_prep["rescale_zero_one"])
        return image_prep

    def extract_patches(self):
        zone_encoding = np.zeros((1, 3), dtype=np.float32)
        if self.info["zone"].lower().startswith('p'):
            zone_encoding[0, ...] = np.array([1, 0, 0])
        elif self.info["zone"].lower().startswith('t'):
            zone_encoding[0, ...] = np.array([0, 1, 0])
        elif self.info["zone"].lower().startswith('a'):
            zone_encoding[0, ...] = np.array([0, 0, 1])
        size_x, size_y, size_z = self.datagen_dict_specs["output_patch_shape"]["size"]  # patch大小

        t2_test = np.zeros((1, size_x, size_y, size_z, 1), dtype=np.float32)
        abk_test = np.zeros((1, size_x // 2, size_y // 2, size_z, 3), dtype=np.float32)

        image = self.read_image()  # 分别读取三种类型的MRI图像, 不包括'Ktrans'
        lps = self.info["lps"]
        ijk = image.TransformPhysicalPointToIndex(lps)

        image_cropped = crop_roi(image, ijk, [size_x, size_y, size_z])
        image_cropped_arr = sitk.GetArrayFromImage(image_cropped)
        image_cropped_arr = np.swapaxes(image_cropped_arr, 0, 2)
        t2_test[0, ..., 0] = image_cropped_arr
        return t2_test, abk_test, zone_encoding

    def mean_std_standarzation(self, t2_arr, abk_arr):
        mean_std_dir = os.path.join(os.path.dirname(__file__), "model/mean_stds/")
        t2_mean = np.load(mean_std_dir + "/training_t2_mean.npy")
        t2_std = np.load(mean_std_dir + "/training_t2_std.npy")
        #
        abk_mean = np.load(mean_std_dir + "/training_abk_mean.npy")
        abk_std = np.load(mean_std_dir + "/training_abk_std.npy")
        #
        t2_arr -= t2_mean
        t2_arr /= t2_std
        #
        abk_arr -= abk_mean
        abk_arr /= abk_std
        return t2_arr, abk_arr
