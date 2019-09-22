from model_loader import ModelLoaderFactory, get_transpose_tuple, ModelType
import util
import dlr
from abc import ABCMeta, abstractmethod
import numpy as np


class SageMakerNeoWrapper:
    def __init__(self, params):
        self.__model = None
        self.__model_loader = None
        self.__result = None
        self.__params = params

    def load(self):
        # load model data
        loader = ModelLoaderFactory.get_loader(self.__params.model_define, self.__params.model_root_path)
        loader.setup()
        model_path = loader.get_model_path()

        # create Deep Learning Runtime
        self.__model_loader = loader
        self.__model = dlr.DLRModel(model_path, self.__params.target_device)

    def run(self, cv2_images):
        if self.__model is None:
            raise NotLoadException("SageMakerNeo Runtime is not initialized! Please call 'load' function.")

        # create input data
        model_define = self.__params.model_define
        transpose_tuple = get_transpose_tuple(model_define)
        input_size = self.__params.model_define["input_size"]
        imgs_ndarray = util.open_and_norm_images(cv2_images, input_size, transpose_tuple)
        input_tensor = imgs_ndarray.astype("float32")
        input_data = util.get_input_data(model_define, input_tensor)

        # run inference
        self.__result = self.__model.run(input_data)

    def get_result(self):
        model_detail = self.__model_loader.get_model_detail()
        return self.__convert_result(origin_result=self.__result, model_type=model_detail.model_type, threshold=self.__params.threshold)

    def draw_boxes(self, images):
        # TODO : imp
        pass

    def __convert_result(self, origin_result, model_type, threshold):
        converter = NeoResultConverterFactory.get_converter(model_type)
        return converter.convert_result(origin_result, threshold)


class NotLoadException(Exception):
    pass


class NeoParameters:
    def __init__(self, model_define, model_root_path, target_device, threshold=0.5):
        self.model_define = model_define
        self.model_root_path = model_root_path
        self.target_device = target_device
        self.threshold = threshold


class NeoInferResult:
    def __init__(self, result):
        self.__result = result

    def get_result(self):
        """
        get result.
        :return:

        result shape is follow.
        [
            - image file
            [
                - object
                [class id, score, boxes(bottom, left, top, right)]

                ...
                []
            ]
            ...
            []
        ]

        (ymin, xmin, ymax, xmax) -> (bottom, left, top, right)
        """
        return self.__result


class NeoResultConverterFactory:
    @classmethod
    def get_converter(cls, model_type):
        if model_type == ModelType.TENSORFLOW:
            return TFResultConverter()
        elif model_type == ModelType.MXNET:
            return MXNetResultConverter()
        else:
            raise NeoResultConverterNotDefinedError("{} : neo result converter is not defined.".format(model_type))


class NeoResultConverterNotDefinedError(Exception):
    pass


class NeoResultConverter:
    __metaclass__ = ABCMeta

    def __init__(self, one_detect_callback=None, one_image_callback=None):
        """
        initialize converter
        :param one_detect_callback: detection callback
            param is [image_number, cid, score, bottom, left, top, right]
        :param one_image_callback:
            param is [image_number]
        """
        self._one_detect_callback = one_detect_callback
        self._one_image_callback = one_image_callback

    @abstractmethod
    def convert_result(self, origin_result, threshold):
        pass

    def draw_boxes(self, origin_result, threshold):
        pass


class TFResultConverter(NeoResultConverter):
    def __init__(self, one_detect_callback=None, one_image_callback=None):
        super(TFResultConverter, self).__init__(one_detect_callback, one_image_callback)

    def convert_result(self, origin_result, threshold):
        boxes, classes, scores, num_det = origin_result

        # get file count
        file_count = len(num_det)

        # start loop
        convert_res_for_imgs = []
        for i in range(file_count):
            n_obj = int(num_det[i])
            convert_res_for_img = []
            for j in range(n_obj):
                # get class id
                cid = int(classes[i][j])

                # get score (check if it is the threshold)
                score = scores[i][j]
                if score < threshold:
                    continue

                # get box size
                box = boxes[i][j]

                # add result(class, id, score, box)
                convert_res_for_img.append([cid, score, box[0], box[1], box[2], box[3]])

                # do callback function if needed
                if self._one_detect_callback is not None:
                    self._one_detect_callback(i, cid, score, box[0], box[1], box[2], box[3])
            convert_res_for_imgs.append(convert_res_for_img)

            # create neo result
            result = NeoInferResult(np.array(convert_res_for_imgs))
            return result


class MXNetResultConverter(NeoResultConverter):
    def __init__(self, one_detect_callback=None, one_image_callback=None):
        super(MXNetResultConverter, self).__init__(one_detect_callback, one_image_callback)

    def convert_result(self, origin_result, threshold):
        convert_res_for_imgs = []
        for res_for_img in origin_result[0]:
            convert_res_for_img = []
            for det in res_for_img:
                # get class id
                cid = int(det[0])
                if cid < 0:
                    continue

                # get score (check if it is the threshold)
                score = det[1]
                if score < threshold:
                    continue

                # get box size
                (left, right, top, bottom) = (det[2], det[4], det[3], det[5])

                # add result(class id, score, box)
                convert_res_for_img.append([cid, score, bottom, left, top, right])
            convert_res_for_imgs.append(convert_res_for_img)

        # create neo result
        result = NeoInferResult(np.array(convert_res_for_imgs))
        return result
