from model_loader import ModelLoaderFactory, get_transpose_tuple, ModelType
import util
import dlr
from abc import ABCMeta, abstractmethod
import numpy as np
import os
from PIL import Image
import cv2
from coco import coco


class SageMakerNeoWrapper:
    def __init__(self, params):
        self.__model = None
        self.__model_loader = None
        self.__result_creator = None
        self.__params = params

        # create converter
        self.__one_detect_callback = None
        if self.__params.is_draw_box:
            def callback(image, cid, score, bottom, left, top, right):
                # debug
                print("callback1 : box size is {}, {}, {}, {}".format(bottom, left, top, right))

                p1 = (int(left), int(top))
                p2 = (int(right), int(bottom))
                cv2.rectangle(image, p1, p2, (77, 255, 9), 3, 1)
                cv2.putText(
                    image, coco.IMAGE_CLASSES[cid], (int(left + 10), int((top + bottom) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                )
            self.__one_detect_callback = callback

        self.__one_image_callback = None
        if self.__params.is_save_image_with_box:
            def callback2(image, file_name):
                out_file_name = os.path.splitext(file_name)[0] + "_with_boxes.png"
                Image.fromarray(image).save(out_file_name)
            self.__one_image_callback = callback2

    def load(self):
        # load model data
        loader = ModelLoaderFactory.get_loader(self.__params.model_define, self.__params.model_root_path)
        loader.setup()
        model_path = loader.get_model_path()

        # create Deep Learning Runtime
        self.__model_loader = loader
        self.__model = dlr.DLRModel(model_path, self.__params.target_device)

        # create result creator
        model_type = self.__model_loader.get_model_detail().model_type
        self.__result_creator = NeoResultConverterFactory.get_converter(model_type,
                                                                        self.__one_detect_callback,
                                                                        self.__one_image_callback)

    def run(self, original_images, output_size, file_name_list=None):
        """
        run inference.
        :param original_images: numpy.ndarray
        :param output_size: tuple
            output size. format is (width, height).
        :param file_name_list: list
        :return:
        """
        # check model state and argument
        if self.__model is None:
            raise NotLoadException("SageMakerNeo Runtime is not initialized! Please call 'load' function.")

        if file_name_list is not None and len(original_images) != len(file_name_list):
            raise ArgumentException("images count is not equal file name list count!")

        # copy origin images
        images = np.copy(original_images)

        # create input data
        model_define = self.__params.model_define
        # TODO : move "transpose_tuple" function from model_loader class to this class.
        transpose_tuple = get_transpose_tuple(model_define)
        input_size = self.__params.model_define["input_size"]
        imgs_ndarray = util.open_and_norm_images(images, input_size, transpose_tuple)
        input_tensor = imgs_ndarray.astype("float32")
        input_data = util.get_input_data(model_define, input_tensor)

        # run inference
        result = self.__model.run(input_data)

        # create result
        return self.__result_creator.create_result(original_images,
                                                   result, output_size,
                                                   self.__params.threshold,
                                                   file_name_list)


class NotLoadException(Exception):
    pass


class ArgumentException(Exception):
    pass


class NeoParameters:
    def __init__(self, model_define, model_root_path, target_device,
                 threshold=0.5, is_draw_box=True, is_save_image_with_box=False
                 ):
        self.model_define = model_define.value
        self.model_root_path = model_root_path
        self.target_device = target_device
        self.threshold = threshold
        self.is_draw_box = is_draw_box
        self.is_save_image_with_box = is_save_image_with_box


class NeoInferResult:
    def __init__(self, result, images):
        self.__result = result
        self.__images = images

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

    def get_images(self):
        """
        get images.
        :return:

        images shape is (N
        """
        return self.__images


class NeoResultConverterFactory:
    @classmethod
    def get_converter(cls, model_type, one_detect_callback=None, one_image_callback=None):
        if model_type == ModelType.TENSORFLOW:
            return TFResultCreator(one_detect_callback, one_image_callback)
        elif model_type == ModelType.MXNET:
            return MXNetResultCreator(one_detect_callback, one_image_callback)
        else:
            raise NeoResultConverterNotDefinedError("{} : neo result converter is not defined.".format(model_type))


class NeoResultConverterNotDefinedError(Exception):
    pass


class AbstractNeoResultCreator:
    __metaclass__ = ABCMeta

    def __init__(self, one_detect_callback=None, one_image_callback=None):
        """
        initialize result creator
        :param one_detect_callback: detection callback
            param is [image array, cid, score, bottom, left, top, right]
        :param one_image_callback:
            param is [image array, file name]
        """
        self._one_detect_callback = one_detect_callback
        self._one_image_callback = one_image_callback

    @abstractmethod
    def create_result(self, origin_images, origin_result, output_size, threshold, file_name_list=None):
        pass


class TFResultCreator(AbstractNeoResultCreator):
    def __init__(self, one_detect_callback=None, one_image_callback=None):
        super(TFResultCreator, self).__init__(one_detect_callback, one_image_callback)

    def create_result(self, origin_images, origin_result, output_size, threshold, file_name_list=None):
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
                width = output_size[0]
                height = output_size[1]
                box = boxes[i][j]
                (bottom, left, top, right) = box[0] * height, box[1] * width, box[2] * height, box[3] * width

                # add result(class, id, score, box)
                convert_res_for_img.append([cid, score, bottom, left, top, right])

                # do callback function if needed
                if self._one_detect_callback is not None:
                    self._one_detect_callback(origin_images[i], cid, score, bottom, left, top, right)
            convert_res_for_imgs.append(convert_res_for_img)

            # do callback function if needed
            if self._one_image_callback is not None and file_name_list is not None:
                self._one_image_callback(origin_images[i], file_name_list[i])

        # create neo result
        result = NeoInferResult(np.array(convert_res_for_imgs), origin_images)
        return result


class MXNetResultCreator(AbstractNeoResultCreator):
    def __init__(self, one_detect_callback=None, one_image_callback=None):
        super(MXNetResultCreator, self).__init__(one_detect_callback, one_image_callback)

    def create_result(self, origin_images, origin_result, output_size, threshold, file_name_list=None):
        convert_res_for_imgs = []
        for i, res_for_img in enumerate(origin_result[0]):
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
                width = output_size[0]
                height = output_size[1]
                (left, right, top, bottom) = (det[2] * width, det[4] * width, det[3] * height, det[5] * height)

                # add result(class id, score, box)
                convert_res_for_img.append([cid, score, bottom, left, top, right])

                # do callback function if needed
                if self._one_detect_callback is not None:
                    self._one_detect_callback(origin_images[i], cid, score, bottom, left, top, right)
            convert_res_for_imgs.append(convert_res_for_img)

            # do callback function if needed
            if self._one_image_callback is not None and file_name_list is not None:
                self._one_image_callback(origin_images[i], file_name_list[i])

        # create neo result
        result = NeoInferResult(np.array(convert_res_for_imgs), origin_images)
        return result
