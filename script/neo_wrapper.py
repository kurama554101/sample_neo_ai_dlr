from model_loader import ModelLoaderFactory, get_transpose_tuple, ModelType
import util
import dlr
from abc import ABCMeta, abstractmethod


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
                [class id, score, boxes(left, top, right, bottom)]

                ...
                []
            ]
            ...
            []
        ]
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

    def __init__(self):
        pass

    @abstractmethod
    def convert_result(self, origin_result, threshold):
        pass

    @abstractmethod
    def draw_boxes(self, origin_result, threshold):
        pass


class TFResultConverter(NeoResultConverter):
    def __init__(self):
        super(TFResultConverter, self).__init__()

    def convert_result(self, origin_result, threshold):
        # TODO : imp
        pass

    def draw_boxes(self, origin_result, threshold):
        # TODO : imp
        pass


class MXNetResultConverter(NeoResultConverter):
    def __init__(self):
        super(MXNetResultConverter, self).__init__()

    def convert_result(self, origin_result, threshold):
        # TODO : imp
        pass

    def draw_boxes(self, origin_result, threshold):
        # TODO : imp
        pass
