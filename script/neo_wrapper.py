from model_loader import ModelLoaderFactory, get_transpose_tuple
import util
import dlr


class SageMakerNeoWrapper:
    def __init__(self, params):
        self.__model = None
        #self.__model_loader = None
        self.__params = params

    def load(self):
        # load model data
        loader = ModelLoaderFactory.get_loader(self.__params.model_define, self.__params.model_root_path)
        loader.setup()
        model_path = loader.get_model_path()

        # create Deep Learning Runtime
        #self.__model_loader = loader
        self.__model = dlr.DLRModel(model_path, self.__params.target_device)

    def run(self, img):
        if self.__model is None:
            raise NotLoadException("SageMakerNeo Runtime is not initialized! Please call 'load' function.")

        # create input data
        model_define = self.__params.model_define
        transpose_tuple = get_transpose_tuple(model_define)
        input_size = self.__params.model_define["input_size"]
        img_ndarray = util.open_and_norm_image(img, input_size, transpose_tuple)
        input_tensor = img_ndarray.astype("float32")
        input_data = util.get_input_data(model_define, input_tensor)

        # run inference
        m_out = self.__model.run(input_data)

        # get result

        return m_out[0][0]


    def _convert_result(self, origin_result):



class NotLoadException(Exception)
    pass


class NeoParameters:
    def __init__(self, model_define, model_root_path, target_device):
        self.model_define = model_define
        self.model_root_path = model_root_path
        self.target_device = target_device


class NeoInferResult:
    def __init__(self):
        # TODO : imp
        pass
