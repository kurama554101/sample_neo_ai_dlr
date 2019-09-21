from abc import ABCMeta, abstractmethod
import util
import os
from enum import Enum


class ModelType(Enum):
    ONNX = 0
    TENSORFLOW = 1
    MXNET = 2


class ModelLoaderType(Enum):
    TF_ZOO_LOADER = 0,
    MXNET_REMOTE_LOADER = 1


class ModelDefine(Enum):
    TF_SSD_MOBILE_NET_V2_COCO = \
        {
            "loader_type": ModelLoaderType.TF_ZOO_LOADER,
            "url": "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz",
            "input_tensor_name": "import/image_tensor:0",  # TODO : should be get from tensorflow graph
            "input_size": (300, 300)
        }
    MXNET_SSD_MOBILE_NET_512 = \
        {
            "loader_type": ModelLoaderType.MXNET_REMOTE_LOADER,
            "model_dir_name": "mxnet-ssd-mobilenet-512",
            "url_list": [
                "https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-mobilenet-512/model.params",
                "https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-mobilenet-512/model.json",
                "https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-mobilenet-512/model.so"
            ],
            "input_size": (512, 512),
            "img_transpose": (2, 0, 1)
        }


def get_transpose_tuple(model_define):
    if "img_transpose" in model_define:
        return model_define["img_transpose"]
    else:
        return None


class ModelInfo:
    def __init__(self, model_type, model_path_map):
        self.model_type = model_type
        self.model_path_map = model_path_map


class ModelLoaderFactory:
    @classmethod
    def get_loader(cls, model_define, root_path):
        loader_type = model_define["loader_type"]
        if loader_type == ModelLoaderType.TF_ZOO_LOADER:
            return TfModelZooLoader(root_path=root_path, url=model_define["url"])
        elif loader_type == ModelLoaderType.MXNET_REMOTE_LOADER:
            return MXNetRemoteModelLoader(
                root_path=root_path, model_dir_name=model_define["model_dir_name"], url_list=model_define["url_list"]
            )
        else:
            raise UndefinedModelLoaderError("{} loader type is not defined!".format(loader_type))


class UndefinedModelLoaderError(Exception):
    pass


class AbstractModelLoader:
    __metaclass__ = ABCMeta

    def __init__(self, root_path, model_type):
        self._root_path = root_path
        self._model_type = model_type
        os.makedirs(root_path, exist_ok=True)

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_model_detail(self):
        pass

    @abstractmethod
    def get_model_path(self):
        pass

    @abstractmethod
    def _check_model_path(self):
        pass

    @abstractmethod
    def _get_model_dir_path(self):
        pass


class RemoteArchiveModelLoader(AbstractModelLoader):
    __metaclass__ = ABCMeta

    def __init__(self, root_path, model_type, url):
        super(RemoteArchiveModelLoader, self).__init__(root_path, model_type)
        self._url = url

    def setup(self):
        # check model path
        if self._check_model_path():
            print("{} path is already exist!".format(self._get_model_dir_path()))
            return

        # download archive model
        zip_file_name = self.__get_zip_filename()
        util.download(self._url, os.path.join(self._root_path, zip_file_name))

        # extract archive
        archive_path = self.__get_archive_path()
        util.extract_all(archive_path, self._root_path)
        os.remove(archive_path)

    @abstractmethod
    def get_model_detail(self):
        pass

    @abstractmethod
    def get_model_path(self):
        pass

    def _check_model_path(self):
        model_dir_path = self._get_model_dir_path()
        return os.path.exists(model_dir_path)

    def _get_model_dir_path(self):
        archive_path = self.__get_archive_path()
        return util.get_extract_dir_path(archive_path, self._root_path)

    def __get_archive_path(self):
        zip_file_name = self.__get_zip_filename()
        return os.path.join(self._root_path, zip_file_name)

    def __get_zip_filename(self):
        return os.path.basename(self._url)


class TfModelZooLoader(RemoteArchiveModelLoader):
    def __init__(self, root_path, url):
        super(TfModelZooLoader, self).__init__(root_path, ModelType.TENSORFLOW, url)

    def get_model_detail(self):
        model_file = os.path.join(self._get_model_dir_path(), "frozen_inference_graph.pb")
        model_path_map = {"model_file": model_file}
        return ModelInfo(self._model_type, model_path_map)

    def get_model_path(self):
        model_info = self.get_model_detail()
        return model_info.model_path_map["model_file"]


class RemoteModelLoader(AbstractModelLoader):
    __metaclass__ = ABCMeta

    def __init__(self, root_path, model_type, model_dir_name, url_list):
        super(RemoteModelLoader, self).__init__(root_path, model_type)
        self._model_dir_name = model_dir_name
        self._url_list = url_list

    def setup(self):
        # check model path
        if self._check_model_path():
            print("{} path is already exist!".format(self._get_model_dir_path()))
            return

        # download model data
        model_dir_path = self._get_model_dir_path()
        os.makedirs(model_dir_path, exist_ok=True)
        for url in self._url_list:
            file_name = os.path.basename(url)
            file_path = os.path.join(model_dir_path, file_name)
            util.download(url, file_path)

    @abstractmethod
    def get_model_detail(self):
        pass

    def get_model_path(self):
        return self._get_model_dir_path()

    def _check_model_path(self):
        model_dir_path = self._get_model_dir_path()
        return os.path.exists(model_dir_path)

    def _get_model_dir_path(self):
        return os.path.join(self._root_path, self._model_dir_name)


class MXNetRemoteModelLoader(RemoteModelLoader):
    def __init__(self, root_path, model_dir_name, url_list):
        super(MXNetRemoteModelLoader, self).__init__(root_path=root_path, model_type=ModelType.MXNET, model_dir_name=model_dir_name, url_list=url_list)

    def get_model_detail(self):
        model_path_map = {}
        model_dir_path = self._get_model_dir_path()
        for url in self._url_list:
            ext = os.path.splitext(url)[1]
            if ext == ".params":
                model_path_map["model_params"] = os.path.join(model_dir_path, os.path.basename(url))
            elif ext == ".so":
                model_path_map["model_lib"] = os.path.join(model_dir_path, os.path.basename(url))
            elif ext == ".json":
                model_path_map["model_json"] = os.path.join(model_dir_path, os.path.basename(url))
            else:
                # TODO : throw exception
                print("Irregular file extension! ext is {}".format(ext))
        return ModelInfo(self._model_type, model_path_map)
