from abc import ABCMeta, abstractmethod
import util
import os
from enum import Enum


class ModelType(Enum):
    ONNX = 0
    TENSORFLOW = 1


class TfModelZooType(Enum):
    SSD_MOBILE_NET_V2_COCO = \
        {
            "url": "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz",
            "input_tensor_name": "import/image_tensor:0",  # TODO : should be get from tensorflow graph
            "input_size": (300, 300)
        }


class ModelInfo:
    def __init__(self, model_type, model_path_map):
        self.model_type = model_type
        self.model_path_map = model_path_map


class AbstractModelLoader:
    __metaclass__ = ABCMeta

    def __init__(self, root_path):
        self._root_path = root_path
        os.makedirs(root_path, exist_ok=True)

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_model(self):
        pass


class RemoteArchiveModelLoader(AbstractModelLoader):
    __metaclass__ = ABCMeta

    def __init__(self, root_path, url):
        super(RemoteArchiveModelLoader, self).__init__(root_path)
        self._url = url
        self._model_root_path = None

    def setup(self):
        # check model path
        zip_file_name = os.path.basename(self._url)
        archive_path = os.path.join(self._root_path, zip_file_name)
        self._model_root_path = util.get_extract_dir_path(archive_path, self._root_path)
        if os.path.exists(self._model_root_path):
            print("{} path is already exist!".format(self._model_root_path))
            return

        # download archive model
        util.download(self._url, os.path.join(self._root_path, zip_file_name))

        # extract archive
        util.extract_all(archive_path, self._root_path)
        os.remove(archive_path)

    @abstractmethod
    def get_model(self):
        pass


class TfModelZooLoader(RemoteArchiveModelLoader):
    def __init__(self, root_path, url):
        super(TfModelZooLoader, self).__init__(root_path, url)

    def get_model(self):
        model_type = ModelType.TENSORFLOW
        model_file = os.path.join(self._model_root_path, "frozen_inference_graph.pb")
        model_path_map = {"model_file": model_file}
        return ModelInfo(model_type, model_path_map)
