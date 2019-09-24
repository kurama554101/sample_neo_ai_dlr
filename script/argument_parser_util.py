import argparse
from model_loader import ModelDefine


def create_argument_parser():
    parser = argparse.ArgumentParser()

    # add parameter
    parser.add_argument(
        "--model_type",
        default="tf_ssd",
        help="set model type. you can select it from 'tf_ssd', 'mx_ssd'. default value is 'tf_ssd'"
    )
    parser.add_argument(
        "--input_file_path",
        default="data/dogs.jpg",
        help="set input file path."
    )
    parser.add_argument(
        "--model_root_path",
        default="model",
        help="set model root path."
    )
    parser.add_argument(
        "--target_device",
        default="cpu",
        help="set target device. you can select it from 'cpu', 'opencl' and so on. default value is 'cpu'"
    )
    return parser


def convert_model_define(arg_model_type):
    if arg_model_type == "tf_ssd":
        return ModelDefine.TF_SSD_MOBILE_NET_V2_COCO
    elif arg_model_type == "mx_ssd":
        return ModelDefine.MXNET_SSD_MOBILE_NET_512
    else:
        raise Exception("{} model type is not defined in ModelDefine class!".format(arg_model_type))
