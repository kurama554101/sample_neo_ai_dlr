import dlr
from model_loader import ModelLoaderFactory, ModelDefine, ModelLoaderType
import argparse
import util


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
        return ModelDefine.TF_SSD_MOBILE_NET_V2_COCO.value
    elif arg_model_type == "mx_ssd":
        return ModelDefine.MXNET_SSD_MOBILE_NET_512.value
    else:
        raise Exception("{} model type is not defined in ModelDefine class!".format(arg_model_type))


def main():
    # get argument from parser
    parser = create_argument_parser()
    args = parser.parse_args()

    # set model type
    model_define = convert_model_define(args.model_type)

    # load model data
    model_root_path = args.model_root_path
    loader = ModelLoaderFactory.get_loader(model_define, model_root_path)
    loader.setup()
    model_path = loader.get_model_path()

    # create Deep Learning Runtime
    target = args.target_device
    m = dlr.DLRModel(model_path, target)

    # get input data
    input_files = [args.input_file_path]
    image_size = model_define["input_size"]
    input_tensor = util.get_ndarray_from_image(input_files, image_size)
    if "input_tensor_name" in model_define.keys():
        input_data = {model_define["input_tensor_name"]: input_tensor}
    else:
        input_data = input_tensor

    # run inference
    res = m.run(input_data)

    # show inference result and recreate image files
    util.recreate_images_with_bounding_boxes(input_files, input_tensor, res)


if __name__ == "__main__":
    main()
