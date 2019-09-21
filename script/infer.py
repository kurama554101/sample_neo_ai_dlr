import dlr
from model_loader import ModelLoaderFactory, ModelDefine
import util
from argument_parser_util import create_argument_parser, convert_model_define


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
