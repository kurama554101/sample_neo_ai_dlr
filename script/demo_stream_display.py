#!/usr/bin/env python3

import cv2
from argument_parser_util import create_argument_parser, convert_model_define
from neo_wrapper import SageMakerNeoWrapper, NeoParameters


class DisplayType(Enum):
    VGA = (640, 480)
    HDTV_720p = (1280, 720)
    HDTV_1080p = (1920, 1080)


def convert_display_type(arg_display_type):
    if arg_display_type == "vga":
        return DisplayType.VGA.value
    elif arg_display_type == "hdtv_720p":
        return DisplayType.HDTV_720p.value
    elif arg_display_type == "hdtv_1080p":
        return DisplayType.HDTV_1080p.value
    else:
        raise("{} display type is not defined!".format(arg_display_type))


def prepare_neo_wrapper(args):
    # set parameter
    model_define = convert_model_define(args.model_type)
    model_root_path = args.model_root_path
    target_device = args.target_device
    param = NeoParameters(model_define=model_define,
                          model_root_path=model_root_path,
                          target_device=target_device)

    # load wrapper
    wrapper = SageMakerNeoWrapper(param)
    wrapper.load()
    return wrapper


def main():
    # get argument from parser
    parser = create_argument_parser()
    parser.add_argument("--display_type", default="vga")
    args = parser.parse_args()

    # get neo wrapper
    wrapper = prepare_neo_wrapper(args)

    # get capture
    cap = cv2.VideoCapture(0)
    display_size = convert_display_type(args.display_type)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])

    # start loop
    while True:
        # read video capture
        ret, capture_image = cap.read()
        if (ret == False):
            continue

        # get result
        out = wrapper.run(original_images=[capture_image], output_size=display_size)

        # display image with bounding boxes
        out_frame = out.get_images()[0]
        cv2.imshow('Single-Threaded Detection', out_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
