#!/usr/bin/env python3

import os
import numpy as np
import time
import cv2
import dlr
from argument_parser_util import create_argument_parser, convert_model_define
from model_loader import ModelLoaderFactory
from enum import Enum
import util


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


# set parameter
class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]


def get_result(model, model_define, image, input_size):
    dtype = "float32"
    orig_img, img_data = open_and_norm_image(image, input_size)
    input_tensor = img_data.astype(dtype)
    input_data = util.get_input_data(model_define, input_tensor)
    m_out = model.run(input_data)
    return m_out[0][0]


def display(frame, w, h, out, thresh=0.5):
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        scales = [frame.shape[1], frame.shape[0]] * 1
        (left, right, top, bottom) = (det[2] * w, det[4] * w, det[3] * h, det[5] * h)
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        cv2.rectangle(frame, p1, p2, (77, 255, 9), 3, 1)
        cv2.putText(
            frame, class_names[cid], (int(left + 10), int((top+bottom)/2)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
        )
    cv2.imshow('Single-Threaded Detection', frame)


# Preprocess image
def open_and_norm_image(frame, input_size):
    orig_img = frame
    img = cv2.resize(orig_img, input_size)
    img = img[:, :, (2, 1, 0)].astype(np.float32)
    img -= np.array([123, 117, 104])
    img = np.transpose(np.array(img), (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return orig_img, img


def prepare_model(args):
    # set model type
    model_define = convert_model_define(args.model_type)

    # load model data
    model_root_path = args.model_root_path
    loader = ModelLoaderFactory.get_loader(model_define, model_root_path)
    loader.setup()
    model_path = loader.get_model_path()

    # create Deep Learning Runtime
    target = args.target_device
    return dlr.DLRModel(model_path, target)


def main():
    # get argument from parser
    parser = create_argument_parser()
    parser.add_argument("--display_type", default="vga")
    args = parser.parse_args()

    # get model
    m = prepare_model(args)

    # get capture
    cap = cv2.VideoCapture(0)
    display_type = convert_display_type(args.display_type)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_type[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_type[1])

    # get parameter
    model_define = convert_model_define(args.model_type)
    input_size = model_define["input_size"]

    # start loop
    while True:
        # read video capture
        ret, capture_image = cap.read()
        if (ret == False):
            continue

        # get result
        out = get_result(m, model_define, capture_image, input_size)

        # display image with bounding boxes
        display(capture_image, display_type[0], display_type[1], out, 0.5)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
