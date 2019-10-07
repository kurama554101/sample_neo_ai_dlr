import time
import argparse
from multiprocessing import set_start_method
import platform
from face_recognition_util import DisplayType, FaceRecognitionMode, VideoCaptureParams, RealTimeFaceRecognition
import streamlit as st


def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--capture_size",
        default="vga",
        help="set capture size. you can select it from 'vga', '720p', '1080p'. default size is 'vga'"
    )
    parser.add_argument(
        "--capture_fps",
        default=60,
        type=int
    )
    parser.add_argument(
        "--mode",
        default="one_face_recognition",
        help="set face recognition mode. you can select it from 'one_face_recognition', 'draw_bounding_box'"
    )
    parser.add_argument(
        "--reduction_ratio",
        default=4,
        type=int
    )
    parser.add_argument(
        "--count_of_using_frame",
        default=50,
        type=int
    )
    return parser


def get_capture_size(args):
    str_arg = args.capture_size
    if str_arg == "vga":
        return DisplayType.VGA.value
    elif str_arg == "720p":
        return DisplayType.HDTV720p.value
    elif str_arg == "1080p":
        return DisplayType.HDTV1080p.value
    else:
        raise Exception("{} capture size is not defined!".format(str_arg))


def get_mode(args):
    str_arg = args.mode
    if str_arg == "one_face_recognition":
        return FaceRecognitionMode.OneFaceRecognitionMode
    elif str_arg == "draw_bounding_box":
        return FaceRecognitionMode.DrawBoundingBoxMode
    else:
        raise Exception("{} mode is not defined!".format(str_arg))


def get_fps(args):
    return args.capture_fps


def get_reduction_ratio(args):
    return args.reduction_ratio


def get_use_frame_count(args):
    return args.count_of_using_frame


if __name__ == "__main__":
    # fix bug of macOS
    if platform.system() == 'Darwin':
        try:
            set_start_method('forkserver')
        except RuntimeError:
            print("RuntimeError: context has already been set")
            pass

    # get parser and parse argument of command line
    parser = create_argument_parser()
    args = parser.parse_args()
    capture_size = get_capture_size(args)
    mode = get_mode(args)
    capture_fps = get_fps(args)
    frame_count_with_use_face_recog = get_use_frame_count(args)
    reduction_ratio = get_reduction_ratio(args)

    param = VideoCaptureParams()
    param.size = capture_size
    param.fps = capture_fps
    recognition = RealTimeFaceRecognition(debug_mode=True,
                                          video_capture_params=param,
                                          face_recognition_mode=mode,
                                          frame_count_with_use_face_recog=frame_count_with_use_face_recog,
                                          reduction_ratio=reduction_ratio
                                          )

    # setup
    print("start to setup...")
    with st.spinner('Wait for setup...'):
        setup_start_time = time.time()
        recognition.setup()
        setup_process_time = time.time() - setup_start_time
    st.success('setup done! process time is {}'.format(setup_process_time))

    # run
    print("start to run..")
    with st.spinner('Wait for run...'):
        run_start_time = time.time()
        recognition.run()
        run_process_time = time.time() - run_start_time
    st.success('run done! process time is {}'.format(run_process_time))

    # write frame of face recognition
    result = recognition.get_result()
    st.image(result.FrameData)

    # set reload button
    st.button("reload")

    # end
    print("end")
