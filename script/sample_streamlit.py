import streamlit as st
from face_recognition_util import RealTimeFaceRecognition, VideoCaptureParams, DisplayType, FaceRecognitionMode
from multiprocessing import set_start_method
import platform
import time


def main():
    # setup bar
    st.sidebar.title("setup parameter")
    fps = st.sidebar.slider("fps", min_value=10, max_value=60, value=30)
    display_mode_str = st.sidebar.selectbox("display mode", ("vga", "720p", "1080p"))
    frame_count_with_use_face_recog = st.sidebar.slider("frame count with use face recognition", min_value=2, max_value=50, value=10)
    reduction_ratio = st.sidebar.selectbox("reduction ratio", (4, 2))
    param = VideoCaptureParams()
    param.size = get_capture_size(display_mode_str)
    param.fps = fps

    with st.spinner('Wait for setup...'):
        rtfr = get_recognition_module(param, frame_count_with_use_face_recog, reduction_ratio)
    st.success('setup done!')

    with st.spinner("Wait for run..."):
        # rtfr.run()
        time.sleep(1)
    st.success('run done!')

    # write frame of face recognition
    # result = rtfr.get_result()
    # st.image(result.FrameData)

    # reload button
    st.sidebar.button("reload")


def get_capture_size(str_arg):
    if str_arg == "vga":
        return DisplayType.VGA.value
    elif str_arg == "720p":
        return DisplayType.HDTV720p.value
    elif str_arg == "1080p":
        return DisplayType.HDTV1080p.value
    else:
        raise Exception("{} capture size is not defined!".format(str_arg))


@st.cache(ignore_hash=True)
def get_recognition_module(param, frame_count_with_use_face_recog, reduction_ratio):
    rtfr = RealTimeFaceRecognition(
        debug_mode=True,
        video_capture_params=param,
        face_recognition_mode=FaceRecognitionMode.OneFaceRecognitionMode,
        frame_count_with_use_face_recog=frame_count_with_use_face_recog,
        reduction_ratio=reduction_ratio
    )
    rtfr.setup()
    return rtfr


@st.cache(ignore_hash=True)
def set_server_method():
    # fix bug of macOS
    if platform.system() == 'Darwin':
        try:
            set_start_method('forkserver')
        except RuntimeError:
            print("RuntimeError: context has already been set")
            pass


if __name__ == "__main__":
    set_server_method()
    main()
