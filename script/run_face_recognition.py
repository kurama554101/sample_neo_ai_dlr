import time
from multiprocessing import set_start_method
import platform
from face_recognition_util import FaceRecognitionMode, VideoCaptureParams, RealTimeFaceRecognition, DisplayType
import streamlit as st


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
    recognition = RealTimeFaceRecognition(debug_mode=True,
                                          video_capture_params=param,
                                          face_recognition_mode=FaceRecognitionMode.OneFaceRecognitionMode,
                                          frame_count_with_use_face_recog=frame_count_with_use_face_recog,
                                          reduction_ratio=reduction_ratio
                                          )
    recognition.setup()
    return recognition


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

    # set sidebar
    st.sidebar.title("setup parameter")
    fps = st.sidebar.slider("fps", min_value=10, max_value=60, value=30)
    display_mode_str = st.sidebar.selectbox("display mode", ("vga", "720p", "1080p"))
    frame_count_with_use_face_recog = st.sidebar.slider("frame count with use face recognition", min_value=2,
                                                        max_value=50, value=10)
    reduction_ratio = st.sidebar.selectbox("reduction ratio", (4, 2))
    st.sidebar.button("reload")

    # create VideoCaptureParams
    param = VideoCaptureParams()
    param.size = get_capture_size(display_mode_str)
    param.fps = fps

    # setup
    print("start to setup...")
    with st.spinner('Wait for setup...'):
        setup_start_time = time.time()
        recognition = get_recognition_module(param, frame_count_with_use_face_recog, reduction_ratio)
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

    # end
    print("end")
