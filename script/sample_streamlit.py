import streamlit as st
from face_recognition_util import RealTimeFaceRecognition, VideoCaptureParams, DisplayType, FaceRecognitionMode
from multiprocessing import set_start_method
import platform


def main():
    param = VideoCaptureParams()
    param.size = DisplayType.VGA.value
    param.fps = 30

    rtfr = RealTimeFaceRecognition(
        debug_mode=True,
        video_capture_params=param,
        face_recognition_mode=FaceRecognitionMode.OneFaceRecognitionMode,
        frame_count_with_use_face_recog=10,
        reduction_ratio=4
    )

    with st.spinner('Wait for setup...'):
        rtfr.setup()
    st.success('setup done!')

    with st.spinner("Wait for run..."):
        rtfr.run()
    st.success('run done!')

    # write frame of face recognition
    result = rtfr.get_result()
    st.image(result.FrameData)

    # reload button
    st.button("reload")


if __name__ == "__main__":
    # fix bug of macOS
    if platform.system() == 'Darwin':
        try:
            set_start_method('forkserver')
        except RuntimeError:
            print("RuntimeError: context has already been set")
            pass

    main()
