from flask import Flask, jsonify
from flask_cors import CORS
import platform
from multiprocessing import set_start_method
from face_recognition_util import FaceRecognitionMode, VideoCaptureParams, RealTimeFaceRecognition, DisplayType
from enum import Enum
import time


# global instance
app = Flask(__name__)
CORS(app)
#face_recognition_module = None


# static value
RESULT_CODE_NAME = "result_code"
RESULT_DETAIL_NAME = "result_detail"
RESULT_TIME_NAME = "result_time"
RESULT_FACE_NAME = "result_face_name"


# result code
class ResultCode(Enum):
    OK = 0
    NotInitialized = 1
    UnknownError = 99


@app.route("/face", methods=["GET"])
def face_recognition():
    # initialize (TODO : need to do this function in other function)
    face_recognition_module = initialize_module()

    # check the whether face module is initialized
    if face_recognition_module is None:
        res = {
            RESULT_CODE_NAME: ResultCode.NotInitialized.value,
            RESULT_DETAIL_NAME: "module is not initialized!"
        }
        return jsonify(res)

    # run inference
    run_start_time = time.time()
    face_recognition_module.run()
    run_process_time = time.time() - run_start_time

    # get result and response
    face_result = face_recognition_module.get_result()
    res = {
        RESULT_CODE_NAME: ResultCode.OK.value,
        RESULT_TIME_NAME: run_process_time,
        RESULT_FACE_NAME: face_result.FaceName
    }
    return jsonify(res)


def set_server_method():
    # fix bug of macOS
    if platform.system() == 'Darwin':
        try:
            set_start_method('forkserver')
        except RuntimeError:
            print("RuntimeError: context has already been set")
            pass


def get_recognition_module(param, frame_count_with_use_face_recog, reduction_ratio):
    recognition = RealTimeFaceRecognition(debug_mode=True,
                                          video_capture_params=param,
                                          face_recognition_mode=FaceRecognitionMode.OneFaceRecognitionMode,
                                          frame_count_with_use_face_recog=frame_count_with_use_face_recog,
                                          reduction_ratio=reduction_ratio
                                          )
    recognition.setup()
    return recognition


def initialize_module():
    set_server_method()
    param = VideoCaptureParams()
    return get_recognition_module(param, 50, 4)


if __name__ == "__main__":
    # initialize
    initialize_module()

    # debug mode
    app.debug = True

    # enable to access from each place
    app.run(host='0.0.0.0')
