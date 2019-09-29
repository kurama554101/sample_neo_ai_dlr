import face_recognition
import cv2
import os
import csv
from enum import Enum
from multiprocessing import Queue, Process, Pipe, set_start_method
import platform
from face_recognition_util import analyze_face_info_with_worker_process, get_face_information


class DisplayType(Enum):
    VGA = (640, 480)
    HDTV720p = (1280, 720)
    HDTV1080p = (1920, 1080)


class FaceRecognitionMode(Enum):
    DrawBoundingBoxMode = 0
    OneFaceRecognitionMode = 1


class VideoCaptureParams:
    def __init__(self):
        self.size = DisplayType.HDTV720p.value
        self.fps = 30


class FaceRecognitionError(Exception):
    pass


class RealTimeFaceRecognition:
    def __init__(self,
                 face_image_folder=os.path.join("data", "face_data"),
                 face_csv=os.path.join("data", "face_list.csv"),
                 face_recognition_mode=FaceRecognitionMode.DrawBoundingBoxMode,
                 get_frame_per_count=2, reduction_ratio=4,
                 video_capture_params=VideoCaptureParams(),
                 frame_count_with_use_face_recog=50,
                 debug_mode=False):
        self.__capture = None
        self.__face_image_folder = face_image_folder
        self.__face_csv = face_csv
        self.__face_recognition_mode = face_recognition_mode
        self.__get_frame_per_count = get_frame_per_count
        self.__reduction_ratio = reduction_ratio
        self.__video_capture_params = video_capture_params
        self.__frame_count_with_use_face_recog = frame_count_with_use_face_recog
        self.__debug_mode = debug_mode

        # local value
        self.__known_face_encodings = []
        self.__known_face_names = []

    def setup(self):
        # set video capture
        self.__set_video_capture()

        # load face image
        self.__load_face_image()

    def __set_video_capture(self):
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FPS, self.__video_capture_params.fps)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.__video_capture_params.size[0])
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__video_capture_params.size[1])
        self.__capture = video_capture

        # if debug_mode is true, video capture parameters are printed.
        if self.__debug_mode:
            print("fps : {}, capture width : {}, capture height : {}".format(
                self.__capture.get(cv2.CAP_PROP_FPS),
                self.__capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                self.__capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ))

    def __load_face_image(self):
        # read csv to load face image
        with open(self.__face_csv, "r") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            for row in reader:
                face_file_name = row[0]
                face_name = row[1]
                face_file_path = os.path.join(self.__face_image_folder, face_file_name)

                # create face encoding
                image = face_recognition.load_image_file(face_file_path)
                encoding = face_recognition.face_encodings(image)[0]

                # set face data into list
                self.__known_face_encodings.append(encoding)
                self.__known_face_names.append(face_name)

        # print face list if debug_mode is true
        if self.__debug_mode:
            print("face data count is {}".format(len(self.__known_face_names)))

    def run(self):
        if self.__face_recognition_mode == FaceRecognitionMode.DrawBoundingBoxMode:
            self.__start_face_recognition_with_drawing_bounding_box()
        elif self.__face_recognition_mode == FaceRecognitionMode.OneFaceRecognitionMode:
            self.__start_one_face_recognition()
        else:
            raise FaceRecognitionError("{} mode is not defined!")

    def __start_face_recognition_with_drawing_bounding_box(self):
        current_frame_count = 1
        face_locations = []
        face_names = []

        while True:
            process_this_frame = (current_frame_count % self.__get_frame_per_count) == 0
            self.__do_capture_frame(process_this_frame, face_locations, face_names)
            current_frame_count = 0 if process_this_frame else current_frame_count+1

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # if loop is end, video capture instance is released
        self.__capture.release()
        cv2.destroyAllWindows()

    def __start_one_face_recognition(self):
        face_locations = []
        face_names = []

        # create queue to save video frame
        q = Queue()

        # start the process to do face recognition
        recv_connection, send_connection = Pipe()
        p = Process(target=analyze_face_info_with_worker_process,
                    args=(send_connection, q, face_locations, face_names,
                          self.__debug_mode, get_face_information, self.__frame_count_with_use_face_recog,
                          self.__known_face_encodings, self.__known_face_names))
        p.start()

        # start video capture
        while True:
            _, rgb_small_frame = self.__get_frame()
            q.put(rgb_small_frame)

            # Hit 'q' on the keyboard to quit
            # if face information is received, video frame process is stopped
            if (cv2.waitKey(1) & 0xFF == ord('q')) or recv_connection.poll():
                if self.__debug_mode:
                    print("video capture is end.")
                break

        # get face info
        face_info = recv_connection.recv()
        print("face_info is {}".format(face_info))

        # TODO : destroy each process
        recv_connection.close()

    def __do_capture_frame(self, process_this_frame, face_locations, face_names):
        frame, rgb_small_frame = self.__get_frame()

        # Only process every other frame of video to save time
        if process_this_frame:
            face_locations, face_names = get_face_information(rgb_small_frame, self.__known_face_encodings, self.__known_face_names)

        # draw boxes into frame
        self.__draw_boxes_into_frame(frame, face_locations, face_names, self.__reduction_ratio)

        # Display the resulting image
        cv2.imshow('Video', frame)

    def __get_frame(self):
        ret, frame = self.__capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=1 / self.__reduction_ratio, fy=1 / self.__reduction_ratio)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        return frame, rgb_small_frame

    def __draw_boxes_into_frame(self, frame, face_locations, face_names, reduction_ratio):
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= reduction_ratio
            right *= reduction_ratio
            bottom *= reduction_ratio
            left *= reduction_ratio

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


if __name__ == "__main__":
    # fix bug of macOS
    if platform.system() == 'Darwin':
        set_start_method('forkserver')

    param = VideoCaptureParams()
    recognition = RealTimeFaceRecognition(debug_mode=True,
                                          video_capture_params=param,
                                          face_recognition_mode=FaceRecognitionMode.OneFaceRecognitionMode
                                          )

    # setup
    print("start to setup...")
    recognition.setup()

    # run
    print("start to run..")
    recognition.run()

    # end
    print("end")
