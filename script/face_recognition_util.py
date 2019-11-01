import face_recognition
import numpy as np
import cv2
import os
import csv
import collections
from multiprocessing import cpu_count, Manager
from concurrent.futures import ProcessPoolExecutor
from enum import Enum


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


class FaceRecognitionResult:
    def __init__(self):
        self.FrameData = None
        self.FaceName = None


class RealTimeFaceRecognition:
    def __init__(self,
                 face_image_folder=os.path.join("data", "face_data"),
                 face_csv=os.path.join("data", "face_list.csv"),
                 face_recognition_mode=FaceRecognitionMode.DrawBoundingBoxMode,
                 get_frame_per_count=2, reduction_ratio=4,
                 video_capture_params=VideoCaptureParams(),
                 frame_count_with_use_face_recog=50,
                 process_count=cpu_count(),
                 debug_mode=False):
        self.__capture = None
        self.__face_image_folder = face_image_folder
        self.__face_csv = face_csv
        self.__face_recognition_mode = face_recognition_mode
        self.__get_frame_per_count = get_frame_per_count
        self.__reduction_ratio = reduction_ratio
        self.__video_capture_params = video_capture_params
        self.__frame_count_with_use_face_recog = frame_count_with_use_face_recog
        self.__process_count = process_count
        self.__debug_mode = debug_mode

        # local value
        self.__known_face_encodings = []
        self.__known_face_names = []
        self.__result = None

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

                # check the cache data of face encoding
                face_encoding_cache_path_without_ext = \
                    os.path.join(self.__face_image_folder, os.path.splitext(os.path.basename(face_file_name))[0])
                face_encoding_cache_path = face_encoding_cache_path_without_ext + ".npy"
                if os.path.exists(face_encoding_cache_path):
                    if self.__debug_mode:
                        print("{} file is exist! face encoding data is loaded from it.".format(face_encoding_cache_path))

                    # load face encoding from cache data
                    encoding = np.load(face_encoding_cache_path)
                else:
                    # create face encoding
                    image = face_recognition.load_image_file(face_file_path)
                    encoding = face_recognition.face_encodings(image)[0]

                    # save cache data
                    np.save(face_encoding_cache_path_without_ext, encoding)

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

    def get_result(self):
        return self.__result

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
        manager = Manager()

        # create queue to save video frame
        q = manager.Queue()

        # create queue to save face recognition result
        # face recognition result is tuple(face_location, face_name)
        fq = manager.Queue()

        # start the process to do face recognition
        executor = ProcessPoolExecutor(max_workers=self.__process_count-1)
        for i in range(self.__process_count):
            executor.submit(analyze_face_info_with_worker_process,
                            q, fq, self.__debug_mode, get_face_information,
                            self.__frame_count_with_use_face_recog,
                            self.__known_face_encodings, self.__known_face_names,)

        # start video capture
        while True:
            frame, rgb_small_frame = self.__get_frame()
            q.put(rgb_small_frame)

            # Hit 'q' on the keyboard to quit
            # if face queue size is max size, video capture is end
            if (cv2.waitKey(1) & 0xFF == ord('q')) or fq.qsize() >= self.__frame_count_with_use_face_recog:
                if self.__debug_mode:
                    print("video capture is end.")
                break

        # get face info
        face_name_list = []
        while not fq.empty():
            face_name_list.append(fq.get()[1])

        counter = collections.Counter(face_name_list)
        face_info = counter.most_common()[0]
        print("face_info is {}".format(face_info))

        # draw face info into frame
        frame, rgb_small_frame = self.__get_frame()
        face_locations = [face_recognition.face_locations(rgb_small_frame)[0]]
        face_names = [face_info[0]]
        self.__draw_boxes_into_frame(frame, face_locations, face_names, self.__reduction_ratio)

        # wait subprocess and terminate queue and process
        manager.shutdown()
        executor.shutdown()

        # display frame until exit command is received
        result = FaceRecognitionResult()
        result.FrameData = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # need to convert from BGR to RGB because frame data is BGR color
        result.FaceName = face_names[0]
        self.__result = result

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


def analyze_face_info_with_worker_process(video_queue, face_information_queue,
                                          debug_mode, get_face_info_func, frame_count_with_use_face_recog,
                                          known_face_encodings, known_face_names):
    while True:
        if video_queue.qsize() > 0:
            frame = video_queue.get()
            tmp_face_locations, tmp_face_names = get_face_info_func(frame, known_face_encodings, known_face_names)
            for face_location, face_name in zip(tmp_face_locations, tmp_face_names):
                face_information_queue.put((face_location, face_name))

        # TODO : should send exit code from main process
        if (cv2.waitKey(1) & 0xFF == ord('q')) or face_information_queue.qsize() >= frame_count_with_use_face_recog:
            if debug_mode:
                print("face count is max! count is {}".format(face_information_queue.qsize()))
            break


def get_face_information(frame, known_face_encodings, known_face_names):
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names
