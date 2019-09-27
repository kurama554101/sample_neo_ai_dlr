import face_recognition
import cv2
import numpy as np
import os
import csv
from enum import Enum


class DisplayType(Enum):
    VGA = (640, 480)
    HDTV720p = (1280, 720)
    HDTV1080p = (1920, 1080)


class VideoCaptureParams:
    def __init__(self):
        self.size = DisplayType.HDTV720p.value
        self.fps = 30


class RealTimeFaceRecognition:
    def __init__(self,
                 face_image_folder=os.path.join("data", "face_data"),
                 face_csv=os.path.join("data", "face_list.csv"),
                 check_frame_count=2, reduction_ratio=4,
                 video_capture_params=VideoCaptureParams(),
                 debug_mode=False):
        self.__capture = None
        self.__face_image_folder = face_image_folder
        self.__face_csv = face_csv
        self.__known_face_encodings = []
        self.__known_face_names = []
        self.__check_frame_count = check_frame_count
        self.__reduction_ratio = reduction_ratio
        self.__video_capture_params = video_capture_params
        self.__debug_mode = debug_mode

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

    def capture_video(self):
        current_frame_count = 1
        face_locations = []
        face_names = []

        while True:
            process_this_frame = (current_frame_count % self.__check_frame_count) == 0
            self.do_capture_frame(process_this_frame, face_locations, face_names)
            current_frame_count = 0 if process_this_frame else current_frame_count+1

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # if loop is end, video capture instance is released
        self.__capture.release()
        cv2.destroyAllWindows()

    def do_capture_frame(self, process_this_frame, face_locations, face_names):
        ret, frame = self.__capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=1/self.__reduction_ratio, fy=1/self.__reduction_ratio)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.__known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.__known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.__known_face_names[best_match_index]

                face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= self.__reduction_ratio
            right *= self.__reduction_ratio
            bottom *= self.__reduction_ratio
            left *= self.__reduction_ratio

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)


if __name__ == "__main__":
    param = VideoCaptureParams()
    recognition = RealTimeFaceRecognition(debug_mode=True, video_capture_params=param)
    recognition.setup()
    recognition.capture_video()
