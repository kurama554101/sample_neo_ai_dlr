import collections
import face_recognition
import numpy as np


def analyze_face_info_with_worker_process(conn, video_queue, face_ls, face_ns,
                                          debug_mode, get_face_info_func, frame_count_with_use_face_recog,
                                          known_face_encodings, known_face_names):
    while True:
        if not video_queue.empty():
            frame = video_queue.get()
            tmp_face_locations, tmp_face_names = get_face_info_func(frame, known_face_encodings, known_face_names)
            face_ls.extend(tmp_face_locations)
            face_ns.extend(tmp_face_names)

        if len(face_ns) >= frame_count_with_use_face_recog:
            if debug_mode:
                print("face count is max! count is {}".format(len(face_ns)))
            counter = collections.Counter(face_ns)
            fi = counter.most_common()[0]
            conn.send(fi)
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
