import cv2


def main():
    # get video capture
    video_capture = get_video_capture(fps=30)

    # set codec of video
    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # read frame and display it
    print("fps is {}, width is {}, height is {}".format(
        video_capture.get(cv2.CAP_PROP_FPS),
        video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ))
    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            break


def get_video_capture(fps=30, frame_width=1920, frame_height=1080):
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FPS, fps)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    return video_capture


if __name__ == "__main__":
    main()
