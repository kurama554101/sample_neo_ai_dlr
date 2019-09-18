import cv2


def test3():
    #cap = cv2.VideoCapture("videotestsrc ! videoconvert ! gtksink")
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        cv2.imshow("", img)
        key = cv2.waitKey(1)
        if key == 27:  # [esc] key
            break


if __name__ == "__main__":
    test3()
