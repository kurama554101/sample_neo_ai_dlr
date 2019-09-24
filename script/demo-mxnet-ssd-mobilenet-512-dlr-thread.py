#!/usr/bin/env python3

import os
import numpy as np
import time
import cv2
import dlr
import threading


# set parameter
ms = lambda: int(round(time.time() * 1000))
test_image = "dog.jpg"
dshape = (1, 3, 512, 512)
dtype = "float32"
w = 1280
h = 720
model_path = "model/mxnet-ssd-mobilenet-512"
class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]


def get_result(model, image):
    orig_img, img_data = open_and_norm_image(image)
    input_data = img_data.astype(dtype)
    m_out = model.run(input_data)
    return m_out[0][0]


def display(model, frame, out, thresh=0.5):
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        scales = [frame.shape[1], frame.shape[0]] * 1
        (left, right, top, bottom) = (det[2] * w, det[4] * w,det[3] * h, det[5] * h)
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        cv2.rectangle(frame, p1, p2, (77, 255, 9), 3, 1)
        cv2.putText(frame, class_names[cid], (int(left + 10), int((top+bottom)/2)), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Single-Threaded Detection',frame)


# Preprocess image
def open_and_norm_image(frame):
    # orig_img = cv2.imread(f)
    orig_img = frame
    img = cv2.resize(orig_img, (dshape[2], dshape[3]))
    img = img[:, :, (2, 1, 0)].astype(np.float32)
    img -= np.array([123, 117, 104])
    img = np.transpose(np.array(img), (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return orig_img, img


######################################################################
# Build TVM runtime
device = 'opencl'
m = dlr.DLRModel(model_path, device)

# get capture
#cap = cv2.VideoCapture(8)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

# start loop
while True:
    # read video capture
    ret, test_image = cap.read()
    if(ret == False):
        continue

    # get result
    out = get_result(m, test_image)

    # display image with bounding boxes
    display(m, test_image, out, 0.5)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
