import traceback
import sys
import cv2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

# Initializes Gstreamer, it's variables, paths
Gst.init(sys.argv)

import gst_overlay_ml


def on_message(bus, message, loop):
    mtype = message.type
    """
        Gstreamer Message Types and how to parse
        https://lazka.github.io/pgi-docs/Gst-1.0/flags.html#Gst.MessageType
    """
    if mtype == Gst.MessageType.EOS:
        print("End of stream")

    elif mtype == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(err, debug)
    elif mtype == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(err, debug)

    return True


def test():
    pipeline = Gst.Pipeline()

    src = Gst.ElementFactory.make("videotestsrc")
    pipeline.add(src)
    print(type(src))
    print(type(gst_overlay_ml.GstOverlayML()))

    sink = Gst.ElementFactory.make("gtksink")
    pipeline.add(sink)

    src.link(sink)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, None)

    # Start pipeline
    print("start pipeline..")
    pipeline.set_state(Gst.State.PLAYING)
    loop = GObject.MainLoop()
    try:
        loop.run()
    except:
        traceback.print_exc()

    # Stop Pipeline
    pipeline.set_state(Gst.State.NULL)


def test2():
    pipeline = Gst.Pipeline()

    src = Gst.ElementFactory.make("avfvideosrc")
    pipeline.add(src)

    convert = Gst.ElementFactory.make("videoconvert")
    pipeline.add(convert)

    cap = Gst.Caps.from_string("video/x-raw, width=640, height=480")
    camfilter = Gst.ElementFactory.make("capsfilter")
    camfilter.set_property("caps", cap)
    pipeline.add(camfilter)

    sink = Gst.ElementFactory.make("gtksink")
    pipeline.add(sink)

    src.link(convert)
    convert.link(sink)
    # camfilter.link(sink)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, None)

    # Start pipeline
    print("start pipeline..")
    pipeline.set_state(Gst.State.PLAYING)
    loop = GObject.MainLoop()
    try:
        loop.run()
    except:
        traceback.print_exc()

    # Stop Pipeline
    pipeline.set_state(Gst.State.NULL)


def test3():
    cap = cv2.VideoCapture("videotestsrc ! videoconvert ! gtksink")

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
