import traceback
import sys

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

# Initializes Gstreamer, it's variables, paths
Gst.init(sys.argv)

# Need to call after Gst.init
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


def main():
    # Gst.Pipeline https://lazka.github.io/pgi-docs/Gst-1.0/classes/Pipeline.html
    pipeline = Gst.Pipeline()

    # Creates element by name
    # https://lazka.github.io/pgi-docs/Gst-1.0/classes/ElementFactory.html#Gst.ElementFactory.make
    src_name = "my_video_test_src"
    src = Gst.ElementFactory.make("v4l2src", "my_video_test_src")
    src.set_property("device", "/dev/video0")
    src.set_property("io-mode", 4)
    pipeline.add(src)

    # add converter
    convert = Gst.ElementFactory.make("videoconvert")
    pipeline.add(convert)

    # capture
    width = 300
    height = 300
    cap1 = Gst.Caps.from_string("video/x-raw, format=NV12, width={}, height={}".format(width, height))
    camerafilter1 = Gst.ElementFactory.make("capsfilter")
    camerafilter1.set_property("caps", cap1)
    pipeline.add(camerafilter1)

    # add ml function
    ml = Gst.ElementFactory.make("gstoverlayml")
    pipeline.add(ml)

    # add sink
    sink = Gst.ElementFactory.make("autovideosink")
    pipeline.add(sink)

    # create link
    src.link(convert)
    convert.link(camerafilter1)
    camerafilter1.link(ml)
    ml.link(sink)

    assert src == pipeline.get_by_name(src_name)

    # https://lazka.github.io/pgi-docs/Gst-1.0/classes/Bus.html
    bus = pipeline.get_bus()

    # allow bus to emit messages to main thread
    bus.add_signal_watch()

    # Add handler to specific signal
    # https://lazka.github.io/pgi-docs/GObject-2.0/classes/Object.html#GObject.Object.connect
    bus.connect("message", on_message, None)

    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Init GObject loop to handle Gstreamer Bus Events
    loop = GObject.MainLoop()

    try:
        loop.run()
    except:
        traceback.print_exc()

    # Stop Pipeline
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
