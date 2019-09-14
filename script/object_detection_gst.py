import traceback
import sys

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

# Initializes Gstreamer, it's variables, paths
Gst.init(sys.argv)

# Need to call after Gst.init
import gst_overlay_ml


class GstConfig:
    def __init__(self):
        # source information
        self.src_name = "my_video_test_src"
        self.src_type = "v4l2src"
        self.src_extract_property = {
            "device": "/dev/video0",
            "io-mode": 4
        }

        # capture information
        self.cap_video_type = "video/x-raw"
        self.cap_extract_property = {
            "format": "NV12",
            "witdh": "640",
            "height": "480"
        }

        # sink information
        self.sink_type = "autovideosink"


class ObjectDetectionGst:

    def __init__(self, config):
        self.config = config

        # Gst.Pipeline https://lazka.github.io/pgi-docs/Gst-1.0/classes/Pipeline.html
        self.pipeline = Gst.Pipeline()

    def setup(self):
        # Gst.Pipeline https://lazka.github.io/pgi-docs/Gst-1.0/classes/Pipeline.html
        pipeline = Gst.Pipeline()

        # Creates element by name
        # https://lazka.github.io/pgi-docs/Gst-1.0/classes/ElementFactory.html#Gst.ElementFactory.make
        src = Gst.ElementFactory.make(self.config.src_type, self.config.src_name)
        for k, v in self.config.src_extract_property.items():
            src.set_property(k, v)
        pipeline.add(src)

        # add converter
        convert = Gst.ElementFactory.make("videoconvert")
        pipeline.add(convert)

        # add capture
        capture_string = self.config.cap_video_type
        for k, v in self.config.cap_extract_property.items():
            capture_string = capture_string + ", {}={}".format(k, v)
        cap = Gst.Caps.from_string(capture_string)
        camerafilter = Gst.ElementFactory.make("capsfilter")
        camerafilter.set_property("caps", cap)
        pipeline.add(camerafilter)

        # add ml function
        ml = Gst.ElementFactory.make("gstoverlayml")
        pipeline.add(ml)

        # add sink
        sink = Gst.ElementFactory.make(self.config.sink_type)
        pipeline.add(sink)

        # create link
        src.link(convert)
        convert.link(camerafilter)
        camerafilter.link(ml)
        ml.link(sink)

        # check source
        assert src == pipeline.get_by_name(self.config.src_name)

        # https://lazka.github.io/pgi-docs/Gst-1.0/classes/Bus.html
        bus = pipeline.get_bus()

        # allow bus to emit messages to main thread
        bus.add_signal_watch()

        # Add handler to specific signal
        # https://lazka.github.io/pgi-docs/GObject-2.0/classes/Object.html#GObject.Object.connect
        bus.connect("message", self._on_message, None)

        # set pipeline
        self.pipeline = pipeline

    def start_play(self):
        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)

        # Init GObject loop to handle Gstreamer Bus Events
        loop = GObject.MainLoop()

        try:
            loop.run()
        except:
            traceback.print_exc()

        # Stop Pipeline
        self.pipeline.set_state(Gst.State.NULL)

    def _on_message(self, bus, message, loop):
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
