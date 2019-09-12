import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase


GST_OVERLAY_ML = 'gstoverlayml'


# https://lazka.github.io/pgi-docs/GstBase-1.0/classes/BaseTransform.html
class GstOverlayML(GstBase.BaseTransform):
    CHANNELS = 3  # RGB

    __gstmetadata__ = ("sample",
                       "Transform",
                       "sample",
                       "sample")

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                            Gst.PadDirection.SRC,
                                            Gst.PadPresence.ALWAYS,
                                            Gst.Caps.new_any()),
                        Gst.PadTemplate.new("sink",
                                            Gst.PadDirection.SINK,
                                            Gst.PadPresence.ALWAYS,
                                            Gst.Caps.new_any()))

    def __init__(self):
        super(GstOverlayML, self).__init__()
        self.overlay = None

    def do_transform_ip(self, inbuffer):
        Gst.info("timestamp(buffer):%s" % (Gst.TIME_ARGS(inbuffer.pts)))
        return Gst.FlowReturn.OK



def register(plugin):
    # https://lazka.github.io/pgi-docs/#GObject-2.0/functions.html#GObject.type_register
    type_to_register = GObject.type_register(GstOverlayML)

    # https://lazka.github.io/pgi-docs/#Gst-1.0/classes/Element.html#Gst.Element.register
    return Gst.Element.register(plugin, GST_OVERLAY_ML, 0, type_to_register)


def register_by_name(plugin_name):
    # Parameters explanation
    # https://lazka.github.io/pgi-docs/Gst-1.0/classes/Plugin.html#Gst.Plugin.register_static
    name = plugin_name
    description = "gst.Element draws on image buffer"
    version = '1.12.4'
    gst_license = 'LGPL'
    source_module = 'gstreamer'
    package = 'gstoverlay'
    origin = 'shotasakamoto554101@gmail.com'
    if not Gst.Plugin.register_static(Gst.VERSION_MAJOR, Gst.VERSION_MINOR,
                                      name, description,
                                      register, version, gst_license,
                                      source_module, package, origin):
        raise ImportError("Plugin {} not registered".format(plugin_name))
    return True


def get_buffer_size(caps):
    """
        Returns width, height of buffer from caps
        :param caps: https://lazka.github.io/pgi-docs/Gst-1.0/classes/Caps.html
        :type caps: Gst.Caps
        :rtype: bool, (int, int)
    """

    caps_struct = caps.get_structure(0)
    (success, width) = caps_struct.get_int('width')
    if not success:
        return False, (0, 0)
    (success, height) = caps_struct.get_int('height')
    if not success:
        return False, (0, 0)
    return True, (width, height)


register_by_name(GST_OVERLAY_ML)
