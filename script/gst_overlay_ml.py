import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase


GST_OVERLAY_ML = 'gstoverlayml'


# https://lazka.github.io/pgi-docs/GstBase-1.0/classes/BaseTransform.html
class GstOverlayML(GstBase.BaseTransform):
    CHANNELS = 3  # RGB

    __gstmetadata__ = ("An example plugin of GstOverlayOpenCv",
                       "gst-filter/gst_overlay_opencv.py",
                       "gst.Element draw on image",
                       "Taras at LifeStyleTransfer.com")

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                            Gst.PadDirection.SRC,
                                            Gst.PadPresence.ALWAYS,
                                            Gst.Caps.from_string("video/x-raw,format=RGB")),
                        Gst.PadTemplate.new("sink",
                                            Gst.PadDirection.SINK,
                                            Gst.PadPresence.ALWAYS,
                                            Gst.Caps.from_string("video/x-raw,format=RGB")))

    def __init__(self):
        super(GstOverlayML, self).__init__()

        # Overlay could be any of your objects as far as it implements __call__
        # and returns numpy.ndarray
        self.overlay = None

    def do_transform_ip(self, inbuffer):
        """
            Implementation of simple filter.
            All changes affected on Inbuffer
            Read more:
            https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-libs/html/GstBaseTransform.html
        """
#        return Gst.FlowReturn.OK

        success, (width, height) = get_buffer_size(self.srcpad.get_current_caps())
        if not success:
            # https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.FlowReturn
            return Gst.FlowReturn.ERROR

        with map_gst_buffer(inbuffer, Gst.MapFlags.READ) as mapped:
            frame = np.ndarray((height, width, self.CHANNELS), buffer=mapped, dtype=np.uint8)

        overlay = self.overlay()
        x = width - overlay.shape[1]
        y = height - overlay.shape[0]
        draw_image(frame, overlay, x, y)

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


@contextmanager
def map_gst_buffer(pbuffer, flags):
    """
        Map Gst.Buffer for Read/Write
        :param pbuffer: https://lazka.github.io/pgi-docs/Gst-1.0/classes/Buffer.html
        :type pbuffer: Gst.Buffer
        :param flags: https://lazka.github.io/pgi-docs/Gst-1.0/flags.html#Gst.MapFlags
        :type flags: Gst.MapFlags
    """
    if pbuffer is None:
        raise TypeError("Cannot pass NULL to _map_gst_buffer")

    ptr = hash(pbuffer)  # Obraining pointer to buffer
    if flags & Gst.MapFlags.WRITE and _libgst.gst_mini_object_is_writable(ptr) == 0:
        raise ValueError("Writable array requested but buffer is not writeable")

    mapping = _GstMapInfo()
    success = _libgst.gst_buffer_map(ptr, mapping, flags)
    if not success:
        raise RuntimeError("Couldn't map buffer")
    try:
        yield cast(
            mapping.data, POINTER(c_byte * mapping.size)).contents
    finally:
        _libgst.gst_buffer_unmap(ptr, mapping)


register_by_name(GST_OVERLAY_ML)
