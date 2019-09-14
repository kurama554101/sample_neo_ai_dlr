from object_detection_gst import ObjectDetectionGst, GstConfig


def main():
    # set config for mac
    config = GstConfig()
    config.src_type = "avfvideosrc"
    config.src_extract_property = {}

    # run gstreamer
    gst_wrapper = ObjectDetectionGst(config)
    gst_wrapper.setup()
    gst_wrapper.start_play()


if __name__ == "__main__":
    main()
