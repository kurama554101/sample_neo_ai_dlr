#!/usr/bin/env bash

# install v4l-utils
apt-get install -y v4l-utils

# install gstreamer
apt-get install -y libgstreamer1.0-0 gstreamer1.0-plugins-base \
                gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
                gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc \
                gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl \
                gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

# install gstreamer wrapper of python
apt-get install -y python-gst-1.0
