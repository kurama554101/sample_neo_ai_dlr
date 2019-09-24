#!/usr/bin/env bash
# this script is used to install all packages without using DockerFile

# update apt-get
apt-get update --fix-missing

# install core packages
bash docker/install/ubuntu_install_core_packages.sh

# install python and pip
bash docker/install/ubuntu_install_python.sh

# install opencv
bash docker/install/ubuntu_install_opencv.sh

# install gstreamer
bash docker/install/ubuntu_install_gstreamer.sh

# install numpy, tensorflow and python wrapper of opencv
pip3 install numpy tensorflow opencv-python

# install sagemaker neo
bash docker/install/ubuntu_install_sagemaker_neo.sh

# install other python packages
pip3 install psutil && pip install psutil
pip3 install pillow && pip install pillow

# install tensorflow models
git clone --recursive https://github.com/tensorflow/models
pip3 install models/research && pip install models/research
