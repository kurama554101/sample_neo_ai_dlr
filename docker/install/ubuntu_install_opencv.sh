#!/usr/bin/env bash

# save directory
v_dir=`pwd`

# get core packages for opencv
apt-get install -y libsm6 libxrender1

# build opencv
ln -s /usr/include/libv4l1-videodev.h /usr/include/linux/videodev.h
mkdir tmp
cd tmp && wget https://github.com/Itseez/opencv/archive/3.1.0.zip && unzip 3.1.0.zip
cd tmp/opencv-3.1.0 && cmake CMakeLists.txt -DWITH_TBB=ON \
                                            -DINSTALL_CREATE_DISTRIB=ON \
                                            -DWITH_FFMPEG=OFF \
                                            -DWITH_IPP=OFF \
                                            -DCMAKE_INSTALL_PREFIX=/usr/local
cd tmp/opencv-3.1.0 && make -j2 && make install

# clean up
rm -rf tmp
cd ${v_dir}

