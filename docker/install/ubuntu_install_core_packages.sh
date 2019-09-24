#!/usr/bin/env bash

# save directory
v_dir=`pwd`

# install core packages
apt-get install -y git wget vim

# install c compiler
apt-get install -yq make gcc g++ unzip build-essential gcc zlib1g-dev

# install cmake
mkdir tmp_cmake
cd tmp_cmake
wget https://github.com/Kitware/CMake/releases/download/v3.6.2/cmake-3.6.2.tar.gz
tar xvf cmake-3.6.2.tar.gz
cd cmake-3.6.2
./bootstrap
make
make install
cd ../../
rm -rf tmp_cmake

# clean up
cd ${v_dir}
