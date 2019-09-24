#!/usr/bin/env bash

# save directory
v_dir=`pwd`

# install sagemaker neo packages
git clone --recursive https://github.com/neo-ai/neo-ai-dlr
cd neo-ai-dlr && \
mkdir build && cd build && \
cmake .. && \
make -j4 && \
cd ../python && python3 setup.py install --user

# check to enable to use this runtime
python3 -c "import dlr"

# clean up
cd ${v_dir}