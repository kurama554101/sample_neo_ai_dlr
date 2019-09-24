#!/usr/bin/env bash

# save directory
v_dir=`pwd`

# python 3.6
apt-get install -y python-dev
apt-get install -y software-properties-common

add-apt-repository ppa:jonathonf/python-3.6
apt-get update
apt-get install -y python-pip python-dev python3.6 python3.6-dev python3-setuptools python-setuptools

rm -f /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3

# Install pip
mkdir tmp
cd /tmp && wget -q https://bootstrap.pypa.io/get-pip.py && python2 get-pip.py && python3.6 get-pip.py

# clean up
rm -rf tmp
cd ${v_dir}