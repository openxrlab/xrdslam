#!/bin/bash

cd ./sparse_octree/
python setup.py install

cd ../sparse_voxels/
python setup.py install

cd ../dpvo_ext/
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d .
python setup.py install

cd ../evaluate_3d_reconstruction_lib/
python setup.py install
