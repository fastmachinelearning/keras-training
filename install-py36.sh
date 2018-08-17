#!/bin/bash
conda create --copy --name keras-training-py36 python=3.6.6
conda install --name keras-training-py36 --file keras-training-py36.conda
source activate keras-training-py36
pip install -r keras-training-py36.pip
cp activateROOT.sh  $CONDA_PREFIX/etc/conda/activate.d/activateROOT.sh 

# For GPU support:
wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp36-cp36m-linux_x86_64.whl
pip install --ignore-installed  --upgrade tensorflow_gpu-1.0.1-cp36-cp36m-linux_x86_64.whl
pip install setGPU
