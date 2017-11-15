#!/bin/bash
conda create --copy --name keras-training python=2.7.13
conda install --name keras-training --file keras-training.conda 
source activate keras-training
pip install -r keras-training.pip
cp activateROOT.sh  $CONDA_PREFIX/etc/conda/activate.d/activateROOT.sh 

# For GPU support:
wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl
pip install --ignore-installed  --upgrade tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl
pip install setGPU
