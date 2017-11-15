#!/bin/bash
# typically executed in your home directory /home/ec2-user/
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh
export PATH="$CONDA_PREFIX/miniconda2/bin:$PATH"
