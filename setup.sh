source activate keras-training
export PYTHONPATH=`pwd`/models:`pwd`/layers:$PYTHONPATH
export KERASTRAINING=`pwd`
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
