#!/bin/bash
# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
# echo $TF_INC
nvcc -std=c++11 -c -o ps_roi_pooling.cu.o ps_roi_pooling.cu.cc -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I /usr/local -L /usr/local/cuda/lib64/ --expt-relaxed-constexpr ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}
# nvcc -std=c++11 -c -o ps_roi_pooling.cu.o ps_roi_pooling.cu.cc -I $TF_INC -I $TF_INC/external/nsync/public -I /usr/local -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda/lib64/ --expt-relaxed-constexpr
