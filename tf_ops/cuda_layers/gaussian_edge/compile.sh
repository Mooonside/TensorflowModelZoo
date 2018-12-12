#!/bin/bash
TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
nvcc -std=c++11 -c -o gaussian_edge.cu.o gaussian_edge.cu.cc \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I /usr/local -L /usr/local/cuda/lib64/ \
    --expt-relaxed-constexpr ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))

g++ -std=c++11 -shared -o gaussian_edge.so gaussian_edge.cc gaussian_edge.cu.o \
    -fPIC -lcudart -L $CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors \
    -I $CUDA_HOME/include -D_GLIBCXX_USE_CXX11_ABI=0 ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}
