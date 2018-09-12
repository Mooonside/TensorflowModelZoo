#!/bin/bash
TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))

nvcc -std=c++11 -c -o roi_align.cu.o roi_align.cu.cc -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I /usr/local -L /usr/local/cuda/lib64/ --expt-relaxed-constexpr ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}


# echo $TF_INC
if [ ! -f $TF_INC/tensorflow/stream_executor/cuda/cuda_config.h ]; then
    cp ./cuda_config.h $TF_INC/tensorflow/stream_executor/cuda/
fi

g++ -std=c++11 -shared -o roi_align.so roi_align.cc roi_align.cu.o -fPIC -lcudart -L $CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors -I $CUDA_HOME/include -D_GLIBCXX_USE_CXX11_ABI=0 ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}
