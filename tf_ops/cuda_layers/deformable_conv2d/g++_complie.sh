#!/bin/bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
# echo $TF_INC
if [ ! -f $TF_INC/tensorflow/stream_executor/cuda/cuda_config.h ]; then
    cp ./cuda_config.h $TF_INC/tensorflow/stream_executor/cuda/
fi
# g++ -std=c++11 -shared -o deform_conv.so deform_conv.cc deform_conv.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors -I $CUDA_HOME/include -D_GLIBCXX_USE_CXX11_ABI=0  -L$TF_LIB -ltensorflow_framework -I $TF_INC/external/nsync/public
g++ -std=c++11 -shared -o deform_conv.so deform_conv.cc deform_conv.cu.o -fPIC -lcudart -L $CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors -I $CUDA_HOME/include -D_GLIBCXX_USE_CXX11_ABI=0 ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}
