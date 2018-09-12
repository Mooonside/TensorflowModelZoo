//
// Created by yifeng on 18-8-16.
//

#ifndef ROI_ALIGN_H
#define ROI_ALIGN_H

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
    using CPUDevice = Eigen::ThreadPoolDevice;
    using GPUDevice = Eigen::GpuDevice;


#if GOOGLE_CUDA
    template <typename T>
    struct Functor_ROIALIGN_FW_GPU {
        void operator()(
                const GPUDevice& device,
                const T* bottom_data,
                const T* bottom_rois,
                const float spatial_scale, const int sample_ratio,
                const int roi_nums, const int channels,
                const int height, const int width,
                const int pooled_height, const int pooled_width,
                T* top_data
        );
    };

    template <typename T>
    struct Functor_ROIALIGN_BP_GPU{
        void operator()(
                const GPUDevice &device,
                const T *top_diff,
                const T *bottom_rois,
                const T spatial_scale,
                const int sample_ratio,
                const int roi_nums,
                const int channels,
                const int height,
                const int width,
                const int pooled_height,
                const int pooled_width,
                T *bottom_diff
        );
    };

    template <typename T>
    struct setZero_GPU{
        void operator() (const GPUDevice& d, const int n, T* result_data);
    };

#endif

}




#endif //ROI_ALIGN_H
