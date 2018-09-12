//
// Created by yifeng on 18-8-5.
//

#ifndef PS_ROI_POOLING_H
#define PS_ROI_POOLING_H

#include "tensorflow/core/framework/op_kernel.h"


namespace tensorflow {
    using CPUDevice = Eigen::ThreadPoolDevice;
    using GPUDevice = Eigen::GpuDevice;


    #if GOOGLE_CUDA
    // Partially specialize functor for GpuDevice.
    template <typename T>
    struct PSROIPoolingFunctor_FW_GPU {
        void operator()(
                const GPUDevice& device,
                const T* bottom_data,
                const T* bottom_rois,
                const float spatial_scale,
                const int roi_nums, const int channels,
                const int height, const int width,
                const int pooled_height, const int pooled_width,
                const int output_dim,
                const int group_size,
                T* top_data
        );
    };

    template <typename T>
    struct PSROIPoolingFunctor_BP_GPU{
        void operator()(
                const GPUDevice &device,
                const T* top_diff,
                const T* bottom_rois,
                const T spatial_scale,
                const int roi_nums, const int channels,
                const int height, const int width,
                const int pooled_height, const int pooled_width,
                const int output_dim,
                T* bottom_diff
        );
    };

    template <typename T>
    struct setZero_GPU{
        void operator() (const GPUDevice& d, const int n, T* result_data);
    };
    #endif

}
#endif //PS_ROI_POOLING_H



