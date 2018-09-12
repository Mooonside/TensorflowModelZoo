//
// Created by yifeng on 18-8-5.
//

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "ps_roi_pooling.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"


using std::max;
using std::min;

// Forward implementations
namespace tensorflow {

    template <typename T>
    __global__ void PSROIPooling_FORWARD_CUDA(
            const int nthreads,
            const T* bottom_data,
            const T* bottom_rois,
            const float spatial_scale,
            const int channels,
            const int height, const int width,
            const int pooled_height, const int pooled_width,
            const int output_dim,
            const int group_size,
            T* top_data) {
        for(int index : CudaGridRangeX(nthreads)) {
            // The output is in order (n, ctop, ph, pw)
            int pw = index % pooled_width;
            int ph = (index / pooled_width) % pooled_height;
            int ctop = (index / pooled_width / pooled_height) % output_dim;
            int n = index / pooled_width / pooled_height / output_dim;

            // [start, end) interval for spatial sampling
            bottom_rois += n * 5;
            int roi_batch_ind = bottom_rois[0];
            T roi_start_w = static_cast<T>(round(bottom_rois[1])) * spatial_scale;
            T roi_start_h = static_cast<T>(round(bottom_rois[2])) * spatial_scale;
            // takes value from [roi_start_w, roi_end_w)
            T roi_end_w = static_cast<T>(round(bottom_rois[3]) + 1.) * spatial_scale;
            T roi_end_h = static_cast<T>(round(bottom_rois[4]) + 1.) * spatial_scale;

            // Force too small ROIs to be 1x1
            T roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
            T roi_height = max(roi_end_h - roi_start_h, 0.1);

            // Compute w and h at bottom
            T bin_size_h = roi_height / static_cast<T>(pooled_height);
            T bin_size_w = roi_width / static_cast<T>(pooled_width);

            int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h + roi_start_h));
            int wstart = static_cast<int>(floor(static_cast<T>(pw)* bin_size_w + roi_start_w));
            int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h + roi_start_h));
            int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w + roi_start_w));
            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart, 0), height);
            hend = min(max(hend, 0), height);
            wstart = min(max(wstart, 0),width);
            wend = min(max(wend, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            int gw = pw;
            int gh = ph;
            int c = (ctop*group_size + gh)*group_size + gw;

            bottom_data += (roi_batch_ind * channels + c) * height * width;
            T out_sum = 0;
            for (int h = hstart; h < hend; ++h){
                for (int w = wstart; w < wend; ++w){
                    int bottom_index = h*width + w;
                    out_sum += bottom_data[bottom_index];
                }
            }

            T bin_area = (hend - hstart)*(wend - wstart);
            top_data[index] = is_empty? 0. : out_sum/bin_area;
        }
    };


    template <typename T>
    void PSROIPoolingFunctor_FW_GPU<T>::operator()(
        const GPUDevice &device,
        const T* bottom_data,
        const T* bottom_rois,
        const float spatial_scale,
        const int roi_nums, const int channels,
        const int height, const int width,
        const int pooled_height, const int pooled_width,
        const int output_dim,
        const int group_size,
        T* top_data)
    {
        int outputs_size = roi_nums * output_dim * pooled_height * pooled_width;
        CudaLaunchConfig config = GetCudaLaunchConfig(outputs_size, device);

        // TODO: initialize top_data and is_mappping information to 0 and -1 respectively

        PSROIPooling_FORWARD_CUDA<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
            outputs_size,
            bottom_data,
            bottom_rois,
            spatial_scale,
            channels,
            height,  width,
            pooled_height, pooled_width,
            output_dim,
            group_size,
            top_data
        );
    }

    template struct PSROIPoolingFunctor_FW_GPU<float>;
    template struct PSROIPoolingFunctor_FW_GPU<double>;
}

// Backward implementations
namespace tensorflow{
    template <typename Dtype>
    __global__ void PSROIPooling_BACKWARD_CUDA(
            const int nthreads,
            const Dtype* top_diff,
            const Dtype* bottom_rois,
            const Dtype spatial_scale,
            const int channels,
            const int height, const int width,
            const int pooled_height, const int pooled_width,
            const int output_dim,
            Dtype* bottom_diff) {
        for(int index : CudaGridRangeX(nthreads)) {
            // The output is in order (n, ctop, ph, pw)
            int pw = index % pooled_width;
            int ph = (index / pooled_width) % pooled_height;
            int ctop = (index / pooled_width / pooled_height) % output_dim;
            int n = index / pooled_width / pooled_height / output_dim;

            // [start, end) interval for spatial sampling
            bottom_rois += n * 5;
            int roi_batch_ind = bottom_rois[0];
            Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
            Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
            Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
            Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

            // Force too small ROIs to be 1x1
            Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
            Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

            // Compute w and h at bottom
            Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
            Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

            int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)* bin_size_h + roi_start_h));
            int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)* bin_size_w + roi_start_w));
            int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h));
            int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w));

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart, 0), height);
            hend = min(max(hend, 0), height);
            wstart = min(max(wstart, 0), width);
            wend = min(max(wend, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Compute c at bottom
            int c = (ctop*pooled_height + ph)*pooled_width + pw;
            Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
            Dtype bin_area = (hend - hstart)*(wend - wstart);
            Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
            for (int h = hstart; h < hend; ++h){
                for (int w = wstart; w < wend; ++w){
                    int bottom_index = h*width + w;
                    CudaAtomicAdd(offset_bottom_diff + bottom_index, diff_val);
//                    caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
                }
            }
        }
    }

    template <typename T>
    void PSROIPoolingFunctor_BP_GPU<T>::operator()(
            const GPUDevice &device,
            const T* top_diff,
            const T* bottom_rois,
            const T spatial_scale,
            const int roi_nums, const int channels,
            const int height, const int width,
            const int pooled_height, const int pooled_width,
            const int output_dim,
            T* bottom_diff
    ){
        int outputs_size = roi_nums * output_dim * pooled_height * pooled_width;
        CudaLaunchConfig config = GetCudaLaunchConfig(outputs_size, device);

        PSROIPooling_BACKWARD_CUDA<<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
                outputs_size,
                top_diff,
                bottom_rois,
                spatial_scale,
                channels,
                height, width,
                pooled_height, pooled_width,
                output_dim,
                bottom_diff
        );
    }

    template struct PSROIPoolingFunctor_BP_GPU<float>;
    template struct PSROIPoolingFunctor_BP_GPU<double>;

}


namespace tensorflow{
    template <typename T>
    __global__ void setZeroKernel(const int n, T* result_data)
    {

        CUDA_1D_KERNEL_LOOP(index, n) {
            *(result_data+index)=T(0);
        }

    }

    template <typename T>
    void setZero_GPU<T>::operator()(const GPUDevice &d, const int n, T *result_data) {
        CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
        setZeroKernel<T> <<< config.block_count, config.thread_per_block, 0, d.stream() >>>(n, result_data);

    };

    template struct setZero_GPU<float>;
    template struct setZero_GPU<int>;
    template struct setZero_GPU<double>;
}

#endif

