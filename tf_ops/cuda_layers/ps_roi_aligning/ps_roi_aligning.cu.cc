#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "ps_roi_aligning.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"

using std::max;
using std::min;

//forward definition
namespace tensorflow{
    template <typename T>
    __device__ T bilinear_interpolate(const T* bottom_data, int height, int width, T y, T x){
        if (y < -1.0 || y > height || x < -1.0 || x > width) {
            return 0;
        }

        if (y <= 0) {
            y = 0;
        }
        if (x <= 0) {
            x = 0;
        }

        int y_low = (int)y;
        int x_low = (int)x;
        int y_high;
        int x_high;

        if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
        } else {
            y_high = y_low + 1;
        }

        if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
        } else {
            x_high = x_low + 1;
        }

        T ly = y - y_low;
        T lx = x - x_low;
        T hy = 1. - ly, hx = 1. - lx;

        T v1 = bottom_data[y_low * width + x_low];
        T v2 = bottom_data[y_low * width + x_high];
        T v3 = bottom_data[y_high * width + x_low];
        T v4 = bottom_data[y_high * width + x_high];
        // 对应下面公式hx:1-u hy:1-v lx:u ly:v
        T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
        return val;
    }


    template <typename T>
    __global__ void PSROIAligning_FW_CUDA(
            const int nthreads,
            const T* bottom_data,
            const T* bottom_rois,
            const float spatial_scale,
            const int sample_ratio, const int channels,
            const int height, const int width,
            const int pooled_height, const int pooled_width,
            const int output_dim, const int group_size,
            T* top_data)
    {
        for(int index : CudaGridRangeX(nthreads)) {
            // The output is in order (n, ctop, ph, pw)
            int pw = index % pooled_width;
            int ph = (index / pooled_width) % pooled_height;
            int ctop = (index / pooled_width / pooled_height) % output_dim;
            int n = index / pooled_width / pooled_height / output_dim;

            bottom_rois += n * 5;
            int roi_batch_ind = bottom_rois[0];
            T roi_start_w = static_cast<T>(bottom_rois[1]) * spatial_scale;
            T roi_start_h = static_cast<T>(bottom_rois[2]) * spatial_scale;
            // takes value from [roi_start_w, roi_end_w)
            T roi_end_w = static_cast<T>(bottom_rois[3] + 1.) * spatial_scale;
            T roi_end_h = static_cast<T>(bottom_rois[4] + 1.) * spatial_scale;

            T roi_width = max(roi_end_w - roi_start_w, (T)1.); //avoid 0
            T roi_height = max(roi_end_h - roi_start_h, (T)1.);

            // Compute w and h at bottom
            T bin_size_h = roi_height / static_cast<T>(pooled_height);
            T bin_size_w = roi_width / static_cast<T>(pooled_width);

            //get the position of the corresponding bin
            T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
            T wstart = static_cast<T>(pw)* bin_size_w + roi_start_w;
            T hend = static_cast<T>(ph + 1) * bin_size_h + roi_start_h;
            T wend = static_cast<T>(pw + 1) * bin_size_w + roi_start_w;

            //in case out of bound
            hstart = min(max(hstart, static_cast<T>(0)), static_cast<T>(height));
            hend = min(max(hend, static_cast<T>(0)), static_cast<T>(height));
            wstart = min(max(wstart, static_cast<T>(0)), static_cast<T>(width));
            wend = min(max(wend, static_cast<T>(0)), static_cast<T>(width));
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // take value from which channel
            int c = (ctop*group_size + ph)*group_size + pw;
            const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

            // do roi aligning
            int roi_bin_grid_h = (sample_ratio > 0) ? sample_ratio : static_cast<int>(ceil(roi_height / pooled_height));
            int roi_bin_grid_w = (sample_ratio > 0) ? sample_ratio : static_cast<int>(ceil(roi_width / pooled_width));

            const T count = roi_bin_grid_h * roi_bin_grid_w;
            T output_val = 0.;
            for (int iy = 0; iy < roi_bin_grid_h; iy++){
                const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
                    T val = bilinear_interpolate(offset_bottom_data, height, width, y, x);
                    output_val += val;
                }
            }
            output_val /= count;
            top_data[index] = is_empty ? static_cast<T>(0):output_val;

        }
    };




    template <typename T>
    void Functor_PSROIAligning_FW_GPU<T>::operator()(
            const GPUDevice& device,
            const T* bottom_data,
            const T* bottom_rois,
            const float spatial_scale, const int sample_ratio,
            const int roi_nums, const int channels,
            const int height, const int width,
            const int pooled_height, const int pooled_width,
            const int output_dim, const int group_size,
            T* top_data)
    {
        int outputs_size = roi_nums * output_dim * pooled_height * pooled_width;
        CudaLaunchConfig config = GetCudaLaunchConfig(outputs_size, device);
        PSROIAligning_FW_CUDA<T><<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
                outputs_size,
                bottom_data,
                bottom_rois,
                spatial_scale,
                sample_ratio, channels,
                height, width,
                pooled_height, pooled_width,
                output_dim, group_size,
                top_data
        );
    }

    template struct Functor_PSROIAligning_FW_GPU<float>;
    template struct Functor_PSROIAligning_FW_GPU<double>;

}


//backward definition
namespace tensorflow{
    template <typename T>
    __device__ void bilinear_interpolate_gradient(
            const int height,
            const int width,
            T y,
            T x,
            T& w1,
            T& w2,
            T& w3,
            T& w4,
            int& x_low,
            int& x_high,
            int& y_low,
            int& y_high) {
        // deal with cases that inverse elements are out of feature map boundary
        if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            w1 = w2 = w3 = w4 = 0.;
            x_low = x_high = y_low = y_high = -1;
            return;
        }

        if (y <= 0) {
            y = 0;
        }
        if (x <= 0) {
            x = 0;
        }

        y_low = (int)y;
        x_low = (int)x;

        if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
        } else {
            y_high = y_low + 1;
        }

        if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
        } else {
            x_high = x_low + 1;
        }

        T ly = y - y_low;
        T lx = x - x_low;
        T hy = 1. - ly, hx = 1. - lx;

        // reference in forward
        // T v1 = bottom_data[y_low * width + x_low];
        // T v2 = bottom_data[y_low * width + x_high];
        // T v3 = bottom_data[y_high * width + x_low];
        // T v4 = bottom_data[y_high * width + x_high];
        // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

        w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        return;
    }


    template <typename T>
    __global__ void PSROIAligning_BW_CUDA(
            const int nthreads,
            const T* top_diff,
            const T* bottom_rois,
            const float spatial_scale,
            const int sample_ratio, const int channels,
            const int height, const int width,
            const int pooled_height, const int pooled_width,
            const int output_dim, T* bottom_diff) {
        for(int index : CudaGridRangeX(nthreads)) {
            // (n, output_dim, ph, pw) is an element in the pooled output
            int pw = index % pooled_width;
            int ph = (index / pooled_width) % pooled_height;
            int ctop = (index / pooled_width / pooled_height) % output_dim;
            int n = index / pooled_width / pooled_height / output_dim;

            const T* offset_bottom_rois = bottom_rois + n * 5;
            int roi_batch_ind = offset_bottom_rois[0];

            T roi_start_w = static_cast<T>(offset_bottom_rois[1]) * spatial_scale;
            T roi_start_h = static_cast<T>(offset_bottom_rois[2]) * spatial_scale;
            T roi_end_w = static_cast<T>(offset_bottom_rois[3] + 1.) * spatial_scale;
            T roi_end_h = static_cast<T>(offset_bottom_rois[4] + 1.) * spatial_scale;


            // Force malformed ROIs to be 1x1
            T roi_width = max(roi_end_w - roi_start_w, (T)1.);
            T roi_height = max(roi_end_h - roi_start_h, (T)1.);

            T bin_size_h = roi_height / static_cast<T>(pooled_height);
            T bin_size_w =roi_width / static_cast<T>(pooled_width);

            //get the position of the corresponding bin
            T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
            T wstart = static_cast<T>(pw)* bin_size_w + roi_start_w;
            T hend = static_cast<T>(ph + 1) * bin_size_h + roi_start_h;
            T wend = static_cast<T>(pw + 1) * bin_size_w + roi_start_w;

            //in case out of bound
            hstart = min(max(hstart, static_cast<T>(0)), static_cast<T>(height));
            hend = min(max(hend, static_cast<T>(0)), static_cast<T>(height));
            wstart = min(max(wstart, static_cast<T>(0)), static_cast<T>(width));
            wend = min(max(wend, static_cast<T>(0)), static_cast<T>(width));
            bool is_empty = (hend <= hstart) || (wend <= wstart);


            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_h = (sample_ratio > 0) ? sample_ratio : static_cast<int>(ceil(roi_height / pooled_height));
            int roi_bin_grid_w = (sample_ratio > 0) ? sample_ratio : static_cast<int>(ceil(roi_width / pooled_width));


            int c = (ctop * pooled_height + ph) * pooled_width + pw;
            T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

            // We do average (integral) pooling inside a bin
            const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

            for (int iy = 0; iy < roi_bin_grid_h; iy++){
                const T y = hstart + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    const T x = wstart + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

                    //calculate gradients for these 4 corner points
                    int x_low, x_high, y_low, y_high;
                    T w1, w2, w3, w4;
                    bilinear_interpolate_gradient(
                            height, width,
                            y, x,
                            w1, w2, w3, w4,
                            x_low, x_high, y_low, y_high);

                    T g1 = is_empty ? static_cast<T>(0) : top_diff[index] * w1 / count;
                    T g2 = is_empty ? static_cast<T>(0) : top_diff[index] * w2 / count;
                    T g3 = is_empty ? static_cast<T>(0) : top_diff[index] * w3 / count;
                    T g4 = is_empty ? static_cast<T>(0) : top_diff[index] * w4 / count;

                    if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                        CudaAtomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
                        CudaAtomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
                        CudaAtomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
                        CudaAtomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
                    } // if
                } // ix
            } // iy
        } // CUDA_1D_KERNEL_LOOP
    } // RoIAlignBackward


    template <typename T>
    void Functor_PSROIAligning_BP_GPU<T>::operator()(
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
            const int output_dim,
            T *bottom_diff) {

        int outputs_size = roi_nums * output_dim * pooled_height * pooled_width;
        CudaLaunchConfig config = GetCudaLaunchConfig(outputs_size, device);

        PSROIAligning_BW_CUDA<<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
                outputs_size,
                top_diff,
                bottom_rois,
                spatial_scale,
                sample_ratio,
                channels,
                height, width,
                pooled_height, pooled_width,
                output_dim,
                bottom_diff
        );
    }

    template struct Functor_PSROIAligning_BP_GPU<float>;
    template struct Functor_PSROIAligning_BP_GPU<double>;
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
