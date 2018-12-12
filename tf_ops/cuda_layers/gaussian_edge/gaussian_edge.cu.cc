#ifndef GAUSSIAN_EDGE_GPU_H
#define GAUSSIAN_EDGE_GPU_H

#define EIGEN_USE_GPU

#include "gaussian_edge.h"
#include "cuda.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include <algorithm>
#include <cstring>
#include <vector>
using std::max;
using std::min;


namespace tensorflow {
    template <typename DType>
    __global__ void setZeroKernel(const int n, DType* result_data) {
        CUDA_1D_KERNEL_LOOP(index, n) {
            *(result_data+index)=DType(0);
        }
    }

    template <typename DType>
    __device__ DType get_gaussian(const int y, const int x,
                                  const int h, const int w,
                                  const DType tsys, const DType tsxs//twice the square of sigma,

    ){
        auto dy = static_cast<DType>(y - h);
        auto dx = static_cast<DType>(x - w);
        dx = dx * dx / (tsys);
        dy = dy * dy / (tsxs);
        dx = exp(-(dx + dy));
        return dx;
    };


    __device__ bool is_edge(const int32* data_im,
                            const int nearest,
                            const int y,
                            const int x,
                            const int height,
                            const int width){
        int pos = y * width + x;
        int32 cval = data_im[pos];

        int y_start = max(y - nearest, 0);
        int y_end = min(y + nearest, height);
        int x_start = max(x - nearest, 0);
        int x_end = min(x + nearest, width);

        for (int i=y_start; i<y_end; i++){
            for (int j=x_start; j<x_end; j++){
                int new_pos = i * width + j;
                int32 nval = data_im[new_pos];
                if(cval != nval)
                    return true;
            }
        }
        return false;
    }


    template <typename DType>
    __global__ void gaussian_edge_gpu_kernel(const int nthreads,
                                             const int* data_im,
                                             const int nearest,
                                             const int height, const int width,
                                             const int kh, const int kw,
                                             const DType tsys, const DType tsxs,
                                             DType* data_out) {
        for(int index : CudaGridRangeX(nthreads)) {
//            // index index of output matrix [H, W]
            int h = index / width;
            int w = index % width;

            // no edge found Orz
            if(is_edge(data_im, nearest, h, w, height, width)) continue;

            int h_kernel_start = h - static_cast<int>(kh / 2.0);
            int h_kernel_end = h_kernel_start + kh;
            h_kernel_start = max(h_kernel_start, 0);
            h_kernel_end = min(h_kernel_end, height);

            int w_kernel_start = w - static_cast<int>(kw / 2.0);
            int w_kernel_end = w_kernel_start + kw;
            w_kernel_start = max(w_kernel_start, 0);
            w_kernel_end = min(w_kernel_end, width);

            if (index % 2 == 0) {
                // add from up to down
                for(int y = h_kernel_start; y < h_kernel_end;++y){
                    for(int x = w_kernel_start; x < w_kernel_end; ++x){
                        int cur = y * width + x;
                        DType val = get_gaussian(
                                y, x,
                                h, w,
                                tsys, tsxs
                        );
                        CudaAtomicMax(data_out + cur, val);
                    }
                }
            }
            else{
                // add from down to up
                for(int y = h_kernel_end-1; y >= h_kernel_start;--y){
                    for(int x = w_kernel_end-1; x >= w_kernel_start; --x){
                        int cur = y * width + x;
                        DType val = get_gaussian<DType>(
                                y, x,
                                h, w,
                                tsys, tsxs
                        );
                        CudaAtomicMax(data_out + cur, val);
                    }
                }
            }
        }
    }


    namespace functor {
        template<typename DType>
        struct setZero<GPUDevice, DType> {
            void operator()(const GPUDevice &d, const int n, DType *result_data) {
                CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
                setZeroKernel<DType> << < config.block_count, config.thread_per_block, 0, d.stream() >> >
                                                                                          (n, result_data);
            }
        };

        template <typename DType>
        struct gaussian_edge<GPUDevice, DType>{
            void operator()(
                    const GPUDevice& d,
                    const int32* data_im,
                    const IntVec& im_shape,
                    const int nearest,
                    const int kh, const int kw,
                    const DType tsys, const DType tsxs,
                    DType* data_out
            ){
                int num_kernels = ProdShape(im_shape, 1);
                CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
                gaussian_edge_gpu_kernel<DType><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                    num_kernels, data_im,
                    nearest,
                    im_shape[1], im_shape[2],
                    kh, kw,
                    tsys, tsxs,
                    data_out
                );
            };
        };
    }


    #define DECLARE_GPU_SPEC(DType)  \
        template struct functor::setZero<GPUDevice, DType>; \
        template struct functor::gaussian_edge<GPUDevice, DType>;

    // extern template struct Copy<GPUDevice, T>;
    TF_CALL_float(DECLARE_GPU_SPEC);
    TF_CALL_double(DECLARE_GPU_SPEC);
    #undef DECLARE_GPU_SPEC

} //END of namespace tensorflow


#endif
