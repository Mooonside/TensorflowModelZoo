

#ifndef GAUSSIAN_EDGE_H
#define GAUSSIAN_EDGE_H

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include <cstring>
#include <vector>


namespace tensorflow {
    using CPUDevice = Eigen::ThreadPoolDevice;
    using GPUDevice = Eigen::GpuDevice;
    typedef std::vector<int32> TShape;
    typedef std::vector<int32> IntVec;
    typedef std::vector<float> FloatVec;

    inline int ProdTensorShape(const TensorShape &shape, int start) {
        int res = 1;
        for(int i=start; i<shape.dims(); i++) {
            res*=shape.dim_size(i);
        }
        return res;
    }

    inline int ProdShape(const IntVec &shape, int start) {
        int res = 1;
        for(int i=start; i<shape.size(); i++) {
            res*=shape[i];
        }
        return res;
    }


//#if GOOGLE_CUDA
    namespace functor {
        template <typename Device, typename DType>
        struct setZero {
            void operator() (const Device& d, const int n, DType* result_data);
        };


        template <typename Device, typename DType>
        struct gaussian_edge{
            void operator()(
                    const Device& d,
                    const int32* data_im,
                    const IntVec& im_shape,
                    const int nearest,
                    const int kh, const int kw,
                    const DType tsys, const DType tsxs,
                    DType* data_out
            );
        };
    }
//#endif

}


#endif //GAUSSIAN_EDGE_H