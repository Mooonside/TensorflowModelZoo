#define EIGEN_USE_THREADS

#include <cfloat>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// #include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/kernels/bounds_check.h"

#include "tensorflow/core/platform/stream_executor.h"
#include "gaussian_edge.h"

using std::max;
using std::min;

namespace tensorflow {
    using shape_inference::DimensionHandle;
    using shape_inference::InferenceContext;
    using shape_inference::ShapeHandle;


    REGISTER_OP("GaussianEdgeOp").Input("x: int32")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .Attr("kernel: list(int)")
    .Attr("sigma: list(float)")
    .Attr("nearest: int = 3")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));
        c->set_output(0, input_shape);

        IntVec kernel;
        TF_RETURN_IF_ERROR(c->GetAttr("kernel", &kernel));
        if (kernel.size() != 2) {
            return errors::InvalidArgument(
                    "Requires kernel to be of length 2, i.e. [kh, kw]: ",
                    kernel.size());
        }

        FloatVec sigma;
        TF_RETURN_IF_ERROR(c->GetAttr("sigma", &sigma));
        if (sigma.size() != 2) {
            return errors::InvalidArgument(
                    "Requires sigma to be of length 2, i.e. [sy, sx]: ",
                    sigma.size());
        }

        int nearest;
        TF_RETURN_IF_ERROR(c->GetAttr("nearest", &nearest));
        if (nearest <= 0) {
            return errors::InvalidArgument(
                    "Requires nearest >= 1 ",
                    sigma.size());
        }

        return Status::OK();
    })
    .Doc(R"doc(only support NCHW now)doc");


    /*implement CPU kernel here
    namespace functor{
        template <typename DType>
        struct gaussian_edge<CPUDevice, DType> {
            void operator()(const CPUDevice& d,
                            const int32* data_im,
                            const IntVec& im_shape,
                            const int kh, const int kw,
                            const DType tsys, const DType tsxs,
                            DType* data_out) {
                int num_kernels = ProdShape(im_shape, 1);
                int height = im_shape[1];
                int width = im_shape[2];

                for (int index=0; index<num_kernels; ++index){
                    if(data_im[index] <= 0) continue;
                    int h = index / width;
                    int w = index % width;

                    int h_kernel_start = h - static_cast<int>(kh / 2.0);
                    int h_kernel_end = h_kernel_start + kh;
                    h_kernel_start = max(h_kernel_start, 0);
                    h_kernel_end = min(h_kernel_end, height);

                    int w_kernel_start = w - static_cast<int>(kw / 2.0);
                    int w_kernel_end = w_kernel_start + kw;
                    w_kernel_start = max(w_kernel_start, 0);
                    w_kernel_end = min(w_kernel_end, width);

                    for(int y = h_kernel_start; y < h_kernel_end;++y){
                        for(int x = w_kernel_start; x < w_kernel_end; ++x){
                            int cur = y * width + x;
                            auto dy = static_cast<DType>(y - h);
                            auto dx = static_cast<DType>(x - w);
                            dx = dx * dx / (tsys);
                            dy = dy * dy / (tsxs);
                            auto val = static_cast<DType>(exp(-(dx + dy)));
                            data_out[cur] = max(data_out[cur], val);
                        }
                    }
                }
            }
        };

        template <typename DType>
        struct setZero<CPUDevice, DType> {
            void operator()(const CPUDevice &d, const int n, DType *result_data) {
                for (int i = 0; i < n; ++i) {
                    result_data[i] = static_cast<DType>(0);
                }
            }
        };

    }
    */

    template<typename Device, typename T>
    class GaussianEdgeOp : public OpKernel {
    public:
        explicit GaussianEdgeOp(OpKernelConstruction *context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("kernel", &kernel_));
            OP_REQUIRES_OK(context, context->GetAttr("sigma", &sigma_));
            OP_REQUIRES_OK(context, context->GetAttr("nearest", &nearest_));
        }

        void Compute(OpKernelContext *context) override {
            const Tensor &input = context->input(0);
            const TensorShape &ishape = input.shape();


            OP_REQUIRES(context, input.dims() == 3,
                        errors::InvalidArgument("input must be 3-dimensional", input.shape().DebugString()));

            int64 N_ = ishape.dim_size(0);
            int64 H_ = ishape.dim_size(1);
            int64 W_ = ishape.dim_size(2);

            OP_REQUIRES(context, FastBoundsCheck(H_,
                                                 std::numeric_limits<int>::max()),
                        errors::InvalidArgument("Input batch too large"));
            OP_REQUIRES(context, FastBoundsCheck(H_,
                                                 std::numeric_limits<int>::max()),
                        errors::InvalidArgument("Input rows too large"));
            OP_REQUIRES(context, FastBoundsCheck(W_,
                                                 std::numeric_limits<int>::max()),
                        errors::InvalidArgument("Input cols too large"));

            N = static_cast<int>(N_);
            H = static_cast<int>(H_);
            W = static_cast<int>(W_);
            VLOG(2) << N << H << W;


            kh = kernel_[0]; kw = kernel_[1];
            sy = sigma_[0]; sx = sigma_[1];

            OP_REQUIRES(context, sy > 0, errors::InvalidArgument("sy must > 0!"));
            OP_REQUIRES(context, sx > 0, errors::InvalidArgument("sx must > 0!"));
            OP_REQUIRES(context, kh > 0, errors::InvalidArgument("kh must > 0!"));
            OP_REQUIRES(context, kw > 0, errors::InvalidArgument("kw must > 0!"));


            tsys = static_cast<T>(2.0 * sy * sy);
            tsxs = static_cast<T>(2.0 * sx * sx);

            Tensor *output_3d = nullptr;
            TensorShape out_shape = ishape;

            OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_3d));
            T *output_3d_ptr = output_3d->template flat<T>().data();

            auto in_data_ptr = input.template flat<int32>().data();

            int num_kernels = ProdTensorShape(ishape, 0);

            functor::setZero<Device, T>()(
                    context->eigen_device<Device>(),
                    num_kernels,
                    output_3d_ptr
            );

            IntVec vec_im_shape = IntVec({N, H, W});
            input_slice = ProdTensorShape(ishape, 1);

            for (int n=0; n < N; ++n) {
                functor::gaussian_edge<Device, T>()(
                    context->eigen_device<Device>(),
                    in_data_ptr + n * input_slice,
                    vec_im_shape,
                    nearest_,
                    kh, kw,
                    tsys, tsxs,
                    output_3d_ptr + n * input_slice
                );
            }
            if (out_shape.num_elements() == 0)
                return;

        }
    private:
        IntVec kernel_;
        FloatVec sigma_;

        int N; int H; int W;

        int kh; int kw;
        int input_slice;
        int nearest_;

        float sy; float sx;
        T tsys; T tsxs;
    };


////REGISTRER CPU KERNEL
//#define REGISTER(T)                                                 \
//REGISTER_KERNEL_BUILDER(                                          \
//    Name("GaussianEdgeOp").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
//    GaussianEdgeOp<CPUDevice, T>);
//TF_CALL_float(REGISTER);
//TF_CALL_double(REGISTER);
//#undef REGISTER


#if GOOGLE_CUDA
#define REGISTER(T)                                                 \
REGISTER_KERNEL_BUILDER(                                          \
    Name("GaussianEdgeOp").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
    GaussianEdgeOp<GPUDevice, T>);
// TF_CALL_GPU_NUMBER_TYPES(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA
}