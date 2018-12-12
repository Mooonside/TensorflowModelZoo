#include "roi_align.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"


// define forward
namespace tensorflow {
    using shape_inference::DimensionHandle;
    using shape_inference::InferenceContext;
    using shape_inference::ShapeHandle;


    REGISTER_OP("RoiAlign")
            .Input("bottom_data: T")
            .Input("bottom_rois: T")
            .Attr("spatial_scale: float")
            .Attr("sample_ratio: int")
            .Attr("pooled_height: int")
            .Attr("pooled_width: int")
            .Attr("data_format: string")
            .Attr("T: {float, double}")
            .Output("output: T")
            .SetShapeFn([](InferenceContext *c) {
                ShapeHandle bottom_data_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &bottom_data_shape));
                ShapeHandle bottom_rois_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &bottom_rois_shape));

                string data_format;
                TensorFormat data_format_;
                TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format));

                int pooled_height;
                TF_RETURN_IF_ERROR(c->GetAttr("pooled_height", &pooled_height));
                int pooled_width;
                TF_RETURN_IF_ERROR(c->GetAttr("pooled_width", &pooled_width));


                DimensionHandle in_channels = c->Dim(bottom_data_shape, 1);
                DimensionHandle roi_nums = c->Dim(bottom_rois_shape, 0);


                ShapeHandle output_shape = c->MakeShape(
                        {roi_nums, in_channels, pooled_height, pooled_width});
                c->set_output(0, output_shape);

                return Status::OK();
            })
            .Doc(R"doc(xxx)doc");


    extern template struct Functor_ROIALIGN_FW_GPU<float>;
    extern template struct Functor_ROIALIGN_FW_GPU<double>;
    template <typename T>
    class ROI_ALIGN_FW_GPU : public OpKernel {
    public:
        explicit ROI_ALIGN_FW_GPU(OpKernelConstruction* context) : OpKernel(context) {
            // get parameters from outside
            OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale_));
            OP_REQUIRES_OK(context, context->GetAttr("sample_ratio", &sample_ratio_));
            OP_REQUIRES_OK(context, context->GetAttr("pooled_height", &pooled_height_));
            OP_REQUIRES_OK(context, context->GetAttr("pooled_width", &pooled_width_));

            OP_REQUIRES(context, pooled_height_ > 0, errors::InvalidArgument( "pooled_height must be > 0"));
            OP_REQUIRES(context, pooled_width_ > 0, errors::InvalidArgument( "pooled_width must be > 0"));

            string data_format;
            OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
            OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                        errors::InvalidArgument("Invalid data format"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& bottom_data = context->input(0);
            const TensorShape& bottom_data_shape = bottom_data.shape();
            const Tensor& bottom_rois = context->input(1);
            const TensorShape& bottom_rois_shape = bottom_rois.shape();

            batch_size_ =  static_cast<int>(GetTensorDim(bottom_data, data_format_, 'N'));
            height_ =  static_cast<int>(GetTensorDim(bottom_data, data_format_, 'H'));
            width_ = static_cast<int>(GetTensorDim(bottom_data, data_format_, 'W'));
            channels_ = static_cast<int>(GetTensorDim(bottom_data, data_format_, 'C'));

            roi_nums_ = static_cast<int>(bottom_rois.dim_size(0));
            //allocate spaces for top_data
            Tensor* top_data;
            TensorShape top_shape = ShapeFromFormat(data_format_, roi_nums_, pooled_height_, pooled_width_, channels_);
            OP_REQUIRES_OK(context, context->allocate_output(0, top_shape, &top_data));


            auto top_data_ptr = top_data->template flat<T>().data();
            auto bottom_data_ptr = bottom_data.template flat<T>().data();
            auto bottom_rois_ptr = bottom_rois.template flat<T>().data();


            Functor_ROIALIGN_FW_GPU<T>()(
                    context->eigen_device<GPUDevice>(),
                    bottom_data_ptr,
                    bottom_rois_ptr,
                    spatial_scale_, sample_ratio_,
                    roi_nums_, channels_,
                    height_, width_,
                    pooled_height_, pooled_width_,
                    top_data_ptr
            );
        }

    private:

        float spatial_scale_;
        int channels_;
        int batch_size_;
        int sample_ratio_;
        int roi_nums_;
        int height_;
        int width_;
        int pooled_height_;
        int pooled_width_;
        TensorFormat data_format_;
    };

#ifdef GOOGLE_CUDA
#define REGISTER_FW_GPU(T)                                          \
              /* Declare explicit instantiations in kernel_example.cu.cc. */ \
              REGISTER_KERNEL_BUILDER(                                       \
                  Name("RoiAlign").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
                  ROI_ALIGN_FW_GPU<T>);
REGISTER_FW_GPU(float);
REGISTER_FW_GPU(double);
#endif  // GOOGLE_CUDA
}


//define bp
namespace tensorflow {
    using shape_inference::DimensionHandle;
    using shape_inference::InferenceContext;
    using shape_inference::ShapeHandle;

    REGISTER_OP("RoiAlignBp")
            .Input("top_diff: T")
            .Input("bottom_data: T")
            .Input("bottom_rois: T")
            .Attr("spatial_scale: float")
            .Attr("sample_ratio: int")
            .Attr("pooled_height: int")
            .Attr("pooled_width: int")
            .Attr("data_format: string")
            .Attr("T: {float, double}")
            .Output("output: T")
            .SetShapeFn([](InferenceContext *c) {
                // do dimension checking here
                return Status::OK();
            })
            .Doc(R"doc(xxx)doc");

    extern template struct Functor_ROIALIGN_BP_GPU<float>;
    extern template struct Functor_ROIALIGN_BP_GPU<double>;
    extern template struct setZero_GPU<float>;
    extern template struct setZero_GPU<double>;

    template <typename T>
    class ROI_ALIGN_BP_GPU : public OpKernel {
    public:
        explicit ROI_ALIGN_BP_GPU(OpKernelConstruction* context) : OpKernel(context) {
            // get parameters from outside
            OP_REQUIRES_OK(context, context->GetAttr("spatial_scale", &spatial_scale_));
            OP_REQUIRES_OK(context, context->GetAttr("sample_ratio", &sample_ratio_));
            OP_REQUIRES_OK(context, context->GetAttr("pooled_height", &pooled_height_));
            OP_REQUIRES_OK(context, context->GetAttr("pooled_width", &pooled_width_));

            OP_REQUIRES(context, pooled_height_ > 0, errors::InvalidArgument( "pooled_height must be > 0"));
            OP_REQUIRES(context, pooled_width_ > 0, errors::InvalidArgument( "pooled_width must be > 0"));


            string data_format;
            OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
            OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                        errors::InvalidArgument("Invalid data format"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& top_diff = context->input(0);
            const TensorShape& top_diff_shape = top_diff.shape();
            const Tensor& bottom_data = context->input(1);
            const TensorShape& bottom_data_shape = bottom_data.shape();
            const Tensor& bottom_rois = context->input(2);
            const TensorShape& bottom_rois_shape = bottom_rois.shape();

            batch_size_ =  static_cast<int>(GetTensorDim(bottom_data, data_format_, 'N'));
            height_ =  static_cast<int>(GetTensorDim(bottom_data, data_format_, 'H'));
            width_ = static_cast<int>(GetTensorDim(bottom_data, data_format_, 'W'));
            channels_ = static_cast<int>(GetTensorDim(bottom_data, data_format_, 'C'));


            roi_nums_ = static_cast<int>(bottom_rois.dim_size(0));
            //allocate spaces for top_data
            Tensor* bottom_diff;
            TensorShape bottom_diff_shape = ShapeFromFormat(data_format_, batch_size_, height_, width_, channels_);
            OP_REQUIRES_OK(context, context->allocate_output(0, bottom_diff_shape, &bottom_diff));

            auto bottom_diff_ptr = bottom_diff->template flat<T>().data();
            auto top_diff_ptr = top_diff.template flat<T>().data();
            auto bottom_data_ptr = bottom_data.template flat<T>().data();
            auto bottom_rois_ptr = bottom_rois.template flat<T>().data();

            setZero_GPU<T>()(context->eigen_device<GPUDevice>(),
                             batch_size_* height_* width_* channels_,
                             bottom_diff_ptr);

            Functor_ROIALIGN_BP_GPU<T>()(
                    context->eigen_device<GPUDevice>(),
                    top_diff_ptr,
                    bottom_rois_ptr,
                    spatial_scale_,
                    sample_ratio_,
                    roi_nums_,
                    channels_,
                    height_,
                    width_,
                    pooled_height_,
                    pooled_width_,
                    bottom_diff_ptr
            );
        }

    private:

        float spatial_scale_;
        int channels_;
        int batch_size_;
        int sample_ratio_;
        int roi_nums_;
        int height_;
        int width_;
        int pooled_height_;
        int pooled_width_;
        TensorFormat data_format_;
    };

    #ifdef GOOGLE_CUDA
    #define REGISTER_BP_GPU(T)                                          \
                  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
                  REGISTER_KERNEL_BUILDER(                                       \
                      Name("RoiAlignBp").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
                      ROI_ALIGN_BP_GPU<T>);
        REGISTER_BP_GPU(float);
        REGISTER_BP_GPU(double);
    #endif  // GOOGLE_CUDA
}