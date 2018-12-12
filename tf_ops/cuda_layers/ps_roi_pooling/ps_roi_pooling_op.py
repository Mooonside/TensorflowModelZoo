import tensorflow as tf
from tensorflow.python.framework import ops
import os.path as osp


_ps_roi_pooling_module = tf.load_op_library(osp.join(osp.dirname(__file__), 'ps_roi_pooling.so'))
_ps_roi_pooling = _ps_roi_pooling_module.ps_roi_pooling
_ps_roi_pooling_bp = _ps_roi_pooling_module.ps_roi_pooling_bp


@ops.RegisterGradient("PsRoiPooling")
def _ps_roi_pooling_grad(op, grad):
    """The gradients for `deform_conv`.
    Args:
      op: The `deform_conv` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `roi_pool` op.
    Returns:
      Gradients with respect to the input of `zero_out`.
    """
    features = op.inputs[0]
    bboxes = op.inputs[1]

    spatial_scale = op.get_attr('spatial_scale')
    output_dim = op.get_attr('output_dim')
    group_size = op.get_attr('group_size')
    data_format = op.get_attr('data_format')
    print(features, bboxes, spatial_scale, output_dim, group_size, data_format)
    # compute gradient
    data_grad = _ps_roi_pooling_bp(grad, features, bboxes, spatial_scale, output_dim, group_size, data_format)

    return [data_grad, None]  # List of one Tensor, since we have one input
