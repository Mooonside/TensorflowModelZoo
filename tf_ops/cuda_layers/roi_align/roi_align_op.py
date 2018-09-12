import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import os


_roi_align_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'roi_align.so'))
_roi_align = _roi_align_module.roi_align
_roi_align_bp = _roi_align_module.roi_align_bp
# print(_ps_roi_pooling_module, _ps_roi_pooling)

def roi_align(features, bboxes, spatial_scale, sample_ratio, pooled_height, pooled_width,  data_format='NHWC'):
    """
    Python  Wrapper for ROI Align operation
    :param features: [N, H, W, C]
    :param bboxes: [N, 5] [batch_id, xmin, ymin, xmax, ymax]
    :param spatial_scale: scale of bboxes
    :param sample_ratio: sampling rate of roi aligning
    :param pooled_height: roi bins
    :param pooled_width:
    :param data_format:
    :return:
    """
    if data_format == 'NHWC':
        features = tf.transpose(features, [0, 3, 1, 2])
    # [n, c, ph, pw]
    roi_aligning_ = _roi_align(
        bottom_data=features,
        bottom_rois=bboxes,
        spatial_scale=spatial_scale,
        sample_ratio=sample_ratio,
        pooled_height=pooled_height,
        pooled_width=pooled_width,
        data_format='NCHW')

    if data_format == 'NHWC':
        #
        roi_aligning_ = tf.transpose(roi_aligning_, [0, 2, 3, 1])

    return roi_aligning_


@ops.RegisterGradient("RoiAlign")
def _roi_align_grad(op, grad):
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
    sample_rate = op.get_attr('sample_ratio')
    pool_height = op.get_attr('pooled_height')
    pool_weight = op.get_attr('pooled_width')
    data_format = op.get_attr('data_format')

    # compute gradient
    data_grad = _roi_align_bp(grad, features, bboxes, spatial_scale, sample_rate, pool_height, pool_weight, data_format)

    return [data_grad, None]  # List of one Tensor, since we have one input


########################################################################################################################
# add unit test here ## add unit test here ## add unit test here ## add unit test here ## add unit test here ## add uni#
########################################################################################################################

if __name__ == '__main__':
    import numpy as np
    a = np.linspace(0, 5*5*9-1, num=5*5*9).reshape(9, 5, 5)
    a = np.transpose(a, [1, 2, 0])
    a = np.stack([a, 2*a, 3*a], axis=0).astype(np.float32)
    a = tf.get_variable(name='a', initializer=a, dtype=tf.float32)
    # a = np.expand_dims(a, axis=0).astype(np.float32)
    # a : [1, 5, 5, 9]
    bbox = np.asarray([0, 1, 1, 4.0, 4.0], dtype=np.float32).reshape([-1, 5])


    rois = roi_align(a, bbox, spatial_scale=1, sample_ratio=1, pooled_height=3, pooled_width=3, data_format='NHWC')
    # rois = ps_roi_pooling(a, bbox, spatial_scale=1, output_dim=1, group_size=3, data_format='NHWC')
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)


    loss = tf.reduce_sum(rois)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
    gradients = optimizer.compute_gradients(loss, var_list=[a])[0][0]
    print(gradients)

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        rois_v, gradients_v = sess.run([rois, gradients])
    print(rois_v.shape)
    print(rois_v[0, ..., 0])
    print(gradients_v[0, ..., 0])
    # print(gradients_v[0, ..., 1])
    # print(gradients_v[0, ..., 2])
