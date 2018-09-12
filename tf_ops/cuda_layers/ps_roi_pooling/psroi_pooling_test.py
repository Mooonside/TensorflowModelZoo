import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

_ps_roi_pooling_module = tf.load_op_library('ps_roi_pooling.so')
_ps_roi_pooling = _ps_roi_pooling_module.ps_roi_pooling
_ps_roi_pooling_bp = _ps_roi_pooling_module.ps_roi_pooling_bp
# print(_ps_roi_pooling_module, _ps_roi_pooling)

def ps_roi_pooling(features, bboxes, spatial_scale, output_dim, group_size=3, data_format='NHWC'):
    """
    Python Wrapper for PS-ROI-POOLING LAYER
    :param features: [N, H, W, output_dim * group_size, group_size]
    :param bboxes: [M, 5]
    :param spatial_scale: how much bboxes should scale
    :param output_dim: (usually class numbers)
    :param group_size: (#bin row-wise)
    :param data_format: 'NHWC' as default
    :return:
    """
    if data_format == 'NHWC':
        data_format = 'NCHW'
        features = tf.transpose(features, [0, 3, 1, 2])

    ps_roi_pooling_ = _ps_roi_pooling(features, bboxes, spatial_scale, output_dim, group_size, data_format)
    return ps_roi_pooling_


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
    bbox = np.asarray([[0, 1, 1, 4.0, 4.0]], dtype=np.float32).reshape([-1, 5])


    rois = ps_roi_pooling(a, bbox, spatial_scale=1, output_dim=1, group_size=3, data_format='NHWC')
    # rois = ps_roi_pooling(a, bbox, spatial_scale=1, output_dim=1, group_size=3, data_format='NHWC')

    loss = tf.reduce_sum(rois)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
    gradients = optimizer.compute_gradients(loss, var_list=[a])[0][0]
    print(gradients)
    # exit()

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        rois_v, gradients_v = sess.run([rois, gradients])
    print(rois_v)
    print(gradients_v[0, ..., 0])
    print(gradients_v[0, ..., 1])
    print(gradients_v[0, ..., 2])
