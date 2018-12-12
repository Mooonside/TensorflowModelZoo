import tensorflow as tf
from tensorflow.python.framework import ops
import os.path as osp
import numpy as np

_ps_roi_aligning_module = tf.load_op_library(osp.join(osp.dirname(__file__), 'ps_roi_aligning.so'))
_ps_roi_aligning = _ps_roi_aligning_module.ps_roi_aligning
_ps_roi_aligning_bp = _ps_roi_aligning_module.ps_roi_aligning_bp

@ops.RegisterGradient("PsRoiAligning")
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
    sample_ratio = op.get_attr('sample_ratio')
    group_size = op.get_attr('group_size')
    data_format = op.get_attr('data_format')
    # compute gradient
    data_grad = _ps_roi_aligning_bp(grad, features, bboxes, spatial_scale, sample_ratio, output_dim, group_size, data_format)

    return [data_grad, None]  # List of one Tensor, since we have one input


def ps_roi_aligning(features, bboxes, spatial_scale, sample_ratio, output_dim, group_size=3, data_format='NHWC'):
    """
    Python Wrapper for PS-ROI-ALIGNING LAYER
    :param features: [N, H, W, output_dim * group_size, group_size]
    :param bboxes: [M, 5], which batch id, ymin, xmin, ymax, xmax
    :param spatial_scale: how much bboxes should scale
    :param sample_ratio: sampling ratio used in roi align. sample_ratio^2 is #pts sample from one bin
    :param output_dim: (usually class numbers)
    :param group_size: (#bin row-wise)
    :param data_format: 'NHWC' as default
    :return:
    """
    if data_format == 'NHWC':
        data_format = 'NCHW'
        features = tf.transpose(features, [0, 3, 1, 2])

    ps_roi_aligning_ = _ps_roi_aligning_module.ps_roi_aligning(
        features,
        bboxes,
        spatial_scale,
        sample_ratio,
        output_dim,
        group_size, data_format
    )
    return ps_roi_aligning_


if __name__ == '__main__':
    import numpy as np
    # a = np.linspace(0, 6*6*9-1, num=6*6*9).reshape(9, 6, 6)
    # a = np.transpose(a, [1, 2, 0])
    # a = np.stack([a, 2*a, 3*a], axis=0).astype(np.float32)
    # a = tf.get_variable(name='a', initializer=a, dtype=tf.float32)
    # a = np.expand_dims(a, axis=0).astype(np.float32)
    # a : [1, 5, 5, 9]
    # bbox = np.asarray([[0, 1, 1, 4.0, 4.0]], dtype=np.float32).reshape([-1, 5])
    a = tf.get_variable(shape=(16, 32 ,32, 100 * 3 * 3), name='a', dtype=tf.float32)
    bboxes = tf.random_uniform(shape=(160, 4), minval=-2, maxval=35, dtype=tf.float32)
    batch_inds = tf.random_uniform(shape=(160, 1), minval=0, maxval=15, dtype=tf.int32)
    batch_inds = tf.cast(batch_inds, tf.float32)
    bboxes = tf.concat([batch_inds, bboxes], axis=-1)

    rois = ps_roi_aligning(a, bboxes, spatial_scale=1, sample_ratio=2, output_dim=100, group_size=3, data_format='NHWC')

    loss = tf.reduce_sum(rois)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
    gradients = optimizer.compute_gradients(loss, var_list=[a])[0][0]

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        rois_v, gradients_v = sess.run([rois, gradients])
    print(rois_v)

    print(gradients_v[0, ..., 0])
    print(gradients_v[0, ..., 1])
    print(gradients_v[0, ..., 2])
