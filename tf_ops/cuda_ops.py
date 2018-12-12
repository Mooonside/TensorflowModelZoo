import tensorflow as tf

from tf_ops.cuda_layers.deformable_conv2d import deform_conv_op
from tf_ops.cuda_layers.ps_roi_pooling import ps_roi_pooling_op

from tf_ops.wrap_ops import same_padding, tensor_shape, conv2d, batch_norm, group_norm2d

add_arg_scope = tf.contrib.framework.add_arg_scope

TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
GLOBAL_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

var_scope = tf.variable_scope
arg_scope = tf.contrib.framework.arg_scope

weight_collections = 'weights_collections'
bias_collections = 'bias_collections'
batch_norm_collections = 'batch_norm_collections'

WEIGHT_COLLECTIONS = [weight_collections, TRAINABLE_VARIABLES, GLOBAL_VARIABLES]
BIAS_COLLECTIONS = [bias_collections, TRAINABLE_VARIABLES, GLOBAL_VARIABLES]
BN_COLLECTIONS = [batch_norm_collections, TRAINABLE_VARIABLES, GLOBAL_VARIABLES]
LOSS_COLLECTIONS = tf.GraphKeys.LOSSES

@add_arg_scope
def deform_conv2d(inputs,
                  num_outputs,
                  kernel_size,
                  stride=1,
                  rate=1,
                  padding='SAME',
                  activation_fn=tf.nn.relu,
                  deformable_group=1,
                  num_groups=1,
                  normalizer_fn=None,
                  weights_initializer=None,
                  weights_regularizer=None,
                  biases_initializer=tf.zeros_initializer,
                  biases_regularizer=None,
                  outputs_collections=None,
                  offsets_collections='offsets',
                  scope=None):
    assert num_outputs % num_groups == 0, print('outc % num_groups != 0')
    kernel_size = [kernel_size, kernel_size] if type(kernel_size) is int else kernel_size
    stride = [stride, stride] if type(stride) is int else stride
    rate = [rate, rate] if type(rate) is int else rate

    with tf.variable_scope(scope, 'deform_conv2d'):
        _, iH, iW, indim = tensor_shape(inputs)
        assert indim % num_groups == 0, print('indim % num_groups != 0')
        assert indim % deformable_group == 0, print('indim % deformable_group != 0')

        offsets = conv2d(
            inputs,
            num_outputs= kernel_size[0] * kernel_size[1] * 2 * deformable_group,
            kernel_size=kernel_size,
            stride=stride,
            rate=rate,
            padding=padding,
            normalizer_fn=None,
            activation_fn=None,
            # may be using zero initializer?
            # weight_init=tf.zeros_initializer,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            biases_initializer=tf.zeros_initializer,
            biases_regularizer=None,
            outputs_collections=offsets_collections,
            scope = 'conv_offsets'
        )
        offsets = tf.transpose(offsets, [0, 3, 1, 2])
        # TODO: MAYA
        offsets *= 0.0

        filters = tf.get_variable(name='weights',
                                  shape= kernel_size + [indim // num_groups, num_outputs],
                                  initializer=weights_initializer,
                                  regularizer=weights_regularizer)

        # transpose filters to required order
        # [outC, inC, ksize, ksize]
        filters = tf.transpose(filters, [3, 2, 0, 1])
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        conv = deform_conv_op.deform_conv_op(x=inputs,
                                             filter=filters,
                                             offset=offsets,
                                             strides=[1, 1] + stride,
                                             rates=[1, 1] + rate,
                                             num_groups=num_groups,
                                             padding=padding,
                                             deformable_group=deformable_group,
                                             name=scope)
        conv = tf.transpose(conv, [0, 2, 3, 1])

        # tf.add_to_collection(outputs_collections, conv)
        if normalizer_fn is not None:
            conv = normalizer_fn(conv)
        elif biases_initializer is not None:
            biases = tf.get_variable(name='biases',
                                     shape=[num_outputs],
                                     initializer=biases_initializer,
                                     regularizer=biases_regularizer,
                                     collections=BIAS_COLLECTIONS)
            conv = conv + biases

        if activation_fn is not None:
            conv = activation_fn(conv)

    tf.add_to_collection(outputs_collections, conv)
    return conv


def ps_roi_pooling(features, bboxes, spatial_scale, output_dim, group_size=3, data_format='NHWC'):
    """
    Python Wrapper for PS-ROI-POOLING LAYER
    :param features: [N, H, W, output_dim * group_size, group_size]
    :param bboxes: [M, 5], which batch id, ymin, xmin, ymax, xmax
    :param spatial_scale: how much bboxes should scale
    :param output_dim: (usually class numbers)
    :param group_size: (#bin row-wise)
    :param data_format: 'NHWC' as default
    :return:
    """
    if data_format == 'NHWC':
        data_format = 'NCHW'
        features = tf.transpose(features, [0, 3, 1, 2])

    # [N, C, gs, gs]
    ps_roi_pooling_ = ps_roi_pooling_op._ps_roi_pooling(features, bboxes, spatial_scale, output_dim, group_size, 'NCHW')

    if data_format == 'NHWC':
        ps_roi_pooling_ = tf.transpose(ps_roi_pooling_, (0, 2, 3, 1))

    return ps_roi_pooling_


# if __name__ == '__main__':
#     inputs = tf.get_variable(shape=[1, 5, 5, 1], dtype=tf.float32, name='inputs')
#     x = deform_conv2d(inputs, 1, [3, 3], strides=[1, 1], ratios=[4, 4], name=None, padding='SAME',
#                   activate=None, deformable_group=1, num_groups=1,
#                   batch_norm=None, group_norm=False, use_bias=None)
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     x_v = sess.run(x)
#     print(x_v.shape, x_v)