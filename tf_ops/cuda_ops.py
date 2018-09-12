import tensorflow as tf

from tf_ops.cuda_layers.deformable_conv_lib import deform_conv_op
from tf_ops.cuda_layers.ps_roi_pooling import ps_roi_pooling_op
from tf_ops.cuda_layers.ps_roi_aligning.ps_roi_aligning_op import ps_roi_aligning
from tf_ops.cuda_layers.roi_align.roi_align_op import roi_align

from tf_ops.wrap_ops import same_padding, tensor_shape, get_variable, conv2d, batch_norm2d, GroupNorm2D

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
def deform_conv2d(inputs, outc, ksize, strides=[1, 1], ratios=[1, 1], name=None, padding='SAME',
                  activate=tf.nn.relu, deformable_group=1, num_groups=1,
                  batch_norm=True, group_norm=False, use_bias=None,
                  weight_init=None, weight_reg=None,
                  bias_init=tf.zeros_initializer, bias_reg=None,
                  offset_init=tf.zeros_initializer, offset_reg=None,
                  outputs_collections=None, offsets_collections='offsets'):
    """
    Wrapper for Conv layers
    :param inputs: [N, H, W, C]
    :param outc: output channels
    :param ksize: [hk, wk]
    :param strides: [hs, ws]
    :param ratios: [hr, wr]
    :param name: var_scope & operation name
    :param padding: padding mode
    :param activate: activate function
    :param batch_norm: whether performs batch norm
    :param use_bias: whether use bias addition
    :param weight_init: weight initializer
    :param weight_reg: weight regularizer
    :param bias_init: bias initializer
    :param bias_reg: bias regularizer
    :param outputs_collections: add result to some collection
    :return: convolution after activation
    """
    # can't use both
    if use_bias is None:
        use_bias = not batch_norm
    assert not (batch_norm and use_bias)
    assert outc % num_groups == 0, print('outc % num_groups != 0')

    with tf.variable_scope(name, 'deform_conv2d'):
        _, iH, iW, indim = tensor_shape(inputs)
        assert indim % num_groups == 0, print('indim % num_groups != 0')
        assert indim % deformable_group == 0, print('indim % deformable_group != 0')

        # use num groups xixi
        filters = get_variable(name='weights', shape=ksize + [indim // num_groups, outc],
                               init=weight_init, reg=weight_reg, collections=WEIGHT_COLLECTIONS)

        # use get_variable merely for debug!
        offsets = conv2d(
            inputs,
            outc=ksize[0] * ksize[1] * 2 * deformable_group,
            ksize=ksize,
            strides=strides,
            ratios=ratios,
            padding=padding,
            batch_norm=False,
            group_norm=False,
            use_bias=True,
            activate=None,
            name='conv_offsets',
            # may be using zero initializer?
            # weight_init=tf.zeros_initializer,
            weight_init=weight_init,
            weight_reg=weight_reg,
            bias_init=tf.zeros_initializer,
            bias_reg=None,
            outputs_collections=offsets_collections
        )
        offsets = tf.transpose(offsets, [0, 3, 1, 2])
        tf.add_to_collection('offsets', offsets)
        # transpose filters to required order
        # [outC, inC, ksize, ksize]
        filters = tf.transpose(filters, [3, 2, 0, 1])

        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        conv = deform_conv_op.deform_conv_op(x=inputs,
                                             filter=filters,
                                             offset=offsets,
                                             strides=[1, 1] + strides,
                                             rates=[1, 1] + ratios,
                                             num_groups=num_groups,
                                             padding=padding,
                                             deformable_group=deformable_group,
                                             name=name)
        conv = tf.transpose(conv, [0, 2, 3, 1])

        # tf.add_to_collection(outputs_collections, conv)
        if batch_norm:
            conv = batch_norm2d(conv)
        elif group_norm:
            conv = GroupNorm2D(conv)
        elif use_bias:
            biases = get_variable(name='biases', shape=[outc], init=bias_init, reg=bias_reg,
                                  collections=BIAS_COLLECTIONS)
            conv = conv + biases

        if activate is not None:
            conv = activate(conv)

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