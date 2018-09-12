"""
Wrapping Functions for Common Use
Written by Yifeng-Chen
"""

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops

from tensorflow.python.ops import nn_ops


add_arg_scope = tf.contrib.framework.add_arg_scope

TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
GLOBAL_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

var_scope = tf.variable_scope
arg_scope = tf.contrib.framework.arg_scope

weight_collections = 'weights_collections'
bias_collections = 'bias_collections'

WEIGHT_COLLECTIONS = [weight_collections, TRAINABLE_VARIABLES, GLOBAL_VARIABLES]
BIAS_COLLECTIONS = [bias_collections, TRAINABLE_VARIABLES, GLOBAL_VARIABLES]
LOSS_COLLECTIONS = tf.GraphKeys.LOSSES

msra_init = tf.contrib.layers.variance_scaling_initializer(
    factor=2.0,
    mode='FAN_IN',
    uniform=False,
    seed=None,
    dtype=tf.float32
)

#################################
# make quick links              #
#################################
batch_norm = layers_lib.batch_norm


@add_arg_scope
def tensor_shape(tensor):
    return [i.value for i in tensor.get_shape()]


@add_arg_scope
def get_variable(name, shape, dtype=tf.float32, device='/CPU:0', init=None, reg=None, collections=None):
    with tf.device(device):
        var = tf.get_variable(name=name, shape=shape, dtype=dtype,
                              initializer=init, regularizer=reg, collections=collections)
    return var


@add_arg_scope
def same_padding(inputs, ksize, ratios):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      ksize: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      ratios: An integer, rate for atrous convolution.

    Returns:
      output: A tensor of size [batch, height_out, width_out, channels] with the
        input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_array = [[0, 0]]
    for idx, k in enumerate(ksize):
        k_effective = k + (k - 1) * (ratios[idx] - 1)
        pad_total = k_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        pad_array.append([pad_beg, pad_end])
    pad_array.append([0, 0])

    padded_inputs = tf.pad(inputs, pad_array)
    return padded_inputs


@add_arg_scope
def conv2d(inputs,
           num_outputs,
           kernel_size,
           stride=1,
           rate=1,
           padding='SAME',
           activation_fn=nn_ops.relu,
           normalizer_fn=None,
           normalizer_params=None,
           weights_initializer=None,
           weights_regularizer=None,
           biases_initializer=tf.zeros_initializer,
           biases_regularizer=None,
           outputs_collections=None,
           scope=None):
    """
    use equally padding
    """
    if padding == 'SAME':
        inputs = same_padding(inputs, [kernel_size, kernel_size], [rate, rate])

    conv = layers_lib.conv2d(inputs,
                             num_outputs,
                             kernel_size,
                             stride=stride,
                             padding='VALID',
                             rate=rate,
                             activation_fn=activation_fn,
                             normalizer_fn=normalizer_fn,
                             normalizer_params=normalizer_params,
                             weights_initializer=weights_initializer,
                             weights_regularizer=weights_regularizer,
                             biases_initializer=biases_initializer,
                             biases_regularizer=biases_regularizer,
                             outputs_collections=outputs_collections,
                             scope=scope)
    return conv


def conv2d_trans(
    inputs,
    num_outputs,
    kernel_size,
    stride=1,
    padding='SAME',
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=init_ops.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None):
    return layers_lib.conv2d_transpose(
            inputs,
            num_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            biases_initializer=biases_initializer,
            biases_regularizer=biases_regularizer,
            reuse=reuse,
            variables_collections=variables_collections,
            outputs_collections=outputs_collections,
            trainable=trainable,
            scope=scope
    )


@add_arg_scope
def group_conv(
        inputs,
        num_outputs,
        kernel_size,
        stride=1,
        rate=1,
        padding='SAME',
        groups=1,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=None,
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer,
        biases_regularizer=None,
        outputs_collections=None,
        scope=None):
    with tf.variable_scope(scope, 'GroupConv'):
        inc = tensor_shape(inputs)[-1]

        assert inc % groups == 0, 'invalid group number'
        assert num_outputs % groups == 0, 'invalid group number'

        group_outc = num_outputs // groups
        group_ins = tf.split(inputs, axis=-1, num_or_size_splits=groups)
        group_outs = []

        for i in range(groups):
            # noinspection PyTypeChecker
            group_conv_ = conv2d(
                group_ins[i],
                num_outputs=group_outc,
                kernel_size=kernel_size,
                stride=stride,
                rate=rate,
                padding=padding,
                activation_fn=None,
                normalizer_fn=None,
                weights_initializer=weights_initializer,
                weights_regularizer=weights_regularizer,
                biases_initializer=biases_initializer,
                biases_regularizer=biases_regularizer,
                scope='group_{}'.format(i)
            )
            group_outs.append(group_conv_)
        conv = tf.concat(group_outs, axis=-1)

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            conv = normalizer_fn(conv, **normalizer_params)

        if activation_fn is not None:
            conv = activation_fn(conv)
    if outputs_collections is not None:
        tf.add_to_collection(outputs_collections, conv)
    return conv


@add_arg_scope
def sep_conv2d(inputs,
               num_outputs,
               kernel_size,
               stride=1,
               rate=1,
               depth_multiplier=1,
               padding='SAME',
               activation_fn=tf.nn.relu,
               activation_fn_middle=None,
               normalizer_fn=None,
               normalizer_params=None,
               weights_initializer=None,
               depthwise_weights_regularizer=None,
               pointwise_weights_regularizer=None,
               biases_initializer=tf.zeros_initializer,
               biases_regularizer=None,
               outputs_collections=None,
               scope='separate_conv'):
    with tf.variable_scope(scope, 'separate_conv'):
        with tf.variable_scope('depthwise_conv'):
            if padding == 'SAME':
                inputs = same_padding(inputs, [kernel_size, kernel_size], [rate, rate])

            indim = tensor_shape(inputs)[-1]
            depthwise_filter = get_variable(name='depthwise_weights',
                                            shape=[kernel_size, kernel_size] + [indim, depth_multiplier],
                                            init=weights_initializer,
                                            reg=depthwise_weights_regularizer,
                                            collections=WEIGHT_COLLECTIONS)
            conv = tf.nn.depthwise_conv2d(
                input=inputs,
                filter=depthwise_filter,
                strides=[1] + [stride, stride] + [1],
                padding='VALID',
                # only specify 2d here!
                rate=[rate, rate],
                name='depthwise_conv',
                data_format="NHWC"
            )
            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                conv = normalizer_fn(conv, **normalizer_params)

            if activation_fn_middle is not None:
                conv = activation_fn_middle(conv)

            tf.add_to_collection(outputs_collections, conv)

        with tf.variable_scope('pointwise_conv'):
            pointwise_filter = get_variable(name='pointwise_weights',
                                            shape=[1, 1] + [indim * depth_multiplier, num_outputs],
                                            init=weights_initializer,
                                            reg=pointwise_weights_regularizer,
                                            collections=WEIGHT_COLLECTIONS)

            conv = tf.nn.conv2d(input=conv,
                                filter=pointwise_filter,
                                strides=[1] + [1, 1] + [1],
                                padding='VALID',
                                use_cudnn_on_gpu=True,
                                data_format="NHWC",
                                dilations=[1] + [1, 1] + [1],
                                name='pointwise_conv')

            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                conv = normalizer_fn(conv, **normalizer_params)
            elif biases_initializer is not None:
                biases = get_variable(name='biases',
                                      shape=[num_outputs],
                                      init=biases_initializer,
                                      reg=biases_regularizer,
                                      collections=BIAS_COLLECTIONS)
                conv = conv + biases

            if activation_fn is not None:
                conv = activation_fn(conv)
            tf.add_to_collection(outputs_collections, conv)
    return conv


@add_arg_scope
def max_pool2d(inputs, kernel_size=2, stride=2, padding='SAME', outputs_collections=None, scope=None):
    """
    eqaully same padding for max_pool2d
    """
    if padding == 'SAME':
        inputs = same_padding(inputs, [kernel_size, kernel_size], [1, 1])

    pool = layers_lib.max_pool2d(
        inputs,
        kernel_size,
        stride=stride,
        padding='VALID',
        data_format='NHWC',
        outputs_collections=outputs_collections,
        scope=scope
    )
    return pool


@add_arg_scope
def avg_pool2d(inputs, kernel_size=2, stride=2, padding='SAME', scope=None, outputs_collections=None):
    """
    SAME PADDING equally
    :param inputs: [N, H, W, C]
    :param kernel_size: kernel size
    :param stride: pooling stride
    :param padding: padding mode
    :param scope: var_scope & operation name
    :param outputs_collections: add result to some collection
    :return:
    """
    if padding == 'SAME':
        inputs = same_padding(inputs, [kernel_size, kernel_size], [1, 1])

    pool = layers_lib.avg_pool2d(
        inputs,
        kernel_size,
        stride=stride,
        padding='VALID',
        data_format='NHWC',
        outputs_collections=outputs_collections,
        scope=scope
    )
    return pool


@add_arg_scope
def l2_norm_1d(inputs, norm_dim=-1, eps=1e-12, scale=True, scale_initializer=tf.ones_initializer, scope=None):
    with tf.variable_scope(scope, 'L2_Norm1D', [inputs]):
        # output = x / sqrt(max(sum(x**2), epsilon))
        outputs = tf.nn.l2_normalize(inputs, norm_dim, eps)
        if scale:
            gamma = get_variable(name='gamma', shape=tensor_shape(inputs)[-1:], dtype=tf.float32,
                                 init=scale_initializer)
            outputs = outputs * gamma
    return outputs


@add_arg_scope
def focal_softmax_with_logits(predictions, labels,
                              ignore_labels=(255,),
                              background_label=0,
                              loss_collections=LOSS_COLLECTIONS,
                              focal_alpha=None,
                              gamma=2):
    dim = tensor_shape(predictions)[-1]
    rank = len(tensor_shape(predictions))
    # labels_shape = tensor_shape(labels)
    logits = tf.reshape(predictions, shape=[-1, dim])
    labels = tf.reshape(labels, [-1])
    # which is all ones
    mask = tf.cast(tf.not_equal(labels, -1), tf.float32)
    for ignore in ignore_labels:
        mask *= tf.cast(tf.not_equal(labels, ignore), tf.float32)

    labels = tf.one_hot(labels, depth=dim)
    labels = tf.stop_gradient(labels)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels, name='sample_wise_loss')

    # loss *= mask
    # add focal loss decay here
    softmax_scores = tf.nn.softmax(logits, axis=-1)
    # 255 's truth score will be zero
    if focal_alpha is not None:
        focal_alpha = tf.cast(labels, dtype=tf.float32) * tf.reshape(focal_alpha, [1] * (rank - 1) + [dim])
        focal_alpha = tf.reduce_sum(focal_alpha, axis=-1)
    truth_scores = softmax_scores * tf.cast(labels, dtype=tf.float32)

    truth_scores = tf.reduce_sum(truth_scores, axis=-1)
    # 255 's focal loss weight will be 1
    focal_decay = tf.pow(1 - truth_scores, gamma)
    # before visualize set those weight to 0
    focal_decay *= mask
    # set alpha
    if focal_alpha is not None:
        focal_decay *= focal_alpha

    loss *= focal_decay

    # if No bg label: normalize by non-background pixels
    if background_label is not None:
        loss = tf.reduce_sum(loss)
        non_bg_pixels = tf.reduce_sum(tf.cast(tf.not_equal(labels, background_label), tf.float32))
        loss /= non_bg_pixels
    else:
        loss = tf.reduce_mean(loss)

    if loss_collections is not None:
        tf.add_to_collection(loss_collections, loss)
    return loss


@add_arg_scope
def softmax_with_logits(predictions, labels,
                        ignore_labels=(255,),
                        loss_collections=LOSS_COLLECTIONS,
                        reduce_method='nonzero_mean',
                        weights=None):
    """
    a loss vector [N*H*W, ]
    :param reduce_method: 'nonzero_mean' / 'sum' / 'mean'
    :param predictions: [N, H, W, c], raw outputs of model
    :param labels: [N ,H, W, 1] int32
    :param ignore_labels: ignore pixels with ground truth in ignore_labels
    :param loss_collections: add to which loss collections
    :param weights: set weight to each loss
    :return: a sample_mean loss
    """
    dim = tensor_shape(predictions)[-1]
    logits = tf.reshape(predictions, shape=[-1, dim])
    labels = tf.reshape(labels, [-1])
    # which is all ones
    mask = tf.ones_like(labels, dtype=tf.float32)
    for ignore in ignore_labels:
        mask *= tf.cast(tf.not_equal(labels, ignore), tf.float32)

    labels = tf.one_hot(labels, depth=dim)
    labels = tf.stop_gradient(labels)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels, name='sample_wise_loss')

    loss *= mask
    if weights is not None:
        loss *= weights

    if reduce_method == 'nonzero_mean':
        non_zero_items = tf.reduce_sum(tf.cast(tf.not_equal(loss, 0), tf.float32))
        non_zero_items = tf.maximum(non_zero_items, 1)
        loss = tf.divide(tf.reduce_sum(loss), non_zero_items, name='mean_loss')
    elif reduce_method == 'sum':
        loss = tf.reduce_sum(loss)
    else:
        loss = tf.reduce_mean(loss)

    if loss_collections is not None:
        tf.add_to_collection(loss_collections, loss)
    return loss


def smooth_l1(x):
    square_selector = tf.cast(tf.less(tf.abs(x), 1), tf.float32)
    x = square_selector * 0.5 * tf.square(x) + (1 - square_selector) * (tf.abs(x) - 0.5)
    return x


def safe_divide(numerator, denominator, name):
    """Divides two values, returning 0 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(
        tf.greater(denominator, 0),
        tf.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)


@add_arg_scope
def group_norm2d(inputs, group_size=None, group_num=None, eps=1e-05, affine=True, scope=None):
    """
    specify wither group size or group num
    do group normalization!
    """
    with tf.variable_scope(scope, default_name='GroupNorm2d'):
        _, h, w, channel_num = tensor_shape(inputs)

        if group_size is None:
            assert channel_num % group_num == 0, 'channels {} must be divided by group nums {}'.format(channel_num,
                                                                                                       group_size)
            group_size = channel_num // group_num

        if group_num is None:
            assert channel_num % group_size == 0, 'channels {} must be mutilples of group sizes {}'.format(channel_num,
                                                                                                           group_size)
            group_num = channel_num // group_size

        # convert [N, C, H, W]
        inputs = tf.transpose(inputs, (0, 3, 1, 2))
        # [N, G, C // G, H, W]
        inputs = tf.reshape(inputs, [-1, group_num, group_size, h, w])
        mean, variance = tf.nn.moments(inputs, axes=[2, 3, 4], keep_dims=True)
        outputs = (inputs - mean) / tf.sqrt(variance + eps)
        outputs = tf.reshape(outputs, [-1, channel_num, h, w])
        outputs = tf.transpose(outputs, (0, 2, 3, 1))

        if affine:
            beta = tf.get_variable('beta', [channel_num], initializer=tf.zeros_initializer,
                                   collections=BIAS_COLLECTIONS)
            gamma = tf.get_variable('gamma', [channel_num], initializer=tf.ones_initializer,
                                    collections=WEIGHT_COLLECTIONS)
            outputs = outputs * gamma + beta

        return outputs


@add_arg_scope
def drop_out(inputs, keep_prob, is_training=True, name=None):
    if type(keep_prob) != float:
        print('Invalid Parameter Specified {}'.format(keep_prob))
    if is_training:
        return tf.nn.dropout(inputs, keep_prob=keep_prob, name=name)
    else:
        return inputs
