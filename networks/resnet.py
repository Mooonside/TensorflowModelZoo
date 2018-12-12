from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tf_ops.wrap_ops import group_norm2d, conv2d, sep_conv2d, max_pool2d, batch_norm


def resnet_arg_scope_gn(weight_decay=0.0001, group_norm_num=32, group_norm_dim=None):
    """Defines the default ResNet arg scope.
    Args:
      weight_decay: The weight decay to use for regularizing the model.
    Returns:
      An `arg_scope` to use for the resnet models.
    """
    with arg_scope(
            [conv2d],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=initializers.variance_scaling_initializer(),
            activation_fn=nn_ops.relu,
            normalizer_fn=group_norm2d):
        with arg_scope([sep_conv2d],
                       normalizer_fn=group_norm2d,
                       activation_fn_middle=nn_ops.relu,
                       pointwise_weights_regularizer=regularizers.l2_regularizer(weight_decay)):
            with arg_scope([group_norm2d],
                           group_num=group_norm_num,
                           group_size=group_norm_dim,
                           eps=1e-05, affine=True):
                with arg_scope([max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc


def _detectron_img_preprocess(img):
    # turn to BGR order
    img  = img[..., ::-1]
    img -= constant([[[102.9801, 115.9465, 122.7717]]])
    return img


@add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None):
    """Bottleneck residual unit variant with BN after convolutions.

    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.

    Returns:
      The ResNet unit's output.
    """
    with variable_scope.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = conv2d(
                inputs,
                depth, 1,
                stride=stride,
                activation_fn=None,
                scope='shortcut')

        residual = conv2d(
            inputs, depth_bottleneck, 1, stride=1, scope='conv1')
        residual = conv2d(
            residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = conv2d(
            residual, depth, 1, stride=1, activation_fn=None, scope='conv3')

        # utils.collect_named_outputs(outputs_collections, sc.name + '/unrelu', residual)
        output = nn_ops.relu(shortcut + residual)
        return utils.collect_named_outputs(outputs_collections, sc.name, output)


@add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    for block in blocks:
        with variable_scope.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with variable_scope.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # for explicit rates, ignore convolution strides!
                    if unit['rate'] != 1:
                        net = block.unit_fn(net, **dict(unit, stride=1))
                    else:
                        net = block.unit_fn(net, **unit)
            net = utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net


def resnet_v1_block(scope, base_depth, num_units, stride, rate=1):
    """Helper function for creating a resnet_v1 bottleneck block.
    downsampling(stride=2) is done in each stage's first block!
    Args:
        scope: The scope of the block.
        base_depth: The depth of the bottleneck layer for each unit.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.

    Returns:
        A resnet_v1 bottleneck block.
    """
    return resnet_utils.Block(scope, bottleneck,
                              [{
                                  'depth': base_depth * 4,
                                  'depth_bottleneck': base_depth,
                                  'stride': stride,
                                  'rate': rate,
                              }] +
                              [{
                                  'depth': base_depth * 4,
                                  'depth_bottleneck': base_depth,
                                  'stride': 1,
                                  'rate': rate
                              }] * (num_units - 1))



def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 reuse=None,
                 rates=(1, 1, 1, 1),
                 normalize_inside=True,
                 scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=1, rate=rates[0]),
        resnet_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=rates[1]),
        resnet_v1_block('block3', base_depth=256, num_units=6, stride=2, rate=rates[2]),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=rates[3]),
    ]
    return resnet_v1(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        include_root_block=True,
        reuse=reuse,
        normalize_inside=normalize_inside,
        scope=scope)


def resnet_v1_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  reuse=None,
                  rates=(1, 1, 1, 1),
                  normalize_inside=True,
                  scope='resnet_v1_101'):
    """
    ResNet-101 model of [1]. See resnet_v1() for arg and return description.
    speicfy rates as dilation rates used in each stages's stride = 2 conv.
    """
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=1, rate=rates[0]),
        resnet_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=rates[1]),
        resnet_v1_block('block3', base_depth=256, num_units=23, stride=2, rate=rates[2]),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=rates[3]),
    ]
    return resnet_v1(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        include_root_block=True,
        reuse=reuse,
        normalize_inside=normalize_inside,
        scope=scope)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None,
              normalize_inside=True):
    """Removes output_stride, use pre-defined rate

    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: If the target output_stride is not valid.
    """
    if normalize_inside:
        # if no normalization is used outside, use detectron style normalization
        inputs = _detectron_img_preprocess(inputs)

    with variable_scope.variable_scope(
            scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with arg_scope(
                [conv2d, bottleneck, stack_blocks_dense, max_pool2d],
                outputs_collections=end_points_collection):
            with arg_scope([batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    # net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = conv2d(net, 64, 7, 2, scope='conv1')
                    net = max_pool2d(net, 3, 2, scope='pool1')
                net = stack_blocks_dense(net, blocks)
                if global_pool:
                    # Global average pooling.
                    net = math_ops.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
                    net = utils.collect_named_outputs(end_points_collection,
                                                      sc.name+'/gap', net)

                if num_classes is not None:
                    net = conv2d(
                        net,
                        num_classes, 1,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='logits')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = utils.convert_collection_to_dict(end_points_collection)

                if num_classes is not None:
                    end_points['predictions'] = layers_lib.softmax(
                        net, scope='predictions')
                return net, end_points


if __name__ == '__main__':
    import tensorflow as tf
    with tf.device('/CPU:0'):
        inputs = tf.placeholder(shape=[None, 224, 224, 3], name='inputs', dtype=tf.float32)
        with arg_scope(resnet_arg_scope_gn(weight_decay=1e-4, group_norm_num=32)):
            net, end_points = resnet_v1_101(
                inputs,
                num_classes=1001,
                is_training=True,
                global_pool=True,
                reuse=tf.AUTO_REUSE,
                rates=(1, 1, 1, 1),
                scope='resnet_v1_101_gn')
            # print(end_points)
        for key in end_points:
            print(key, end_points[key].shape)
