import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

LOCAL_VARIABLE_UPDATE_OPS = 'LOCAL_VARIABLE_UPDATE_OPS'


def inspect_ckpt(ckpt_path):
    """
    inspect var names and var shapes in a checkpoint
    :param ckpt_path: checkpoint name (give prefix if multiple files for one ckpt)
    :return: None
    """
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()
    total_volume = 0
    var_names = sorted(var_to_shape_map.keys())
    for var_name in var_names:
        var = reader.get_tensor(var_name)
        shape = var_to_shape_map[var_name]
        # dtype = var_to_dtype_map[var_name]
        # print(var_name, shape, dtype)
        if 'Momentum' in var_name or 'moving' in var_name:
            pass
        else:
            total_volume += np.prod(shape)

        # print(var_name, shape, np.mean(var), np.var(var))
        print(var_name)
    print('Totol Volume {} MB'.format(total_volume * 4 / 1024 / 1024))


def rename_vars_in_ckpt(ckpt_path, name_map, output_path):
    """
    rename vars in ckpt according to a dict name_map
    :param ckpt_path: original ckpt path
    :param name_map: dict {org_name: new_name}
    :param output_path: renamed ckpt path
    :return: None
    """
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_dtype_map = reader.get_variable_to_dtype_map()

    var_names = sorted(var_to_dtype_map.keys())

    sess = tf.Session()

    for var_name in var_names:
        var = reader.get_tensor(var_name)
        dtype = var_to_dtype_map[var_name]
        if var_name in name_map.keys():
            newname = name_map[var_name]
            tf.get_variable(name=newname, dtype=dtype, initializer=var)
        else:
            print('Ignoring {}'.format(var_name))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, save_path=output_path)
    sess.close()
    print('Renamed Model Saved')


def partial_restore(cur_var_lists, ckpt_path):
    """
    return an assign op that restoring existing vars in checkpoint
    :param cur_var_lists: variables trying to restore
    :param ckpt_path: checkpoint path
    :return: an assignment function
    """
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    assign_op = []
    for var in list(cur_var_lists):
        name = var.name.rstrip(':0')
        if name not in var_to_shape_map.keys():
            print('NO {} Found in ckpt So deserted.'.
                  format(name))
            continue

        if var_to_shape_map[name] != var.shape:
            print('{} shape disagrees, lhs {}, rhs {}. So deserted.'.
                  format(var.name, var_to_shape_map[name], var.shape))
            continue
        print('Restoring {}'.format(var.name))
        op = tf.assign(var, reader.get_tensor(name))
        assign_op.append(op)

    return tf.group(assign_op)


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)


def add_var_summary(var):
    if var is not None:
        tf.summary.histogram(var.name + "/value", var)


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_iou_summary(ious, classnames):
    for idx, name in enumerate(classnames):
        tf.summary.scalar(name + '_iou', ious[idx])


def parse_device_name(name):
    name = str(name)
    if 'cpu' in name.lower():
        return '/CPU:0'
    elif name in '012345678':
        return '/GPU:{}'.format(name)
    else:
        return None


# =========================================================================== #
# General tools.
# =========================================================================== #
def reshape_list(l, shape=None, idx=0):
    if shape is None:
        r = []
        if type(l) is list or type(l) is tuple:
            shape = []
            for a in l:
                flatten, orig_shape = reshape_list(a)
                r += flatten
                shape.append(orig_shape)
            return r, shape
        else:
            return [l], 1
    else:
        r = []
        for s in shape:
            if type(s) is list or type(s) is tuple:
                re, idx = reshape_list(l, s, idx)
                r.append(re)
            else:
                r.append(l[idx])
                idx += 1
        return r, idx


# def reshape_list(l, shape=None):
#     """Reshape list of (list): 1D to 2D or the other way around.
#
#     Args:
#       l: List or List of list.
#       shape: 1D or 2D shape.
#     Return
#       Reshaped list.
#     """
#     r = []
#     if shape is None:
#         # Flatten everything.
#         for a in l:
#             if isinstance(a, (list, tuple)):
#                 r = r + list(a)
#             else:
#                 r.append(a)
#     else:
#         # Reshape to list of list.
#         i = 0
#         for s in shape:
#             if s == 1:
#                 r.append(l[i])
#             else:
#                 r.append(l[i:i+s])
#             i += s
#     return r


def stream_to_batch(inputs_lists, batch_size, num_threads):
    flatten_list, orig_shape = reshape_list(inputs_lists)
    r = tf.train.batch(
        # flatten it
        flatten_list,
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size)
    re, idx = reshape_list(r, orig_shape)
    return re


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    inspect_ckpt('/home/chenyifeng/TF_Models/atrain/xception_gn/./0.935_39.ckpt')
    # # inspect_ckpt('/mnt/disk50_CHENYIFENG/TF_Models/ptrain/SEGS/xception_gn/test.ckpt')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # old_names = open('old_xception.txt').readlines()
    # old_names = [ i.rstrip(' \n')for i in old_names]
    # new_names = open('new_xception.txt').readlines()
    # new_names = [ i.rstrip(' \n')for i in new_names]
    # #
    # name_map = dict(zip(old_names, new_names))
    # rename_vars_in_ckpt(ckpt_path='/mnt/disk50_CHENYIFENG/TF_Models/ptrain/xception/model.ckpt',
    #                     name_map=name_map,
    #                     output_path='/mnt/disk50_CHENYIFENG/TF_Models/ptrain/SEGS/xception_bn/model.ckpt')
