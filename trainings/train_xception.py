import os
from datetime import datetime
from time import time

import tensorflow as tf
from numpy import ceil

from networks.xception import xception_arg_scope_gn, xception_65
from datasets.ilsvrc.ilsvrc_reader import get_dataset, get_next_batch, TRAIN_DIR, TRAIN_NUM, inception_augmentation
from tf_ops.wrap_ops import softmax_with_logits
from tf_utils import parse_device_name, average_gradients, partial_restore, LOCAL_VARIABLE_UPDATE_OPS

arg_scope = tf.contrib.framework.arg_scope

flags = tf.app.flags
flags.DEFINE_string('data_dir', TRAIN_DIR, 'where training set is put')
flags.DEFINE_integer('reshape_height', 224, 'reshape height')
flags.DEFINE_integer('reshape_weight', 224, 'reshape weight')
flags.DEFINE_integer('num_classes', 1001, '#classes')

# learning configs
flags.DEFINE_integer('epoch_num', 5, 'epoch_nums')
flags.DEFINE_integer('epoch_len', TRAIN_NUM, 'epoch_len')
flags.DEFINE_integer('batch_size', 16, 'batch size')
# 0.045 in xception paper
# LOG : 0.045/10 => 0.02 for 6 => 0.01 for 5
flags.DEFINE_float('weight_learning_rate', 0.01, 'weight learning rate')
# 0.05 in train ILSVRC in one hour
# flags.DEFINE_float('weight_learning_rate', 0.05, 'weight learning rate')

flags.DEFINE_float('bias_learning_rate', None, 'bias learning rate')
flags.DEFINE_float('decay_rate', 0.9, 'learning rate decay')
# not using exponential decay
flags.DEFINE_float('decay_epochs', None, 'learning rate decay epochs')
# using polynomial decay
flags.DEFINE_float('max_iter', 50, 'maximum_iteration_epochs')
flags.DEFINE_float('ema_decay', 0.99, 'decay for ema')

flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_float('clip_grad_by_norm', 5, 'clip_grad_by_norm')

# deploy configs
flags.DEFINE_string('store_device', '/CPU:0', 'where to place the variables')
flags.DEFINE_string('run_device', '0,1,2,3', 'where to run the models')
flags.DEFINE_boolean('allow_growth', True, 'allow memory growth')

# regularization
flags.DEFINE_float('weight_decay', 1e-5, 'weight regularization scale')
flags.DEFINE_float('bias_reg_scale', None, 'bias regularization scale')
flags.DEFINE_string('bias_reg_func', None, 'use which func to regularize bias')

# model load & save configs
flags.DEFINE_string('summaries_dir',
                    '/data/chenyifeng/TF_Logs/TensorflowModelZoo/xception_gn_{}'.format(
                        datetime.isoformat(datetime.today())),
                    'where to store summary log')

flags.DEFINE_string('pretrained_ckpts',
                    None,
                    'where to load pretrained model')

flags.DEFINE_string('last_ckpt', '/data/chenyifeng/TF_Models/atrain/TensorflowModelZoo/xception_gn',
                    'where to load last saved model')

flags.DEFINE_string('next_ckpt', '/data/chenyifeng/TF_Models/atrain/TensorflowModelZoo/xception_gn',
                    'where to store current model')

FLAGS = flags.FLAGS
store_device = parse_device_name(FLAGS.store_device)
run_device = parse_device_name(FLAGS.run_device)

# config devices
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
if not 'CPU' in FLAGS.run_device:
    GPU_NUMS = len(FLAGS.run_device.split(','))
    print('Deploying Model on {} GPUs'.format(GPU_NUMS))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(FLAGS.run_device)
    config.gpu_options.allow_growth = FLAGS.allow_growth
else:
    GPU_NUMS = 1
    print('Deploying Model on CPU')

# set up step
sess = tf.Session(config=config)
EPOCH_STEPS = int(ceil(FLAGS.epoch_len / FLAGS.batch_size / GPU_NUMS))

with tf.device(store_device):
    global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
    # read data
    reshape_size = [FLAGS.reshape_height, FLAGS.reshape_weight]
    dataset = get_dataset(FLAGS.data_dir,
                          FLAGS.batch_size,
                          FLAGS.epoch_num,
                          reshape_size,
                          augment_func=inception_augmentation,
                          num_readers=4)


def inference_to_loss():
    name_batch, image_batch, label_batch = get_next_batch(dataset)
    with arg_scope(xception_arg_scope_gn(weight_decay=FLAGS.weight_decay)):
        net, end_points = xception_65(image_batch,
                                      global_pool=True,
                                      num_classes=FLAGS.num_classes,
                                      is_training=True,
                                      output_stride=32,
                                      keep_prob=0.5)
        logits = tf.squeeze(net, axis=[1, 2])
        loss = softmax_with_logits(logits, label_batch,
                                   ignore_labels=[],
                                   loss_collections=tf.GraphKeys.LOSSES,
                                   reduce_method='nonzero_mean')

        # add accuracy
        predictions = tf.argmax(logits, axis=-1, output_type=label_batch.dtype)
        label_batch = tf.squeeze(label_batch, axis=-1)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, label_batch), tf.float32))
        metrics_loss, _ = tf.metrics.mean(loss, updates_collections=LOCAL_VARIABLE_UPDATE_OPS)
        metrics_acc, _ = tf.metrics.mean(accuracy, updates_collections=LOCAL_VARIABLE_UPDATE_OPS)

        # add summaries
        with tf.name_scope('summary_input_output'):
            # tf.summary.image('image_batch', image_batch, max_outputs=3)
            # tf.summary.histogram('label_batch', label_batch)
            tf.summary.scalar('classfication_loss', metrics_loss)
            tf.summary.scalar('classfication_acc', metrics_acc)
    return loss, accuracy


# learning_rate = FLAGS.weight_learning_rate
learning_rate = tf.train.polynomial_decay(learning_rate=FLAGS.weight_learning_rate,
                                          global_step=global_step,
                                          decay_steps=FLAGS.max_iter * EPOCH_STEPS,
                                          power=FLAGS.decay_rate,
                                          end_learning_rate=1e-5)
# Exp learning rate decay
# learning_rate = tf.train.exponential_decay(
#     learning_rate = FLAGS.weight_learning_rate,
#     global_step=global_step,
#     decay_rate=FLAGS.decay_rate,
#     decay_steps=EPOCH_STEPS * FLAGS.decay_epoch,
# )

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=FLAGS.momentum)
ratio = (FLAGS.weight_learning_rate / FLAGS.bias_learning_rate) \
    if FLAGS.bias_learning_rate is not None else 2

print('Setting bias lr {}, weights lr {}'.format(ratio * FLAGS.weight_learning_rate,
                                                 FLAGS.weight_learning_rate))

gather_losses = []
gather_accs = []
gather_weight_gradients = []
gather_bias_gradients = []

ema = tf.train.ExponentialMovingAverage(FLAGS.ema_decay)

first_clone_scope = None
with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    for gpu_id in range(GPU_NUMS):
        with tf.device('/GPU:{}'.format(gpu_id)):
            with tf.name_scope('tower_{}'.format(gpu_id)):
                clone_scope = tf.contrib.framework.get_name_scope()
                loss, acc = inference_to_loss()
                reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

                if gpu_id == 0:
                    first_clone_scope = clone_scope

                total_loss = loss + reg_loss
                # print(tf.global_variables())
                # solve for gradients
                weight_vars = []
                bias_vars = []

                vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                for var in vars:
                    if 'bias' in var.name.split('/')[-1]:
                        bias_vars.append(var)
                    else:
                        weight_vars.append(var)
                # print(len(weight_vars), len(bias_vars))

                weight_grads = optimizer.compute_gradients(total_loss, weight_vars)
                weight_grads = [(tf.clip_by_norm(grad, clip_norm=FLAGS.clip_grad_by_norm), var)
                                for grad, var in weight_grads if grad is not None]

                bias_grads = optimizer.compute_gradients(total_loss, bias_vars)
                bias_grads = [(tf.clip_by_norm(ratio * grad, clip_norm=FLAGS.clip_grad_by_norm), var)
                              for grad, var in bias_grads if grad is not None]

                gather_losses.append(total_loss)
                gather_accs.append(acc)
                gather_weight_gradients.append(weight_grads)
                gather_bias_gradients.append(bias_grads)

gather_weight_gradients = average_gradients(gather_weight_gradients)
gather_bias_gradients = average_gradients(gather_bias_gradients)
gather_losses = tf.add_n(gather_losses) / len(gather_losses)
gather_accs = tf.add_n(gather_accs) / len(gather_accs)

metrics_gather_losses, _ = tf.metrics.mean(gather_losses, updates_collections=LOCAL_VARIABLE_UPDATE_OPS)
metrics_gather_accs, _ = tf.metrics.mean(gather_accs, updates_collections=LOCAL_VARIABLE_UPDATE_OPS)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=first_clone_scope)
update_ops += tf.get_collection(LOCAL_VARIABLE_UPDATE_OPS)
print(len(update_ops))

# TODO: Turn Off Ema Temporarily
# with tf.control_dependencies(update_ops):
#     grads_op = optimizer.apply_gradients(gather_weight_gradients + gather_bias_gradients,
#                                          global_step=global_step)
#     with tf.control_dependencies([grads_op]):
#         train_op = ema.apply(weight_vars + bias_vars)
#         ema_vars = [ ema.average(var) for var in weight_vars + bias_vars]

with tf.control_dependencies(update_ops):
    train_op = optimizer.apply_gradients(gather_weight_gradients + gather_bias_gradients,
                                         global_step=global_step)

with tf.name_scope('summary_gathers'):
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('gather_losses', metrics_gather_losses)
    tf.summary.scalar('gather_accs', 100.0 * metrics_gather_accs)
    tf.summary.scalar('regularize_loss', reg_loss)

# with tf.name_scope('summary_vars'):
#     for weight in weight_vars:
#         add_var_summary(weight)
#     for bias in bias_vars:
#         add_var_summary(bias)
#
# with tf.name_scope('summary_gradients'):
#     for grad in gather_weight_gradients:
#         add_var_summary(grad[0])
#     for grad in gather_bias_gradients:
#         add_var_summary(grad[0])

# with tf.name_scope('ema_vars'):
#     for ema_var in ema_vars:
#         add_var_summary(ema_var)
#
merge_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, sess.graph)
# saver = tf.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=5)
# print(tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
# # print(tf.global_variables())

# may want to reinitialize last layer
# saver = tf.train.Saver(var_list = [ i for i in tf.global_variables() if not i.name.startswith('xception_65/logits')])
saver = tf.train.Saver(max_to_keep=5)

# initialize
ckpt = None
if FLAGS.last_ckpt is not None:
    ckpt = tf.train.latest_checkpoint(FLAGS.last_ckpt)
    if ckpt is not None:
        # set up save configurationlast_ckpt
        saver.restore(sess, ckpt)
        print('Recovering From {}'.format(ckpt))
elif FLAGS.pretrained_ckpts is not None:
    print('No previous Model Found in {}'.format(ckpt))
    # pre-train priority higher
    sess.run(tf.global_variables_initializer())
    partial_restore_op = partial_restore(tf.global_variables(), FLAGS.pretrained_ckpts)
    sess.run(partial_restore_op)
    print('Recovering From Pretrained Model {}'.format(FLAGS.pretrained_ckpts))
else:
    print('Training From Scartch TvT')
    sess.run(tf.global_variables_initializer())

# reinitialize softmax
# last_layer_vars = [ i for i in tf.global_variables() if i.name.startswith('xception_65/logits')]
# last_layer_vars_init = tf.variables_initializer(last_layer_vars)
# sess.run(last_layer_vars_init)
# sess.run(tf.assign(global_step, 0))

try:
    local_step = 0
    epoch_basis = int(ceil(sess.run(global_step) / EPOCH_STEPS))
    print(epoch_basis)

    for epoch_id in range(epoch_basis, epoch_basis + FLAGS.epoch_num):
        sess.run(tf.local_variables_initializer())
        for batch_id in range(EPOCH_STEPS):
            tic = time()
            losses_v, accs_v, step, summary, _ = \
                sess.run([metrics_gather_losses, metrics_gather_accs, global_step, merge_summary, train_op])
            toc = time()
            train_writer.add_summary(summary, step)
            local_step += 1
            # save model per xxx steps

            print("{}-{}-{} : loss {:.4f} acc {:.4f} cost {:.3f}s"
                  .format(epoch_id, batch_id, step, losses_v, accs_v * 100, toc - tic))

        save_path = saver.save(sess, os.path.join(FLAGS.next_ckpt, '{:.3f}_{}.ckpt'.format(losses_v, epoch_id + 1)))
        print("Model saved in path: %s" % save_path)



except tf.errors.OutOfRangeError:
    print('Done training')
