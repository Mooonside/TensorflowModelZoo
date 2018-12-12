import os
from datetime import datetime
from time import time

import tensorflow as tf
from numpy import ceil

from datasets.ilsvrc.ilsvrc_reader import TRAIN_DIR, TRAIN_NUM, VALID_NUM, VALIDATION_DIR,\
    get_dataset, get_next_batch, inception_augmentation, eval_preprocess
from networks.deformable_resnet import resnet_v1_101_deform, resnet_arg_scope_deform_gn
from tf_ops.wrap_ops import softmax_with_logits, tensor_shape
from tf_utils import parse_device_name, average_gradients, partial_restore, LOCAL_VARIABLE_UPDATE_OPS

arg_scope = tf.contrib.framework.arg_scope

# define flags
flags = tf.app.flags

flags.DEFINE_string('train_dir', TRAIN_DIR, 'where training set is put')
flags.DEFINE_string('valid_dir', VALIDATION_DIR, 'where validation set is put')
flags.DEFINE_integer('reshape_height', 224, 'reshape height')
flags.DEFINE_integer('reshape_weight', 224, 'reshape weight')
flags.DEFINE_integer('num_classes', 1000, '#classes')

# learning configs
flags.DEFINE_integer('epoch_num', 10, 'epoch_nums')
flags.DEFINE_integer('epoch_len', TRAIN_NUM, 'epoch_len')
flags.DEFINE_integer('valid_per_epoch', 1, 'valid_per_epoch')
flags.DEFINE_integer('batch_size', 8, 'batch size')
# 0.045 in xception paper
# flags.DEFINE_float('weight_learning_rate', 0.045, 'weight learning rate')
# 0.05 in train ILSVRC in one hour
# flags.DEFINE_float('weight_learning_rate', 0.05, 'weight learning rate')
# since we are fine tuning, we set learning rate to be smaller
flags.DEFINE_float('weight_learning_rate', 0.01, 'weight learning rate')
flags.DEFINE_float('bias_learning_ratio', 2, 'bias learning ratio')
flags.DEFINE_float('offset_learning_ratio', 10, 'offset learning rate')
flags.DEFINE_float('decay_rate', 0.9, 'learning rate decay')
flags.DEFINE_float('max_iter_epoch', 10, 'max_iter_epoch')
flags.DEFINE_float('end_learning_rate', 1e-5, 'end_learning_rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_float('clip_grad_by_norm', 5, 'clip_grad_by_norm')

# deploy configs
flags.DEFINE_string('store_device', '/CPU:0', 'where to place the variables')
flags.DEFINE_string('run_device', '0,1,2,3', 'where to run the models')
flags.DEFINE_boolean('allow_growth', True, 'allow memory growth')

# regularization
flags.DEFINE_float('weight_decay', 1e-4, 'weight regularization scale')
flags.DEFINE_float('bias_reg_scale', None, 'bias regularization scale')
flags.DEFINE_string('bias_reg_func', None, 'use which func to regularize bias')

# model load & save configs
flags.DEFINE_string('summaries_dir',
                    '~/volume/TF_Logs/TensorflowModelZoo/resnet101_deform/',
                    'where to store summary log')

flags.DEFINE_string('pretrained_ckpts', '/home/chenyifeng/TF_Models/ptrain/ILSVRC/ResNet_101_V1_GN/model.ckpt',
                    'where to load pretrained model')

flags.DEFINE_string('last_ckpt', None,
                    'where to load last saved model')

flags.DEFINE_string('next_ckpt', '/home/chenyifeng/TF_Models/atrain/TensorflowModelZoo/resnet101_deform/',
                    'where to store current model')

FLAGS = flags.FLAGS


# config devices
store_device = parse_device_name(FLAGS.store_device)
run_device = parse_device_name(FLAGS.run_device)
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

if not 'CPU' in FLAGS.run_device:
    GPU_NUMS = len(FLAGS.run_device.split(','))
    print('====================================Deploying Model on {} GPUs===================================='
          .format(GPU_NUMS))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(FLAGS.run_device)
    config.gpu_options.allow_growth = FLAGS.allow_growth
else:
    GPU_NUMS = 1
    print('====================================Deploying Model on CPU====================================')

TRAIN_EPOCH_STEPS = int(ceil(TRAIN_NUM / FLAGS.batch_size / GPU_NUMS))
VALID_EPOCH_STEPS = int(ceil(VALID_NUM / FLAGS.batch_size / GPU_NUMS))
print(TRAIN_EPOCH_STEPS, VALID_EPOCH_STEPS)


def inference_to_loss(image_batch, label_batch, is_training=True):
    with arg_scope(resnet_arg_scope_deform_gn(weight_decay=FLAGS.weight_decay, group_norm_num=32)):
        ########################### Define Network Structure ############################
        net, end_points = resnet_v1_101_deform(image_batch,
                                              num_classes=FLAGS.num_classes,
                                              is_training=is_training,
                                              global_pool=True,
                                              rates=(1, 1, 2, 4),
                                              scope='resnet_v1_101_gn')

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

        with tf.name_scope('summary_input_output'):
            tf.summary.scalar('classfication_loss', metrics_loss)
            tf.summary.scalar('classfication_acc', metrics_acc)
    return loss, metrics_acc


print("====================================BUILDING UP TRAINING GRAPH====================================")
train_graph = tf.Graph()
with train_graph.as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
    train_dataset = get_dataset(FLAGS.train_dir,
                                FLAGS.batch_size * GPU_NUMS,
                                FLAGS.epoch_num,
                                reshape_size=[FLAGS.reshape_height, FLAGS.reshape_weight],
                                augment_func=inception_augmentation,
                                num_readers=4)

    _, train_image, train_labels = get_next_batch(train_dataset)
    # TODO: MINUS train_labels by 1
    train_labels -= 1

    learning_rate = tf.train.polynomial_decay(learning_rate=FLAGS.weight_learning_rate,
                                              global_step=global_step,
                                              decay_steps=FLAGS.max_iter_epoch * TRAIN_EPOCH_STEPS,
                                              power=FLAGS.decay_rate,
                                              end_learning_rate=FLAGS.end_learning_rate)
    # learning_rate = FLAGS.weight_learning_rate
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=FLAGS.momentum)

    train_gather_losses = []
    train_gather_accs = []
    train_gather_gradients = []

    # ema = tf.train.ExponentialMovingAverage(FLAGS.ema_decay)
    first_clone_scope = None
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for gpu_id in range(GPU_NUMS):
            with tf.device('/GPU:{}'.format(gpu_id)):
                with tf.name_scope('tower_{}'.format(gpu_id)):
                    clone_scope = tf.contrib.framework.get_name_scope()
                    train_loss, train_macc = inference_to_loss(
                        train_image[gpu_id * FLAGS.batch_size:(gpu_id + 1) * FLAGS.batch_size, ...],
                        train_labels[gpu_id * FLAGS.batch_size:(gpu_id + 1) * FLAGS.batch_size, ...],
                        is_training=True,
                    )
                    train_reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                    if gpu_id == 0:
                        first_clone_scope = clone_scope
                        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

                    train_total_loss = train_loss + train_reg_loss
                    grads = optimizer.compute_gradients(train_total_loss, vars)
                    train_gather_losses.append(train_total_loss)
                    train_gather_accs.append(train_macc)
                    train_gather_gradients.append(grads)

    train_gather_gradients = average_gradients(train_gather_gradients)
    ratio_gradients = []
    for (grad, var) in train_gather_gradients:
        rate = 1
        if 'bias' in var.name:
            rate *= FLAGS.bias_learning_ratio
        if 'offset' in var.name:
            rate *= FLAGS.offset_learning_ratio
        ratio_gradients.append((tf.clip_by_norm(grad, clip_norm=FLAGS.clip_grad_by_norm), var))

    train_gather_losses = tf.add_n(train_gather_losses) / len(train_gather_losses)
    train_gather_accs = tf.add_n(train_gather_accs) / len(train_gather_accs)

    train_metrics_gather_losses, _ = tf.metrics.mean(train_gather_losses, updates_collections=LOCAL_VARIABLE_UPDATE_OPS)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=first_clone_scope)
    update_ops += tf.get_collection(LOCAL_VARIABLE_UPDATE_OPS)
    print(len(update_ops))

    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(ratio_gradients, global_step=global_step)

    with tf.name_scope('overall_summary'):
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('gather_losses', train_metrics_gather_losses)
        tf.summary.scalar('gather_accs', 100.0 * train_gather_accs)
        tf.summary.scalar('regularize_loss', train_reg_loss)

    train_merge_summary = tf.summary.merge_all()
    print(len(tf.global_variables()))
    train_saver = tf.train.Saver(keep_checkpoint_every_n_hours=4.0, var_list=tf.global_variables())

print("====================================END BUILDING TRIANING GRAPH====================================")


if FLAGS.valid_per_epoch is not None:
    print("====================================BUILDING UP EVALUATION GRAPH====================================")
    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        eval_dataset = get_dataset(FLAGS.valid_dir,
                                   batch_size=FLAGS.batch_size * GPU_NUMS,
                                   num_epochs=FLAGS.epoch_num + 1,
                                   augment_func=eval_preprocess,
                                   reshape_size=[224, 224],
                                   num_readers=4,
                                   shuffle=None)
        _, eval_image, eval_labels = get_next_batch(eval_dataset)
        # TODO: labels -= 1
        eval_labels -= 1

        eval_gather_losses = []
        eval_gather_accs = []

        # ema = tf.train.ExponentialMovingAverage(FLAGS.ema_decay)
        first_clone_scope = None
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for gpu_id in range(GPU_NUMS):
                with tf.device('/GPU:{}'.format(gpu_id)):
                    with tf.name_scope('tower_{}'.format(gpu_id)):
                        clone_scope = tf.contrib.framework.get_name_scope()
                        eval_loss, eval_macc = inference_to_loss(
                            eval_image[gpu_id * FLAGS.batch_size:(gpu_id + 1) * FLAGS.batch_size, ...],
                            eval_labels[gpu_id * FLAGS.batch_size:(gpu_id + 1) * FLAGS.batch_size, ...],
                            is_training=False,
                        )
                        eval_reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        if gpu_id == 0:
                            first_clone_scope = clone_scope
                            eval_total_loss = eval_loss + eval_reg_loss
                        # print(tf.global_variables())
                        eval_gather_losses.append(eval_total_loss)
                        eval_gather_accs.append(eval_macc)

        eval_gather_losses = tf.add_n(eval_gather_losses) / len(eval_gather_losses)
        eval_gather_accs = tf.add_n(eval_gather_accs) / len(eval_gather_accs)
        eval_metrics_gather_losses, _ = tf.metrics.mean(eval_gather_losses, updates_collections=LOCAL_VARIABLE_UPDATE_OPS)

        eval_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=first_clone_scope)
        eval_update_ops += tf.get_collection(LOCAL_VARIABLE_UPDATE_OPS)
        print(len(eval_update_ops))
        eval_op = eval_update_ops
        print(len(tf.global_variables()))
        eval_loader = tf.train.Saver(var_list=tf.global_variables())
    print("====================================END BUILDING EVALUATION GRAPH====================================")

# set up session
train_session = tf.Session(config=config, graph=train_graph)
summaries_dir = os.path.join(FLAGS.summaries_dir + '/{}'.format(datetime.isoformat(datetime.today())))
train_writer = tf.summary.FileWriter(summaries_dir, train_session.graph)

if FLAGS.valid_per_epoch is not None:
    eval_session = tf.Session(config=config, graph=eval_graph)

print('====================================Revovering Model====================================')
ckpt = None
if FLAGS.last_ckpt is not None:
    ckpt = tf.train.latest_checkpoint(FLAGS.last_ckpt)
    if ckpt is not None:
        # set up save configuration
        train_saver.restore(train_session, ckpt)
        print('Recovering From {}'.format(ckpt))
elif FLAGS.pretrained_ckpts is not None:
    print('No previous Model Found in {}'.format(ckpt))
    # pre-train priority higher
    with train_graph.as_default():
        train_session.run(tf.global_variables_initializer())
        partial_restore_op = partial_restore(tf.global_variables(), FLAGS.pretrained_ckpts)
        train_session.run(partial_restore_op)
    print('Recovering From Pretrained Model {}'.format(FLAGS.pretrained_ckpts))
else:
    print('Training From Scartch TvT')
    train_session.run(tf.global_variables_initializer())
print('====================================RUNNING!====================================')


try:
    local_step = 0
    epoch_basis = int(ceil(train_session.run(global_step) / TRAIN_EPOCH_STEPS))
    print(epoch_basis)

    for epoch_id in range(epoch_basis, epoch_basis + FLAGS.epoch_num):
        with train_graph.as_default():
            train_session.run(tf.local_variables_initializer())
        print("====================================Start of One Epoch====================================")
        for batch_id in range(TRAIN_EPOCH_STEPS):
            tic = time()
            losses_v, accs_v, step, summary, _ = \
                train_session.run([train_metrics_gather_losses, train_gather_accs,
                                   global_step, train_merge_summary, train_op])
            toc = time()
            train_writer.add_summary(summary, step)
            local_step += 1
            # save model per xxx steps

            print("\t TRAINING {}-{}/{} : loss {:.4f} acc {:.4f} cost {:.3f}s"
                  .format(epoch_id, batch_id+1, TRAIN_EPOCH_STEPS, losses_v, accs_v * 100, toc - tic))
        print("====================================End of One Epoch==============================================")
        ckpt_name = 'T{}_L{:.4f}_A{:.4f}'.format(epoch_id + 1, losses_v, accs_v)

        if FLAGS.valid_per_epoch is not None and (epoch_id - epoch_basis + 1) % FLAGS.valid_per_epoch == 0:
            save_path = train_saver.save(train_session,
                                         os.path.join('/tmp', '{:.3f}_{}.ckpt'.format(losses_v, epoch_id + 1)))
            # after one epoch end, do evaluation
            eval_loader.restore(eval_session, save_path)
            print("====================================Start of Eval==============================================")
            with eval_graph.as_default():
                eval_session.run(tf.local_variables_initializer())
            for batch_id in range(VALID_EPOCH_STEPS):
                tic = time()
                losses_v, accs_v, step, _ = \
                    eval_session.run([eval_metrics_gather_losses, eval_gather_accs,
                                      eval_step, eval_op])
                toc = time()
                print("\t EVALING {}/{} loss {:.4f} acc {:.4f} cost {:.3f}s".format(
                    batch_id+1, VALID_EPOCH_STEPS, losses_v, accs_v * 100, toc-tic
                ))
            print("EVALING RESULT loss {:.4f} acc {:.4f}".format(losses_v, accs_v * 100))
            print("====================================End of Eval==============================================")
            ckpt_name += 'V_L{:.4f}_A{:.4f}'.format(losses_v, accs_v)

        save_path = train_saver.save(train_session,
                                     os.path.join(FLAGS.next_ckpt, '{}.ckpt'.format(ckpt_name)))
        print("Model saved in path: %s" % save_path)


except tf.errors.OutOfRangeError:
    print('Done training')
