# author: dstamoulis
#
# This code extends codebase from the "MNasNet on TPU" GitHub repo:
# https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================
"""Search ConvNets with Single-Path NAS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

import imagenet_input
import supernet_macro
import nas_utils
from tensorflow.contrib.tpu.python.tpu import async_checkpoint
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator
from tensorflow.python.keras import backend as K
from tensorflow.python.client import device_lib


FLAGS = flags.FLAGS
FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'

flags.DEFINE_bool(
    'use_tpu', default=False,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
         'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

# Model specific flags
flags.DEFINE_string(
    'data_dir', default=FAKE_DATA_DIR,
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_string(
    'model_name',
    default='single-path-search',
    help=(
        'The model name to select models among existing MobileNet-like search spaces'
        'single-path-search: 3x3-3, 3x3-6, 5x5-3, 5x5-6.'
    ))

flags.DEFINE_string(
    'mode', default='train_and_eval',
    help='One of {"train_and_eval", "train", "eval"}.')

flags.DEFINE_integer(
    'train_steps', default=100008,
    help=('The number of steps to use for search. Default is 10008'
          ' with batch size 1024 and with warmup steps 6255'))

flags.DEFINE_integer(
    'input_image_size', default=224, help='Input image size.')

flags.DEFINE_integer(
    'train_batch_size', default=128, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=128, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'num_train_images', default=1281167, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=50000, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'steps_per_eval', default=12000,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_bool(
    'skip_host_call', default=False,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

flags.DEFINE_integer(
    'iterations_per_loop', default=1251,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'num_parallel_calls', default=64,
    help=('Number of parallel threads in CPU for the input pipeline'))

flags.DEFINE_string(
    'bigtable_project', None,
    'The Cloud Bigtable project.  If None, --gcp_project will be used.')
flags.DEFINE_string(
    'bigtable_instance', None,
    'The Cloud Bigtable instance to load data from.')
flags.DEFINE_string(
    'bigtable_table', 'imagenet',
    'The Cloud Bigtable table to load data from.')
flags.DEFINE_string(
    'bigtable_train_prefix', 'train_',
    'The prefix identifying training rows.')
flags.DEFINE_string(
    'bigtable_eval_prefix', 'validation_',
    'The prefix identifying evaluation rows.')
flags.DEFINE_string(
    'bigtable_column_family', 'tfexample',
    'The column family storing TFExamples.')
flags.DEFINE_string(
    'bigtable_column_qualifier', 'example',
    'The column name storing TFExamples.')

flags.DEFINE_string(
    'data_format', default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))
flags.DEFINE_integer(
    'num_label_classes', default=1000, help='Number of classes, at least 2')
flags.DEFINE_float(
    'batch_norm_momentum',
    default=None,
    help=('Batch normalization layer momentum of moving average to override.'))
flags.DEFINE_float(
    'batch_norm_epsilon',
    default=None,
    help=('Batch normalization layer epsilon to override..'))

flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))
flags.DEFINE_bool(
    'export_to_tpu', default=False,
    help=('Whether to export additional metagraph with "serve, tpu" tags'
          ' in addition to "serve" only metagraph.'))
flags.DEFINE_bool(
    'post_quantize', default=True, help=('Enable post quantization.'))

flags.DEFINE_float(
    'base_learning_rate',
    default=0.016,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'moving_average_decay', default=0,
    help=('Moving average decay rate.'))

flags.DEFINE_float(
    'weight_decay', default=1e-5,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_float(
    'dropout_rate', default=0.2,
    help=('Dropout rate for the final output layer.'))

flags.DEFINE_integer('log_step_count_steps', 64, 'The number of steps at '
                                                 'which the global step information is logged.')

flags.DEFINE_bool(
    'use_cache', default=True, help=('Enable cache for training input.'))

flags.DEFINE_float(
    'depth_multiplier', default=None, help=('Depth multiplier per layer.'))

flags.DEFINE_float(
    'depth_divisor', default=None, help=('Depth divisor (default to 8).'))

flags.DEFINE_float(
    'min_depth', default=None, help=('Minimal depth (default to None).'))

flags.DEFINE_bool(
    'use_async_checkpointing', default=False, help=('Enable async checkpoint'))

flags.DEFINE_float(
    'runtime_lambda_val', default=0.1,
    help=('Lambda val for trading off loss and runtime'))

# Learning rate schedule
LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def nas_model_fn(features, labels, mode, params):
    """The model_fn for TPUEstimator-based NAS search

    Args:
      features: `Tensor` of batched images.
      labels: `Tensor` of labels for the data samples
      mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
      params: `dict` of parameters passed to the model from the TPUEstimator,
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `TPUEstimatorSpec` for the model
    """
    if isinstance(features, dict):
        features = features['feature']

    # In most cases, the default data format NCHW instead of NHWC should be
    # used for a significant performance boost on GPU/TPU. NHWC should be used
    # only if the network needs to be run on CPU since the pooling operations
    # are only supported on NHWC.
    if FLAGS.data_format == 'channels_first':
        assert not FLAGS.transpose_input  # channels_first only for GPU
        features = tf.transpose(features, [0, 3, 1, 2])

    if FLAGS.transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
        features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

    # Normalize the image to zero mean and unit variance.
    features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
    features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    has_moving_average_decay = (FLAGS.moving_average_decay > 0)
    # This is essential, if using a keras-derived model.
    K.set_learning_phase(is_training)
    tf.logging.info('Using open-source implementation for NAS definition.')
    override_params = {}
    if FLAGS.batch_norm_momentum:
        override_params['batch_norm_momentum'] = FLAGS.batch_norm_momentum
    if FLAGS.batch_norm_epsilon:
        override_params['batch_norm_epsilon'] = FLAGS.batch_norm_epsilon
    if FLAGS.dropout_rate:
        override_params['dropout_rate'] = FLAGS.dropout_rate
    if FLAGS.data_format:
        override_params['data_format'] = FLAGS.data_format
    if FLAGS.num_label_classes:
        override_params['num_classes'] = FLAGS.num_label_classes
    if FLAGS.depth_multiplier:
        override_params['depth_multiplier'] = FLAGS.depth_multiplier
    if FLAGS.depth_divisor:
        override_params['depth_divisor'] = FLAGS.depth_divisor
    if FLAGS.min_depth:
        override_params['min_depth'] = FLAGS.min_depth

    global_step = tf.train.get_global_step()
    warmup_steps = 6255
    dropout_rate = nas_utils.build_dropout_rate(global_step, warmup_steps)

    logits, runtime_val, indicators = supernet_macro.build_supernet(
        features,
        model_name=FLAGS.model_name,
        training=is_training,
        override_params=override_params,
        dropout_rate=dropout_rate)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # If necessary, in the model_fn, use params['batch_size'] instead the batch
    # size flags (--train_batch_size or --eval_batch_size).
    # batch_size = params['batch_size']   # pylint: disable=unused-variable

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    one_hot_labels = tf.one_hot(labels, FLAGS.num_label_classes)
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=one_hot_labels,
        label_smoothing=FLAGS.label_smoothing)

    runtime_lambda = nas_utils.build_runtime_lambda(global_step,
                                                    warmup_steps, FLAGS.runtime_lambda_val)
    runtime_loss = runtime_lambda * 1e3 * tf.log(runtime_val)  # 1e3 to sec

    # dstamoulis NOTE: No reshaping led to crashing:
    # ValueError: Cannot reshape a tensor with 2 elements to shape [1]
    # (1 elements) for 'Reshape_1' (op: 'Reshape') with input shapes: [2], [1]
    # and with input tensors computed as partial shapes: input[1] = [1].  #
    # --> solution: matching size, so that trainin_loop while does not complain
    runtime_loss = tf.reshape(runtime_loss, shape=cross_entropy.shape)

    # Add weight decay to the loss for non-batch-normalization variables.
    loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name]) + runtime_loss

    if has_moving_average_decay:
        ema = tf.train.ExponentialMovingAverage(
            decay=FLAGS.moving_average_decay, num_updates=global_step)
        ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
        for v in tf.global_variables():
            # We maintain mva for batch norm moving mean and variance as well.
            if 'moving_mean' in v.name or 'moving_variance' in v.name:
                ema_vars.append(v)
        ema_vars = list(set(ema_vars))

    host_call = None
    restore_vars_dict = None
    if is_training:
        # Compute the current epoch and associated learning rate from global_step.
        current_epoch = (
                tf.cast(global_step, tf.float32) / params['steps_per_epoch'])

        scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)

        # NOTE: dstamoulis -- cancelled out warm-up epochs with -1!!
        learning_rate = nas_utils.build_learning_rate(scaled_lr, global_step,
                                                      params['steps_per_epoch'],
                                                      warmup_epochs=-1)
        optimizer = nas_utils.build_optimizer(learning_rate)
        if FLAGS.use_tpu:
            # When using TPU, wrap the optimizer with CrossShardOptimizer which
            # handles synchronization details between different TPU cores. To the
            # user, this should look like regular synchronous training.
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        # Batch normalization requires UPDATE_OPS to be added as a dependency to
        # the train operation.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)

        if has_moving_average_decay:
            with tf.control_dependencies([train_op]):
                train_op = ema.apply(ema_vars)

        if not FLAGS.skip_host_call:
            def host_call_fn(gs, loss, lr, runtime,
                             t5x5_1, t50c_1, t100c_1, t5x5_2, t50c_2, t100c_2,
                             t5x5_3, t50c_3, t100c_3, t5x5_4, t50c_4, t100c_4,
                             t5x5_5, t50c_5, t100c_5, t5x5_6, t50c_6, t100c_6,
                             t5x5_7, t50c_7, t100c_7, t5x5_8, t50c_8, t100c_8,
                             t5x5_9, t50c_9, t100c_9, t5x5_10, t50c_10, t100c_10,
                             t5x5_11, t50c_11, t100c_11, t5x5_12, t50c_12, t100c_12,
                             t5x5_13, t50c_13, t100c_13, t5x5_14, t50c_14, t100c_14,
                             t5x5_15, t50c_15, t100c_15, t5x5_16, t50c_16, t100c_16,
                             t5x5_17, t50c_17, t100c_17, t5x5_18, t50c_18, t100c_18,
                             t5x5_19, t50c_19, t100c_19, t5x5_20, t50c_20, t100c_20):
                """Training host call. Creates scalar summaries for training metrics.

                This function is executed on the CPU and should not directly reference
                any Tensors in the rest of the `model_fn`. To pass Tensors from the
                model to the `metric_fn`, provide as part of the `host_call`. See
                https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
                for more information.

                Arguments should match the list of `Tensor` objects passed as the second
                element in the tuple passed to `host_call`.

                Args:
                  gs: `Tensor with shape `[batch]` for the global_step
                  loss: `Tensor` with shape `[batch]` for the training loss.
                  lr: `Tensor` with shape `[batch]` for the learning_rate.
                  ce: `Tensor` with shape `[batch]` for the current_epoch.

                Returns:
                  List of summary ops to run on the CPU host.
                """
                gs = gs[0]

                t_list = [[t5x5_1, t50c_1, t100c_1], [t5x5_2, t50c_2, t100c_2],
                          [t5x5_3, t50c_3, t100c_3], [t5x5_4, t50c_4, t100c_4],
                          [t5x5_5, t50c_5, t100c_5], [t5x5_6, t50c_6, t100c_6],
                          [t5x5_7, t50c_7, t100c_7], [t5x5_8, t50c_8, t100c_8],
                          [t5x5_9, t50c_9, t100c_9], [t5x5_10, t50c_10, t100c_10],
                          [t5x5_11, t50c_11, t100c_11], [t5x5_12, t50c_12, t100c_12],
                          [t5x5_13, t50c_13, t100c_13], [t5x5_14, t50c_14, t100c_14],
                          [t5x5_15, t50c_15, t100c_15], [t5x5_16, t50c_16, t100c_16],
                          [t5x5_17, t50c_17, t100c_17], [t5x5_18, t50c_18, t100c_18],
                          [t5x5_19, t50c_19, t100c_19], [t5x5_20, t50c_20, t100c_20]]

                # Host call fns are executed FLAGS.iterations_per_loop times after one
                # TPU loop is finished, setting max_queue value to the same as number of
                # iterations will make the summary writer only flush the data to storage
                # once per loop.
                with tf.contrib.summary.create_file_writer(
                        FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop).as_default():
                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar('loss', loss[0], step=gs)
                        tf.contrib.summary.scalar('learning_rate', lr[0], step=gs)
                        # tf.contrib.summary.scalar('current_epoch', ce[0], step=gs)
                        tf.contrib.summary.scalar('runtime_ms', runtime[0], step=gs)
                        for idx, t_ in enumerate(t_list):
                            for label_, t_tensor in zip(['t5x5_', 't50c_', 't100c_'], t_):
                                sum_label_ = label_ + str(idx + 1)
                                tf.contrib.summary.scalar(sum_label_, t_tensor[0], step=gs)

                        return tf.contrib.summary.all_summary_ops()


            # To log the loss, current learning rate, and epoch for Tensorboard, the
            # summary op needs to be run on the host CPU via host_call. host_call
            # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
            # dimension. These Tensors are implicitly concatenated to [params['batch_size']].
            gs_t = tf.reshape(global_step, [1])
            loss_t = tf.reshape(cross_entropy, [1])
            lr_t = tf.reshape(learning_rate, [1])
            runtime_t = tf.reshape(runtime_val, [1])

            # Single-Path additions: get the threshold decisions per design space
            t_list = []
            decision_labels = ['d5x5', 'd50c', 'd100c']
            t_list = []
            for idx in range(20):
                key_ = 'block_' + str(idx + 1)
                for decision_label in decision_labels:
                    v = indicators[key_][decision_label]
                    t_list.append(tf.reshape(v, [1]))

            host_call = (host_call_fn, [gs_t, loss_t, lr_t, runtime_t] + t_list)

    else:
        train_op = None
        if has_moving_average_decay:
            # Load moving average variables for eval.
            restore_vars_dict = ema.variables_to_restore(ema_vars)

    metric_fn = None
    if mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(labels, logits):
            """Evaluation metric function. Evaluates accuracy.

            This function is executed on the CPU and should not directly reference
            any Tensors in the rest of the `model_fn`. To pass Tensors from the model
            to the `metric_fn`, provide as part of the `eval_metrics`. See
            https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
            for more information.

            Arguments should match the list of `Tensor` objects passed as the second
            element in the tuple passed to `eval_metrics`.

            Args:
              labels: `Tensor` with shape `[batch]`.
              logits: `Tensor` with shape `[batch, num_classes]`.

            Returns:
              A dict of the metrics to return from evaluation.
            """
            predictions = tf.argmax(logits, axis=1)
            top_1_accuracy = tf.metrics.accuracy(labels, predictions)
            in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
            top_5_accuracy = tf.metrics.mean(in_top_5)

            return {
                'top_1_accuracy': top_1_accuracy,
                'top_5_accuracy': top_5_accuracy,
            }


        eval_metrics = (metric_fn, [labels, logits])

    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('number of trainable parameters: {}'.format(num_params))


    def _scaffold_fn():
        saver = tf.train.Saver(restore_vars_dict)
        return tf.train.Scaffold(saver=saver)


    # return tf.contrib.tpu.TPUEstimatorSpec(
    #     mode=mode,
    #     loss=loss,
    #     train_op=train_op,
    #     host_call=host_call,
    #     eval_metrics=eval_metrics,
    #     scaffold_fn=_scaffold_fn if has_moving_average_decay else None)

    if metric_fn is None:
        eval_metric_ops = None
    else:
        eval_metric_ops = metric_fn(labels, logits)

    if is_training:
        gs, loss, lr, runtime,t5x5_1, t50c_1, t100c_1, t5x5_2, t50c_2, t100c_2,        t5x5_3, t50c_3, t100c_3, t5x5_4, t50c_4, t100c_4,        t5x5_5, t50c_5, t100c_5, t5x5_6, t50c_6, t100c_6,        t5x5_7, t50c_7, t100c_7, t5x5_8, t50c_8, t100c_8,        t5x5_9, t50c_9, t100c_9, t5x5_10, t50c_10, t100c_10,        t5x5_11, t50c_11, t100c_11, t5x5_12, t50c_12, t100c_12,        t5x5_13, t50c_13, t100c_13, t5x5_14, t50c_14, t100c_14,        t5x5_15, t50c_15, t100c_15, t5x5_16, t50c_16, t100c_16,        t5x5_17, t50c_17, t100c_17, t5x5_18, t50c_18, t100c_18,        t5x5_19, t50c_19, t100c_19, t5x5_20, t50c_20, t100c_20 = gs_t, loss_t, lr_t, runtime_t, t_list[0], t_list[1], t_list[2], t_list[3], t_list[4], t_list[5],                                  t_list[6], t_list[7], t_list[8], t_list[9], t_list[10], t_list[11], t_list[12], t_list[13], t_list[14],                                  t_list[15], t_list[16], t_list[17], t_list[18], t_list[19], t_list[20], t_list[21], t_list[22],                                  t_list[23], t_list[24], t_list[25], t_list[26], t_list[27], t_list[28], t_list[29], t_list[30],                                  t_list[31], t_list[32], t_list[33], t_list[34], t_list[35], t_list[36], t_list[37], t_list[38],                                  t_list[39], t_list[40], t_list[41], t_list[42], t_list[43], t_list[44], t_list[45], t_list[46],                                  t_list[47], t_list[48], t_list[49], t_list[50], t_list[51], t_list[52], t_list[53], t_list[54],                                  t_list[55], t_list[56], t_list[57], t_list[58], t_list[59]

        gs = gs[0]

        t_list = [[t5x5_1, t50c_1, t100c_1], [t5x5_2, t50c_2, t100c_2],
                  [t5x5_3, t50c_3, t100c_3], [t5x5_4, t50c_4, t100c_4],
                  [t5x5_5, t50c_5, t100c_5], [t5x5_6, t50c_6, t100c_6],
                  [t5x5_7, t50c_7, t100c_7], [t5x5_8, t50c_8, t100c_8],
                  [t5x5_9, t50c_9, t100c_9], [t5x5_10, t50c_10, t100c_10],
                  [t5x5_11, t50c_11, t100c_11], [t5x5_12, t50c_12, t100c_12],
                  [t5x5_13, t50c_13, t100c_13], [t5x5_14, t50c_14, t100c_14],
                  [t5x5_15, t50c_15, t100c_15], [t5x5_16, t50c_16, t100c_16],
                  [t5x5_17, t50c_17, t100c_17], [t5x5_18, t50c_18, t100c_18],
                  [t5x5_19, t50c_19, t100c_19], [t5x5_20, t50c_20, t100c_20]]

        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        # with tf.contrib.summary.create_file_writer(
        #         '/u/big/home/fabbrim/log/single-path-nas/lambda-val-0.020-chinook-2/events', max_queue=FLAGS.iterations_per_loop).as_default():
        #     with tf.contrib.summary.always_record_summaries():
        tf.summary.scalar('loss', loss[0])
        tf.summary.scalar('learning_rate', lr[0])
        # tf.contrib.summary.scalar('current_epoch', ce[0], step=gs)
        tf.summary.scalar('runtime_ms', runtime[0])
        for idx, t_ in enumerate(t_list):
            for label_, t_tensor in zip(['t5x5_', 't50c_', 't100c_'], t_):
                sum_label_ = label_ + str(idx + 1)
                tf.summary.scalar(sum_label_, t_tensor[0])

        # tf.contrib.summary.all_summary_ops()

    # writer = tf.contrib.summary.create_file_writer(FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop)
    # writer.set_as_default()

    # summary_op = host_call_fn(gs_t, loss_t, lr_t, runtime_t, t_list[0], t_list[1], t_list[2], t_list[3], t_list[4], t_list[5],
    #                               t_list[6], t_list[7], t_list[8], t_list[9], t_list[10], t_list[11], t_list[12], t_list[13], t_list[14],
    #                               t_list[15], t_list[16], t_list[17], t_list[18], t_list[19], t_list[20], t_list[21], t_list[22],
    #                               t_list[23], t_list[24], t_list[25], t_list[26], t_list[27], t_list[28], t_list[29], t_list[30],
    #                               t_list[31], t_list[32], t_list[33], t_list[34], t_list[35], t_list[36], t_list[37], t_list[38],
    #                               t_list[39], t_list[40], t_list[41], t_list[42], t_list[43], t_list[44], t_list[45], t_list[46],
    #                               t_list[47], t_list[48], t_list[49], t_list[50], t_list[51], t_list[52], t_list[53], t_list[54],
    #                               t_list[55], t_list[56], t_list[57], t_list[58], t_list[59])
    #
    # summary_hook = tf.train.SummarySaverHook(
    #     FLAGS.iterations_per_loop,
    #     output_dir=FLAGS.model_dir,
    #     summary_op=summary_op,
    # )
    # train_hook_list = None
    # if is_training:
    #     train_hook_list = []
    #     train_tensors_log = {'accuracy': loss_t[0],
    #                          'lossetta': loss_t[0],
    #                          'zio': loss_t[0]}
    #     train_hook_list.append(tf.train.LoggingTensorHook(
    #         tensors=train_tensors_log, every_n_iter=5))

    # if is_training:
    #     tf.summary.scalar("dai", loss_t[0])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        # host_call=host_call,
        # training_chief_hooks=[summary_hook],
        eval_metric_ops=eval_metric_ops,
        # scaffold_fn=_scaffold_fn if has_moving_average_decay else None
    )


def _verify_non_empty_string(value, field_name):
    """Ensures that a given proposed field value is a non-empty string.

    Args:
      value:  proposed value for the field.
      field_name:  string name of the field, e.g. `project`.

    Returns:
      The given value, provided that it passed the checks.

    Raises:
      ValueError:  the value is not a string, or is a blank string.
    """
    if not isinstance(value, str):
        raise ValueError(
            'Bigtable parameter "%s" must be a string.' % field_name)
    if not value:
        raise ValueError(
            'Bigtable parameter "%s" must be non-empty.' % field_name)
    return value


def _select_tables_from_flags():
    """Construct training and evaluation Bigtable selections from flags.

    Returns:
      [training_selection, evaluation_selection]
    """
    project = _verify_non_empty_string(
        FLAGS.bigtable_project or FLAGS.gcp_project,
        'project')
    instance = _verify_non_empty_string(FLAGS.bigtable_instance, 'instance')
    table = _verify_non_empty_string(FLAGS.bigtable_table, 'table')
    train_prefix = _verify_non_empty_string(FLAGS.bigtable_train_prefix,
                                            'train_prefix')
    eval_prefix = _verify_non_empty_string(FLAGS.bigtable_eval_prefix,
                                           'eval_prefix')
    column_family = _verify_non_empty_string(FLAGS.bigtable_column_family,
                                             'column_family')
    column_qualifier = _verify_non_empty_string(FLAGS.bigtable_column_qualifier,
                                                'column_qualifier')
    return [
        imagenet_input.BigtableSelection(
            project=project,
            instance=instance,
            table=table,
            prefix=p,
            column_family=column_family,
            column_qualifier=column_qualifier)
        for p in (train_prefix, eval_prefix)
    ]


def export(est, export_dir, post_quantize=True):
    """Export graph to SavedModel and TensorFlow Lite.

    Args:
      est: estimator instance.
      export_dir: string, exporting directory.
      post_quantize: boolean, whether to quantize model checkpoint after training.

    Raises:
      ValueError: the export directory path is not specified.
    """
    if not export_dir:
        raise ValueError('The export directory path is not specified.')


    # The guide to serve a exported TensorFlow model is at:
    #    https://www.tensorflow.org/serving/serving_basic
    def lite_image_serving_input_fn():
        """serving input fn for raw images."""
        input_shape = [1, FLAGS.input_image_size, FLAGS.input_image_size, 3]
        images = tf.placeholder(shape=input_shape, dtype=tf.float32)
        return tf.estimator.export.ServingInputReceiver(images, {'images': images})


    tf.logging.info('Starting to export model.')
    est.export_saved_model(
        export_dir_base=export_dir,
        serving_input_receiver_fn=lite_image_serving_input_fn)

    subfolder = sorted(tf.gfile.ListDirectory(export_dir), reverse=True)[0]
    tf.logging.info('Starting to export TFLite.')
    converter = tf.lite.TFLiteConverter.from_saved_model(
        os.path.join(export_dir, subfolder),
        input_arrays=['truediv'],
        output_arrays=['logits'])
    tflite_model = converter.convert()
    tflite_file = os.path.join(export_dir, FLAGS.model_name + '.tflite')
    tf.gfile.GFile(tflite_file, 'wb').write(tflite_model)

    if post_quantize:
        tf.logging.info('Starting to export quantized TFLite.')
        converter = tf.lite.TFLiteConverter.from_saved_model(
            os.path.join(export_dir, subfolder),
            input_arrays=['truediv'],
            output_arrays=['logits'])
        converter.post_training_quantize = True
        quant_tflite_model = converter.convert()
        quant_tflite_file = os.path.join(export_dir,
                                         FLAGS.model_name + '_postquant.tflite')
        tf.gfile.GFile(quant_tflite_file, 'wb').write(quant_tflite_model)


def main(unused_argv):
    # tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    #     FLAGS.tpu if (FLAGS.tpu or FLAGS.use_tpu) else '',
    #     zone=FLAGS.tpu_zone,
    #     project=FLAGS.gcp_project)

    if FLAGS.use_async_checkpointing:
        save_checkpoints_steps = None
    else:
        save_checkpoints_steps = max(100, FLAGS.iterations_per_loop)

    NUM_GPUS = len(get_available_gpus())
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
    gpu_options = tf.GPUOptions(allow_growth=True)

    # config = tf.contrib.tpu.RunConfig(
    #     # cluster=tpu_cluster_resolver,
    #     model_dir=FLAGS.model_dir,
    #     save_checkpoints_steps=save_checkpoints_steps,
    #     log_step_count_steps=FLAGS.log_step_count_steps,
    #     session_config=tf.ConfigProto(
    #         graph_options=tf.GraphOptions(
    #             rewrite_options=rewriter_config_pb2.RewriterConfig(
    #                 disable_meta_optimizer=True)), gpu_options=gpu_options),
    #     train_distribute=distribution,
    #     # tpu_config=tf.contrib.tpu.TPUConfig(
    #     #     iterations_per_loop=FLAGS.iterations_per_loop,
    #     #     per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
    #     #     .PER_HOST_V2)
    # )  # pylint: disable=line-too-long
    config = tf.estimator.RunConfig(
        # cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        log_step_count_steps=FLAGS.log_step_count_steps,
        session_config=tf.ConfigProto(allow_soft_placement=True,
            graph_options=tf.GraphOptions(
                rewrite_options=rewriter_config_pb2.RewriterConfig(
                    disable_meta_optimizer=True)), gpu_options=gpu_options),
        train_distribute=distribution,
        # log_step_count_steps=None,
        # save_summary_steps=None
        # tpu_config=tf.contrib.tpu.TPUConfig(
        #     iterations_per_loop=FLAGS.iterations_per_loop,
        #     per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
        #     .PER_HOST_V2)
    )  # pylint: disable=line-too-long

    # Initializes model parameters.
    # params = dict(steps_per_epoch=FLAGS.num_train_images / FLAGS.train_batch_size)
    # nas_est = tf.contrib.tpu.TPUEstimator(
    #     use_tpu=FLAGS.use_tpu,
    #     model_fn=nas_model_fn,
    #     config=config,
    #     train_batch_size=FLAGS.train_batch_size,
    #     eval_batch_size=FLAGS.eval_batch_size,
    #     export_to_tpu=FLAGS.export_to_tpu,
    #     params=params)
    params = dict(steps_per_epoch=FLAGS.num_train_images / FLAGS.train_batch_size, batch_size=FLAGS.train_batch_size)
    nas_est = tf.estimator.Estimator(
        model_fn=nas_model_fn,
        config=config,
        params=params)

    # Input pipelines are slightly different (with regards to shuffling and
    # preprocessing) between training and evaluation.
    if FLAGS.bigtable_instance:
        tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
        select_train, select_eval = _select_tables_from_flags()
        imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
            is_training=is_training,
            use_bfloat16=False,
            transpose_input=FLAGS.transpose_input,
            selection=selection) for (is_training, selection) in
            [(True, select_train),
             (False, select_eval)]]
    else:
        if FLAGS.data_dir == FAKE_DATA_DIR:
            tf.logging.info('Using fake dataset.')
        else:
            tf.logging.info('Using dataset: %s', FLAGS.data_dir)
        imagenet_train, imagenet_eval = [
            imagenet_input.ImageNetInput(
                is_training=is_training,
                data_dir=FLAGS.data_dir,
                transpose_input=FLAGS.transpose_input,
                cache=FLAGS.use_cache and is_training,
                image_size=FLAGS.input_image_size,
                num_parallel_calls=FLAGS.num_parallel_calls,
                use_bfloat16=False) for is_training in [True, False]
        ]

    if FLAGS.mode == 'eval':
        eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size
        # Run evaluation when there's a new checkpoint
        for ckpt in evaluation.checkpoints_iterator(
                FLAGS.model_dir, timeout=FLAGS.eval_timeout):
            tf.logging.info('Starting to evaluate.')
            try:
                start_timestamp = time.time()  # This time will include compilation time
                eval_results = nas_est.evaluate(
                    input_fn=imagenet_eval.input_fn,
                    steps=eval_steps,
                    checkpoint_path=ckpt)
                elapsed_time = int(time.time() - start_timestamp)
                tf.logging.info('Eval results: %s. Elapsed seconds: %d',
                                eval_results, elapsed_time)

                # Terminate eval job when final checkpoint is reached
                current_step = int(os.path.basename(ckpt).split('-')[1])
                if current_step >= FLAGS.train_steps:
                    tf.logging.info(
                        'Evaluation finished after training step %d', current_step)
                    break

            except tf.errors.NotFoundError:
                # Since the coordinator is on a different job than the TPU worker,
                # sometimes the TPU worker does not finish initializing until long after
                # the CPU job tells it to start evaluating. In this case, the checkpoint
                # file could have been deleted already.
                tf.logging.info(
                    'Checkpoint %s no longer exists, skipping checkpoint', ckpt)

        if FLAGS.export_dir:
            export(nas_est, FLAGS.export_dir, FLAGS.post_quantize)
    else:  # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
        current_step = estimator._load_global_step_from_checkpoint_dir(
            FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long

        tf.logging.info(
            'Training for %d steps (%.2f epochs in total). Current'
            ' step %d.', FLAGS.train_steps,
            FLAGS.train_steps / params['steps_per_epoch'], current_step)

        start_timestamp = time.time()  # This time will include compilation time

        if FLAGS.mode == 'train':
            hooks = []
            if FLAGS.use_async_checkpointing:
                hooks.append(
                    async_checkpoint.AsyncCheckpointSaverHook(
                        checkpoint_dir=FLAGS.model_dir,
                        save_steps=max(100, FLAGS.iterations_per_loop)))
            nas_est.train(
                input_fn=imagenet_train.input_fn,
                max_steps=FLAGS.train_steps,
                hooks=hooks)

        else:
            assert FLAGS.mode == 'train_and_eval'
            while current_step < FLAGS.train_steps:
                # Train for up to steps_per_eval number of steps.
                # At the end of training, a checkpoint will be written to --model_dir.
                next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                                      FLAGS.train_steps)
                nas_est.train(
                    input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
                current_step = next_checkpoint

                tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                                next_checkpoint, int(time.time() - start_timestamp))

                # Evaluate the model on the most recent model in --model_dir.
                # Since evaluation happens in batches of --eval_batch_size, some images
                # may be excluded modulo the batch size. As long as the batch size is
                # consistent, the evaluated images are also consistent.
                tf.logging.info('Starting to evaluate.')
                eval_results = nas_est.evaluate(
                    input_fn=imagenet_eval.input_fn,
                    steps=FLAGS.num_eval_images // FLAGS.eval_batch_size)
                tf.logging.info('Eval results at step %d: %s',
                                next_checkpoint, eval_results)

            elapsed_time = int(time.time() - start_timestamp)
            tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                            FLAGS.train_steps, elapsed_time)
            if FLAGS.export_dir:
                export(nas_est, FLAGS.export_dir, FLAGS.post_quantize)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
