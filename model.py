# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
File containing the model function and parameterizations for the
Covid CVAE model.
"""
import tensorflow as tf
from anl_shared.cvae.tf.layers import build_model
from anl_shared.cvae.tf.optimizer import get_optimizer, wrap_optimizer
from anl_shared.cvae.tf.utils import log_hist
from tensorflow.keras.losses import Reduction
from tensorflow.python.training.experimental.loss_scale import FixedLossScale

from cerebras.tf.tf_helper import summary_layer
from modelzoo.common.tf.optimizers.GradAccumOptimizer import GradAccumOptimizer
from modelzoo.common.tf.optimizers.LossScale import CSDynamicLossScale

################################################################################
## MODEL FUNCTION DEFINITION
################################################################################
_REDUCTION_TYPES = ["sum", "mean"]


def model_fn(features, labels, mode, params):
    targets = labels
    loss = None
    train_op = None
    bce_loss_batch = None
    kl_loss_batch = None
    mparams = params["model"]
    oparams = params["optimizer"]
    variational = mparams["variational"]
    mixed_precision = mparams["mixed_precision"]
    fp_loss = mparams["full_precision_loss"]
    recon_loss_red_type = mparams["reconstruction_loss_reduction_type"]
    summary = False
    assert (
        recon_loss_red_type in _REDUCTION_TYPES
    ), f"invalid reconstruction loss reduction type: {recon_loss_red_type}"
    model_random_seed = mparams["model_random_seed"]
    loss_scale_type = oparams["loss_scale_type"]
    static_loss_scale = float(oparams["static_loss_scale"])
    initial_loss_scale = float(oparams["initial_loss_scale"])
    steps_per_increase = oparams["steps_per_increase"]

    tf.compat.v1.set_random_seed(model_random_seed)

    # training_hooks = []
    eval_metric_ops = {}
    log_metrics = False
    log_grads = False
    global_step = tf.compat.v1.train.get_global_step()

    if mixed_precision:
        policy = tf.keras.mixed_precision.experimental.Policy(
            'mixed_float16', loss_scale=None,
        )
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if summary:
        features = summary_layer(features)
    # Model output
    outputs, output_acts, kl_loss_batch = build_model(features, params)
    tf.compat.v1.logging.info(f"Model Outputs Shape: {outputs.get_shape()}")
    if mode != tf.estimator.ModeKeys.PREDICT:
        tf.compat.v1.logging.info(f"Targets Shape: {targets.get_shape()}")
    # Losses
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        # Cast outputs, targets to FP32 if full_precision_reconstruction_loss
        if fp_loss:
            targets = tf.cast(targets, tf.float32)
            outputs = tf.cast(outputs, tf.float32)
        if summary:
            targets = summary_layer(targets)
            outputs = summary_layer(outputs)

        # Variational loss
        if variational:
            assert kl_loss_batch is not None, "KL loss not found"
            # Average across batch
            kl_loss = tf.reduce_mean(kl_loss_batch)

            bce_loss_batch = tf.compat.v1.losses.sigmoid_cross_entropy(
                targets, outputs, loss_collection=None, reduction=Reduction.NONE
            )
            # Binary cross entropy
            if recon_loss_red_type == "sum":
                # Sum across elements
                bce_loss_batch = tf.reduce_sum(
                    input_tensor=bce_loss_batch, axis=1
                )
            else:
                # Average across elements
                bce_loss_batch = tf.reduce_mean(bce_loss_batch, axis=1)
            # Average across batch
            bce_loss = tf.reduce_mean(input_tensor=bce_loss_batch)

            # Add BCE + KL
            if summary:
                bce_loss = summary_layer(bce_loss)
                kl_loss = summary_layer(kl_loss)
            loss = tf.reduce_mean(bce_loss + kl_loss)
            if summary:
                loss = summary_layer(loss)

            if log_metrics:
                for name, tensor in [
                    ("variational_losses/bce", bce_loss),
                    ("variational_losses/kl", kl_loss),
                ]:
                    tf.summary.scalar(name, tensor)
                    eval_metric_ops[name] = tf.compat.v1.metrics.mean(tensor)

        # Deterministic model, squared error loss
        else:
            if recon_loss_red_type == "sum":
                sse = tf.reduce_sum(
                    input_tensor=tf.math.squared_difference(targets, outputs),
                    axis=1,
                )
                loss = tf.reduce_mean(input_tensor=sse)
            else:
                mse = tf.compat.v1.keras.losses.MeanSquaredError()
                loss = mse(y_true=targets, y_pred=outputs)
            if summary:
                loss = summary_layer(loss)

    # Optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Choose the right optimizer
        optimizer = get_optimizer(params=oparams)
        output_act_names = [v.name for v in output_acts]

        # Get the loss scale and wrap the optimizer
        if loss_scale_type == 'dynamic':
            loss_scale = CSDynamicLossScale(
                initial_loss_scale=initial_loss_scale,
                increment_period=steps_per_increase,
                multiplier=2.0,
            )
            loss_scale_value = loss_scale()
        else:
            loss_scale = FixedLossScale(1.0)
            loss_scale._loss_scale_value = static_loss_scale
            loss_scale_value = static_loss_scale
        optimizer = wrap_optimizer(optimizer, loss_scale)

        if log_metrics:
            tf.summary.scalar('loss_scale', loss_scale_value)

        # Maybe accumulate gradients
        grad_accum_steps = oparams["grad_accum_steps"]
        if grad_accum_steps > 1:
            optimizer = GradAccumOptimizer(
                optimizer, grad_accum_steps=grad_accum_steps
            )

        # Compute the unscaled gradients
        grads_vars = optimizer.compute_gradients(
            loss, tf.compat.v1.trainable_variables() + output_acts
        )
        trainable_grads_vars = [
            (g, v) for (g, v) in grads_vars if v.name not in output_act_names
        ]

        # And use them to minimize the loss
        train_op = optimizer.apply_gradients(
            trainable_grads_vars, global_step=global_step
        )

        if log_grads:
            # Log the scaled gradients and deltas
            def rescale(g):
                return g * tf.cast(loss_scale_value, g.dtype)

            for (g, v) in grads_vars:
                if "kernel" in v.name:
                    log_hist(rescale(g), v.name, f"kernel_grads")
                elif "bias" in v.name:
                    log_hist(rescale(g), v.name, f"bias_grads")
                elif v.name in output_act_names:
                    log_hist(rescale(g), v.name, f"output_grads")

            # Log the model weights
            for v in tf.compat.v1.trainable_variables():
                cast_v = tf.cast(v, tf.float16)
                if "kernel" in v.name:
                    log_hist(cast_v, v.name, f"kernel")
                elif "bias" in v.name:
                    log_hist(cast_v, v.name, f"bias")

    espec = tf.estimator.EstimatorSpec(
        mode=mode, predictions=outputs, loss=loss, train_op=train_op,
    )
    if mode == tf.estimator.ModeKeys.EVAL and (
        (bce_loss_batch != None) and (kl_loss_batch != None)
    ):
        espec.host_call = (
            build_eval_metric_ops,
            [bce_loss_batch, kl_loss_batch],
        )
    return espec


def build_eval_metric_ops(
    bce_loss_batch=None, kl_loss_batch=None,
):
    eval_metric_ops = {}
    if bce_loss_batch != None:
        # Average across batch
        bce_loss = tf.reduce_mean(bce_loss_batch)
        eval_metric_ops["variational_losses/bce"] = tf.compat.v1.metrics.mean(
            bce_loss
        )
        tf.summary.scalar("variational_losses/bce", bce_loss)
    if kl_loss_batch != None:
        # Average across batch
        kl_loss = tf.reduce_mean(kl_loss_batch)
        eval_metric_ops["variational_losses/kl"] = tf.compat.v1.metrics.mean(
            kl_loss
        )
        tf.summary.scalar("variational_losses/kl", kl_loss)
    return eval_metric_ops
