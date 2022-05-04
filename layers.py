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

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from anl_shared.cvae.tf.utils import make_sparsity_summary

from cerebras.tf.tf_helper import summary_layer

_REDUCTION_TYPES = ["sum", "mean"]


def conv2d(
    inputs,
    filters,
    kernel_size,
    padding,
    dilation_rate=(1, 1),
    strides=(1, 1),
    activation="relu",
    name="conv",
    use_bias=True,
    log_sparsity=False,
    output_acts=[],
    summary=False,
):
    if strides != (1, 1) and dilation_rate != (1, 1):
        raise ValueError(
            f"both strides and dilation should not be specified as greater than 1. \
            strides set to {strides} and dilation_rate set to {dilation_rate}"
        )

    net = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        data_format="channels_first",
        name=name,
        use_bias=use_bias,
    )(inputs)

    if summary:
        net = summary_layer(net)
    if activation is not None:
        # Moving activation as standalone layer to match with Cerebras.
        net = tf.keras.layers.Activation(
            activation, name=f"{name}_{activation}"
        )(net)
        if summary:
            net = summary_layer(net)

    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")
    output_acts.append(net)
    if log_sparsity:
        make_sparsity_summary(name, inputs)

    return net, output_acts


def deconv2d(
    inputs,
    filters,
    kernel_size,
    padding,
    dilation_rate=(1, 1),
    strides=(1, 1),
    activation="relu",
    name="deconv",
    use_bias=True,
    log_sparsity=False,
    output_acts=[],
    summary=False,
):
    if strides != (1, 1) and dilation_rate != (1, 1):
        raise ValueError(
            f"both strides and dilation should not be specified as greater than 1. \
            strides set to {strides} and dilation_rate set to {dilation_rate}"
        )

    net = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        data_format="channels_first",
        name=name,
        use_bias=use_bias,
    )(inputs)

    if summary:
        net = summary_layer(net)
    if activation is not None:
        # Moving activation as standalone layer to match with Cerebras.
        net = tf.keras.layers.Activation(
            activation, name=f"{name}_{activation}"
        )(net)
        if summary:
            net = summary_layer(net)

    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")
    output_acts.append(net)
    if log_sparsity:
        make_sparsity_summary(name, inputs)

    return net, output_acts


def dense(
    inputs,
    units,
    activation,
    name="dense",
    use_bias=True,
    log_sparsity=False,
    output_acts=[],
    summary=False,
):
    net = tf.keras.layers.Dense(name=name, units=units, use_bias=use_bias,)(
        inputs
    )

    if summary:
        net = summary_layer(net)
    if activation is not None:
        # Moving activation as standalone layer to match with Cerebras.
        net = tf.keras.layers.Activation(
            activation, name=f"{name}_{activation}"
        )(net)
        if summary:
            net = summary_layer(net)

    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")
    output_acts.append(net)
    if log_sparsity:
        make_sparsity_summary(name, inputs)

    return net, output_acts


def deterministic_embedding(
    inputs,
    latent_ndim,
    name="embedding",
    log_sparsity=False,
    output_acts=[],
    summary=False,
):
    mean, output_acts = dense(
        inputs=inputs,
        units=latent_ndim,
        activation=None,
        name="enc_dense_mean",
        log_sparsity=log_sparsity,
        output_acts=output_acts,
    )

    net = mean
    net = tf.identity(net, name=f"{name}")

    if summary:
        net = summary_layer(net)

    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")
    output_acts.append(net)
    if log_sparsity:
        make_sparsity_summary(f"{name}_mean", mean)

    return net, output_acts


def variational_embedding(
    inputs,
    latent_ndim,
    kl_loss_reduction_type="sum",
    fp_loss=False,
    name="embedding",
    log_sparsity=False,
    output_acts=[],
    summary=False,
    identity_normal_dist=False,
):
    mean, output_acts = dense(
        inputs=inputs,
        units=latent_ndim,
        activation=None,
        name="enc_dense_mean",
        log_sparsity=log_sparsity,
        output_acts=output_acts,
        summary=summary,
    )
    # if summary:
    #    mean = summary_layer(mean)

    logvar, output_acts = dense(
        inputs=inputs,
        units=latent_ndim,
        activation=None,
        name="enc_dense_logvar",
        log_sparsity=log_sparsity,
        output_acts=output_acts,
        summary=summary,
    )
    # if summary:
    #    logvar = summary_layer(logvar)

    # Need this for debugging only.
    if identity_normal_dist:
        eps = tf.ones_like(mean, dtype=mean.dtype, name="epsilon")
    else:
        # Sample embedding
        eps = tf.random.normal(
            shape=mean.shape, dtype=mean.dtype, name="epsilon"
        )
    if summary:
        eps = summary_layer(eps)

    net = mean + K.exp(0.5 * logvar) * eps
    net = tf.identity(net, name=f"{name}")
    if summary:
        net = summary_layer(net)

    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")
    output_acts.append(net)
    if log_sparsity:
        make_sparsity_summary(f"{name}_mean", mean)
        make_sparsity_summary(f"{name}_logvar", logvar)

    # KL loss
    if kl_loss_reduction_type == "sum":
        reduce_op = tf.reduce_sum
    else:
        reduce_op = tf.reduce_mean
    if fp_loss:
        mean = tf.cast(mean, tf.float32)
        logvar = tf.cast(logvar, tf.float32)
    # Reduce along latent_ndim
    kl_loss = -0.5 * reduce_op(
        (1 + logvar - K.square(mean) - K.exp(logvar)), axis=1
    )

    return net, output_acts, kl_loss


def build_model(net, params):
    iparams = params["train_input"]
    mparams = params["model"]
    input_shape = iparams["input_shape"]
    assert len(input_shape) == 3, "Input shape must be 3-dim"

    enc_conv_kernels = mparams["enc_conv_kernels"]
    enc_conv_filters = mparams["enc_conv_filters"]  # output filters
    enc_conv_strides = mparams["enc_conv_strides"]
    activation = mparams["activation"]
    assert len(enc_conv_kernels) == len(
        enc_conv_filters
    ), "encoder layers are misspecified: len(kernels) != len(filters)"
    assert len(enc_conv_kernels) == len(
        enc_conv_strides
    ), "encoder layers are misspecified: len(kernels) != len(strides)"

    dec_conv_kernels = mparams["dec_conv_kernels"]
    dec_conv_filters = mparams["dec_conv_filters"]  # input filters
    dec_conv_strides = mparams["dec_conv_strides"]
    # Last decoder layer does not have an activation.
    dec_conv_activations = [activation] * (len(dec_conv_strides) - 1) + [None]
    assert len(dec_conv_kernels) == len(
        dec_conv_filters
    ), "decoder layers are misspecified: len(kernels) != len(filters)"
    assert len(dec_conv_kernels) == len(
        dec_conv_strides
    ), "decoder layers are misspecified: len(kernels) != len(strides)"

    summary = False

    deconv = mparams["deconv"]
    identity_normal_dist = mparams["identity_normal_dist"]

    dense_units = mparams["dense_units"]
    latent_ndim = mparams["latent_ndim"]
    variational = mparams["variational"]
    kl_loss_red_type = mparams["kl_loss_reduction_type"]
    if variational:
        assert (
            kl_loss_red_type in _REDUCTION_TYPES
        ), f"invalid reconstruction loss reduction type: {kl_loss_red_type}"
    fp_loss = mparams["full_precision_loss"]
    log_sparsity = False

    upsample = np.product(dec_conv_strides)
    assert (
        input_shape[1] % upsample == 0
    ), "Input shape dim1 must be divisible by decoder strides"
    assert (
        input_shape[2] % upsample == 0
    ), "Input shape dim2 must be divisible by decoder strides"
    unflatten_filters = dec_conv_filters[0]
    unflatten_shape = [unflatten_filters] + [
        d // upsample for d in input_shape[1:]
    ]
    unflatten_nelem = np.product(unflatten_shape)

    # Input
    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")

    # Encoder conv layers
    output_acts = []
    enc_conv_activations = [activation] * (len(enc_conv_strides))
    for count, (kernel, filters, stride, activation) in enumerate(
        zip(
            enc_conv_kernels,
            enc_conv_filters,
            enc_conv_strides,
            enc_conv_activations,
        )
    ):
        net, output_acts = conv2d(
            inputs=net,
            filters=filters,
            kernel_size=[kernel, kernel],
            strides=(stride, stride),
            activation=activation,
            padding="SAME",
            name=f"enc_conv_{count+1}",
            log_sparsity=log_sparsity,
            output_acts=output_acts,
            summary=summary,
        )

    # Flatten
    net = tf.keras.layers.Flatten(name="flatten_encoder")(net)
    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")

    # Encoder dense layers
    net, output_acts = dense(
        inputs=net,
        units=dense_units,
        activation=activation,
        name=f"enc_dense_1",
        log_sparsity=log_sparsity,
        output_acts=output_acts,
        summary=summary,
    )

    # Embedding
    if variational:
        # Variational, sampled embedding
        # Two FCs for mean and logvar
        net, output_acts, maybe_kl_loss = variational_embedding(
            inputs=net,
            latent_ndim=latent_ndim,
            kl_loss_reduction_type=kl_loss_red_type,
            fp_loss=fp_loss,
            name="embedding",
            log_sparsity=log_sparsity,
            output_acts=output_acts,
            summary=summary,
            identity_normal_dist=identity_normal_dist,
        )
    else:
        # Deterministic embedding
        # Single FC for mean
        net, output_acts = deterministic_embedding(
            inputs=net,
            latent_ndim=latent_ndim,
            name="embedding",
            log_sparsity=log_sparsity,
            output_acts=output_acts,
            summary=summary,
        )
        maybe_kl_loss = None

    # Decoder dense layers
    net, output_acts = dense(
        inputs=net,
        units=dense_units,
        activation=activation,
        name=f"dec_dense_1",
        log_sparsity=log_sparsity,
        output_acts=output_acts,
        summary=summary,
    )

    net, output_acts = dense(
        inputs=net,
        units=unflatten_nelem,
        activation=activation,
        name=f"dec_dense_2",
        log_sparsity=log_sparsity,
        output_acts=output_acts,
        summary=summary,
    )

    # Unflatten
    net = tf.keras.layers.Reshape(unflatten_shape, name="unflatten")(net)
    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")
    if summary:
        net = summary_layer(net)

    # Decoder layers
    dec_conv_filters_out = dec_conv_filters[1:] + [input_shape[0]]
    # Use deconvolution
    if deconv:
        for count, (kernel, filters, stride, activation) in list(
            enumerate(
                zip(
                    dec_conv_kernels,
                    dec_conv_filters_out,
                    dec_conv_strides,
                    dec_conv_activations,
                )
            )
        ):
            net, output_acts = deconv2d(
                inputs=net,
                filters=filters,
                kernel_size=[kernel, kernel],
                strides=(stride, stride),
                padding="SAME",
                activation=activation,
                name=f"dec_deconv_{count+1}",
                log_sparsity=log_sparsity,
                output_acts=output_acts,
                summary=summary,
            )

    # Use stride=1 convolutions
    else:
        for count, (kernel, filters, stride, activation) in list(
            enumerate(
                zip(
                    dec_conv_kernels,
                    dec_conv_filters_out,
                    dec_conv_strides,
                    dec_conv_activations,
                )
            )
        ):
            assert stride == 1, "Stride must be 1 if deconv is False"
            net, output_acts = conv2d(
                inputs=net,
                filters=filters,
                kernel_size=[kernel, kernel],
                strides=(stride, stride),
                padding="SAME",
                activation=activation,
                name=f"dec_conv_{count+1}",
                log_sparsity=log_sparsity,
                output_acts=output_acts,
                summary=summary,
            )

    # Flatten output
    net = tf.keras.layers.Flatten(name="flatten_final")(net)
    tf.compat.v1.logging.info(f"Shape net after {net.name}: {net.get_shape()}")
    if summary:
        net = summary_layer(net)

    return net, output_acts, maybe_kl_loss
