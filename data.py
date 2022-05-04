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
File containing input data functions used for the Covid CVAE model.
"""
import numpy as np
import tensorflow as tf


def get_real_datasets(data_dir):
    train_ds = tf.data.TFRecordDataset(f"{data_dir}/train.tfrecords")
    val_ds = tf.data.TFRecordDataset(f"{data_dir}/val.tfrecords")
    return train_ds, val_ds


def get_real_batch_map_fn(
    tfrecord_shape, input_shape, dtype, input_transform=False
):
    feature_description = {
        'data': tf.io.FixedLenFeature([1], tf.string),
    }

    def _map_fn(record):
        features = tf.io.parse_example(record, feature_description)
        data = tf.compat.v1.io.decode_raw(features['data'], tf.bool,)
        BS = data.shape.as_list()[0]
        x = tf.reshape(data, [BS] + tfrecord_shape)
        input_x = tf.slice(x, [0, 0, 0, 0], [BS] + input_shape)
        if input_transform:
            transform_input_x = 2 * tf.cast(input_x, tf.float32) - 1
        flat_input_x = tf.reshape(input_x, [BS, -1])
        if input_transform:
            return (
                tf.cast(transform_input_x, dtype),
                tf.cast(flat_input_x, dtype),
            )
        else:
            return (tf.cast(input_x, dtype), tf.cast(flat_input_x, dtype))

    return _map_fn


def get_fake_datasets(total_size, latent_ndim, seed=None):
    np.random.seed(seed)

    def _get_ds(size):
        data = np.random.rand(size, latent_ndim).astype("float32")
        return tf.data.Dataset.from_tensor_slices(data)

    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_ds = _get_ds(train_size)
    val_ds = _get_ds(val_size)
    return train_ds, val_ds


def get_fake_batch_map_fn(latent_ndim, input_shape, dtype, seed=None):
    np.random.seed(seed)
    N = np.product(input_shape)
    transform = tf.constant(
        np.random.rand(latent_ndim, N) < 0.1, dtype=tf.float32
    )

    def _map_fn(x):
        x = tf.keras.backend.dot(x, transform)
        x = tf.math.ceil(x)
        x = tf.clip_by_value(x, 0.0, 1.0)
        x = tf.reshape(x, [-1] + input_shape)
        flat_x = tf.reshape(x, [-1, N])
        return (tf.cast(x, dtype), tf.cast(flat_x, dtype))

    return _map_fn


def input_fn(params):
    iparams = params["train_input"]
    mparams = params["model"]
    mode = params["runconfig"]["mode"]
    batch_size = iparams["batch_size"]
    input_shape = iparams["input_shape"]
    mp = mparams["mixed_precision"]
    dtype = tf.float16 if mp else tf.float32
    latent_ndim = mparams["latent_ndim"]
    data_dir = iparams["data_dir"]
    tfrecord_shape = iparams["tfrecord_shape"]
    use_real_data = iparams["use_real_data"]
    seed = iparams["data_random_seed"]
    input_transform = iparams["input_transform"]
    FAKE_DATASET_SEED = 42
    FAKE_TRANSFORM_SEED = 42

    if use_real_data:
        train_ds, val_ds = get_real_datasets(data_dir)
        map_fn = get_real_batch_map_fn(
            tfrecord_shape, input_shape, dtype, input_transform=input_transform
        )
    else:
        train_ds, val_ds = get_fake_datasets(
            1000, latent_ndim, seed=FAKE_DATASET_SEED
        )
        map_fn = get_fake_batch_map_fn(
            latent_ndim, input_shape, dtype, FAKE_TRANSFORM_SEED
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        ds = train_ds
    elif mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        ds = val_ds
        batch_size = 1
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if mode == tf.estimator.ModeKeys.TRAIN:
        ds = ds.repeat()
        ds = ds.shuffle(1000, seed=seed)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
