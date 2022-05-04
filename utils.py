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


import tensorflow as tf
import yaml


def get_params(params_file):
    """
    Return params dict from yaml file.
    """
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    return params


def make_sparsity_summary(name, tensor):
    # tf.compat.v1.summary.scalar(
    #     f"input_sparsity/{name}",
    #     (
    #         1.0 - tf.math.count_nonzero(tensor, dtype=tf.float32) /
    #         tf.size(tensor, out_type=tf.float32)
    #     )
    # )
    log_hist(tensor, name, "act")


def log_hist(tensor, name, family):
    tf.compat.v1.summary.scalar(
        f"sparsity_{family}/{name}",
        (
            1.0
            - tf.math.count_nonzero(tensor, dtype=tf.float32)
            / tf.size(tensor, out_type=tf.float32)
        ),
    )
    tf.compat.v1.summary.scalar(
        f"denormal_{family}/{name}",
        (
            tf.reduce_sum(
                tf.cast(
                    tf.math.logical_and(
                        tf.math.less(tf.abs(tensor), 2 ** -14),
                        tf.math.not_equal(tensor, 0),
                    ),
                    tf.float32,
                )
            )
            / tf.size(tensor, out_type=tf.float32)
        ),
    )
    tf.compat.v1.summary.histogram(
        f"{family}/{name}",
        tf.math.log(
            tf.math.minimum(tf.cast(tf.abs(tensor), tf.float32), 2.0 ** 25)
            + 2.0 ** -50
        )
        / tf.math.log(2.0),
    )
