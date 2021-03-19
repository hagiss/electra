# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Pre-trains an ELECTRA model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json

import tensorflow.compat.v1 as tf

import configure_pretraining
from model import modeling
from model import optimization
import modeling_pred
from pretrain import pretrain_data
from pretrain import pretrain_helpers
from util import training_utils
from util import utils


class PretrainingModel(object):
    """Transformer pre-training using the replaced-token-detection task."""

    def __init__(self, config: configure_pretraining.PretrainingConfig,
                 features, is_training):
        # Set up model config
        self._config = config
        self._bert_config = training_utils.get_bert_config(config)
        if config.debug:
            self._bert_config.num_hidden_layers = 3
            self._bert_config.hidden_size = 144
            self._bert_config.intermediate_size = 144 * 4
            self._bert_config.num_attention_heads = 4

        # Mask the input
        unmasked_inputs = pretrain_data.features_to_inputs(features)
        masked_inputs = pretrain_helpers.mask(
            config, unmasked_inputs, config.mask_prob)
        masked_inputs2 = pretrain_helpers.mask(
            config, unmasked_inputs, config.mask_prob, already_masked=masked_inputs.masked_lm_ids
        )


        online = modeling_pred.BertModel(
                config=self._bert_config,
                is_training=is_training,
                input_ids=masked_inputs.input_ids,
                input_mask=masked_inputs.input_mask
            )
        target = modeling_pred.BertModel(
                config=self._bert_config,
                is_training=is_training,
                input_ids=masked_inputs2.input_ids,
                input_mask=masked_inputs2.input_mask
            )
        # mlm_output = self._get_masked_lm_output(masked_inputs, generator)
        loss1 = get_BYOL_output(self._bert_config,
                                    online.get_sequence_output(),
                                    target.get_sequence_output(),
                                    masked_inputs.masked_lm_positions)
        loss2 = get_BYOL_output(self._bert_config,
                                    target.get_sequence_output(),
                                    online.get_sequence_output(),
                                    masked_inputs2.masked_lm_positions)
        # self.mlm_output = mlm_output
        # self.total_loss = config.gen_weight * (
        #     cloze_output.loss if config.two_tower_generator else mlm_output.loss)
        self.total_loss = loss1 + loss2

        # Evaluation
        eval_fn_inputs = {
            "input_ids": masked_inputs.input_ids,
            "total_loss": self.total_loss,
            "masked_lm_ids": masked_inputs.masked_lm_ids,
            "masked_lm_weights": masked_inputs.masked_lm_weights,
            "input_mask": masked_inputs.input_mask,
            "masked_lm_ids2": masked_inputs2.masked_lm_ids,
            "masked_lm_weights2": masked_inputs2.masked_lm_weights,
            "input_mask2": masked_inputs2.input_mask
        }
        eval_fn_keys = eval_fn_inputs.keys()
        eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]

        def metric_fn(*args):
            """Computes the loss and accuracy of the model."""
            d = {k: arg for k, arg in zip(eval_fn_keys, args)}
            metrics = dict()
            # metrics["sentence_loss"] = tf.metrics.accuracy(
            #     labels=tf.reshape(d["masked_lm_ids"], [-1]),
            #     predictions=tf.reshape(d["masked_lm_preds"], [-1]),
            #     weights=tf.reshape(d["masked_lm_weights"], [-1]))
            metrics["masked_lm_loss"] = tf.metrics.mean(
                values=tf.reshape(d["mlm_loss"], [-1]),
                weights=tf.reshape(d["masked_lm_weights"], [-1]))
            return metrics

        self.eval_metrics = (metric_fn, eval_fn_values)


def get_token_logits(input_reprs, embedding_table, bert_config):
    hidden = tf.layers.dense(
        input_reprs,
        units=modeling.get_shape_list(embedding_table)[-1],
        activation=modeling.get_activation(bert_config.hidden_act),
        kernel_initializer=modeling.create_initializer(
            bert_config.initializer_range))
    hidden = modeling.layer_norm(hidden)
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(hidden, embedding_table, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    return logits


def get_softmax_output(logits, targets, weights, vocab_size):
    oh_labels = tf.one_hot(targets, depth=vocab_size, dtype=tf.float32)
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)
    label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)
    numerator = tf.reduce_sum(weights * label_log_probs)
    denominator = tf.reduce_sum(weights) + 1e-6
    loss = numerator / denominator
    SoftmaxOutput = collections.namedtuple(
        "SoftmaxOutput", ["logits", "probs", "loss", "per_example_loss", "preds",
                          "weights"])
    return SoftmaxOutput(
        logits=logits, probs=probs, per_example_loss=label_log_probs,
        loss=loss, preds=preds, weights=weights)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def get_BYOL_output(bert_config, online_tensor, target_tensor, online_positions):
    first_online_tensor = tf.squeeze(online_tensor[:, 0:1, :], axis=1)
    first_target_tensor = tf.squeeze(target_tensor[:, 0:1, :], axis=1)
    online_tensor = gather_indexes(online_tensor, online_positions)
    target_tensor = gather_indexes(target_tensor, online_positions)

    with tf.variable_scope("BYOL_loss"):
        with tf.variable_scope("BYOL_token_loss"):
            # non-linear transformation for projection
            mlp = tf.layers.dense(
                online_tensor,
                4096,
                kernel_initializer=modeling_pred.create_initializer(bert_config.initializer_range))
            mlp = tf.nn.relu(modeling_pred.layer_norm(mlp))
            prediction = tf.layers.dense(
                mlp,
                256,
                kernel_initializer=modeling_pred.create_initializer(bert_config.initializer_range))

        token_loss = tf.reduce_sum(prediction, target_tensor, axis=2)
        token_loss = token_loss / (tf.norm(prediction, axis=2) * tf.norm(target_tensor, axis=2))
        token_loss = tf.reduce_mean(2 - 2 * token_loss)

        with tf.variable_scope("BYOL_first_loss"):
            mlp_pred = tf.layers.dense(
                first_online_tensor,
                4096,
                kernel_initializer=modeling_pred.create_initializer(bert_config.initializer_range))
            mlp_pred = tf.nn.relu(modeling_pred.layer_norm(mlp_pred))
            pred_first = tf.layer.dense(
                mlp_pred,
                256,
                kernel_initializer=modeling_pred.create_initializer(bert_config.initializer_range))

        first_loss = tf.reduce_sum(pred_first, first_target_tensor, axis=1)
        first_loss = first_loss / (tf.norm(pred_first, axis=1) * tf.norm(first_target_tensor, axis=1))
        first_loss = tf.reduce_mean(2 - 2 * first_loss)

        loss_param = 1
        return first_loss + loss_param * token_loss


class TwoTowerClozeTransformer(object):
    """Build a two-tower Transformer used as Electric's generator."""

    def __init__(self, config, bert_config, inputs: pretrain_data.Inputs,
                 is_training, embedding_size):
        ltr = build_transformer(
            config, inputs, is_training, bert_config,
            untied_embeddings=config.untied_generator_embeddings,
            embedding_size=(None if config.untied_generator_embeddings
                            else embedding_size),
            scope="generator_ltr", ltr=True)
        rtl = build_transformer(
            config, inputs, is_training, bert_config,
            untied_embeddings=config.untied_generator_embeddings,
            embedding_size=(None if config.untied_generator_embeddings
                            else embedding_size),
            scope="generator_rtl", rtl=True)
        ltr_reprs = ltr.get_sequence_output()
        rtl_reprs = rtl.get_sequence_output()
        self._sequence_output = tf.concat([roll(ltr_reprs, -1),
                                           roll(rtl_reprs, 1)], -1)
        self._embedding_table = ltr.embedding_table

    def get_sequence_output(self):
        return self._sequence_output

    def get_embedding_table(self):
        return self._embedding_table


def build_transformer(config: configure_pretraining.PretrainingConfig,
                      inputs: pretrain_data.Inputs, is_training,
                      bert_config, reuse=False, **kwargs):
    """Build a transformer encoder network."""
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        return modeling.BertModel(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=inputs.input_ids,
            input_mask=inputs.input_mask,
            token_type_ids=inputs.segment_ids,
            use_one_hot_embeddings=config.use_tpu,
            **kwargs)


def roll(arr, direction):
    """Shifts embeddings in a [batch, seq_len, dim] tensor to the right/left."""
    return tf.concat([arr[:, direction:, :], arr[:, :direction, :]], axis=1)


def get_generator_config(config: configure_pretraining.PretrainingConfig,
                         bert_config: modeling.BertConfig):
    """Get model config for the generator network."""
    gen_config = modeling.BertConfig.from_dict(bert_config.to_dict())
    gen_config.hidden_size = int(round(
        bert_config.hidden_size * config.generator_hidden_size))
    gen_config.num_hidden_layers = int(round(
        bert_config.num_hidden_layers * config.generator_layers))
    gen_config.intermediate_size = 4 * gen_config.hidden_size
    gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
    return gen_config


def model_fn_builder(config: configure_pretraining.PretrainingConfig):
    """Build the model for training."""

    def model_fn(features, labels, mode, params):
        """Build the model for training."""
        model = PretrainingModel(config, features,
                                 mode == tf.estimator.ModeKeys.TRAIN)
        utils.log("Model is built!")
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                model.total_loss, config.learning_rate, config.num_train_steps,
                weight_decay_rate=config.weight_decay_rate,
                use_tpu=config.use_tpu,
                warmup_steps=config.num_warmup_steps,
                lr_decay_power=config.lr_decay_power
            )
            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=model.total_loss,
                train_op=train_op,
                training_hooks=[training_utils.ETAHook(
                    {} if config.use_tpu else dict(loss=model.total_loss),
                    config.num_train_steps, config.iterations_per_loop,
                    config.use_tpu)]
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=model.total_loss,
                eval_metrics=model.eval_metrics,
                evaluation_hooks=[training_utils.ETAHook(
                    {} if config.use_tpu else dict(loss=model.total_loss),
                    config.num_eval_steps, config.iterations_per_loop,
                    config.use_tpu, is_training=False)])
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported")
        return output_spec

    return model_fn


def train_or_eval(config: configure_pretraining.PretrainingConfig):
    """Run pre-training or evaluate the pre-trained model."""
    if config.do_train == config.do_eval:
        raise ValueError("Exactly one of `do_train` or `do_eval` must be True.")
    if config.debug and config.do_train:
        utils.rmkdir(config.model_dir)
    utils.heading("Config:")
    utils.log_config(config)

    is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    tpu_cluster_resolver = None
    if config.use_tpu and config.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            config.tpu_name, zone=config.tpu_zone, project=config.gcp_project)
    tpu_config = tf.estimator.tpu.TPUConfig(
        iterations_per_loop=config.iterations_per_loop,
        num_shards=config.num_tpu_cores,
        tpu_job_name=config.tpu_job_name,
        per_host_input_for_training=is_per_host)
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=config.model_dir,
        save_checkpoints_steps=config.save_checkpoints_steps,
        keep_checkpoint_max=config.keep_checkpoint_max,
        tpu_config=tpu_config)
    model_fn = model_fn_builder(config=config)
    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=config.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size)

    if config.do_train:
        utils.heading("Running training")
        estimator.train(input_fn=pretrain_data.get_input_fn(config, True),
                        max_steps=config.num_train_steps)
    if config.do_eval:
        utils.heading("Running evaluation")
        result = estimator.evaluate(
            input_fn=pretrain_data.get_input_fn(config, False),
            steps=config.num_eval_steps)
        for key in sorted(result.keys()):
            utils.log("  {:} = {:}".format(key, str(result[key])))
        return result


def train_one_step(config: configure_pretraining.PretrainingConfig):
    """Builds an ELECTRA model an trains it for one step; useful for debugging."""
    train_input_fn = pretrain_data.get_input_fn(config, True)
    features = tf.data.make_one_shot_iterator(train_input_fn(dict(
        batch_size=config.train_batch_size))).get_next()
    model = PretrainingModel(config, features, True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        utils.log(sess.run(model.total_loss))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Location of data files (model weights, etc).")
    parser.add_argument("--model-name", required=True,
                        help="The name of the model being fine-tuned.")
    parser.add_argument("--hparams", default="{}",
                        help="JSON dict of model hyperparameters.")
    args = parser.parse_args()
    if args.hparams.endswith(".json"):
        hparams = utils.load_json(args.hparams)
    else:
        hparams = json.loads(args.hparams)
    tf.logging.set_verbosity(tf.logging.ERROR)
    train_or_eval(configure_pretraining.PretrainingConfig(
        args.model_name, args.data_dir, **hparams))


if __name__ == "__main__":
    main()
