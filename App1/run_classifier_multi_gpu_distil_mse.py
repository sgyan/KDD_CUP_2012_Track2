# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import math
import time
import random

import modeling
import create_pretraining_data as cpd
import optimization_multi_gpu
import tokenization
import tensorflow as tf
import horovod.tensorflow as hvd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-previous-model-path",type=str, default="")
parser.add_argument("--input-training-data-path", default="")
parser.add_argument("--input-validation-data-path", default='')
parser.add_argument("--output-model-path", default='')

parser.add_argument("--input_previous_model_path",type=str, default="")
parser.add_argument("--input_training_data_path", default="")
parser.add_argument("--input_validation_data_path", default='')
parser.add_argument("--output_model_path", default='')

(args, unknown) = parser.parse_known_args()

tf.app.flags.DEFINE_string('phillyarg','blobpathconv', 'fix phillyarg issue')

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters

#flags.DEFINE_integer(
#    "n_gpus", 1,
#    "gpu number")

flags.DEFINE_integer(
    "num_train_steps", -1,
    "num_train_steps")

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

#flags.DEFINE_string("vocab_file", None,
#                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("pos_vocab_file", "pos_vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run pre dict on the dev set.")

flags.DEFINE_integer("train_batch_size", 128, "Total batch size for training.")

#flags.DEFINE_integer("eval_batch_size", 128, "Total batch size for eval.")

#flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("pos_size", 82, "pos_size")
flags.DEFINE_integer("pos_embedding_size", 9, "pos_embedding_size")

flags.DEFINE_bool("use_float32_to_float16", True, "Whether to use_float32_to_float16.")
flags.DEFINE_bool("assume_pos_is_always_one", False, "assume_pos_is_always_one.")
flags.DEFINE_bool("profiler", False, "profiler")
flags.DEFINE_bool("xla", False, "xla")
flags.DEFINE_bool("opt_cpu", False, "opt_cpu")

flags.DEFINE_bool("multi_task", True, "multi_task")
flags.DEFINE_bool("with_mlm_loss", True, "with_mlm_loss")
flags.DEFINE_bool("with_w_mlm_loss", False, "with_w_mlm_loss")
flags.DEFINE_float("w_mlm", 1.0, "w_mlm")
flags.DEFINE_bool("with_trainable_w_multi_loss", False, "with_trainable_w_multi_loss")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("only_pos_embedding", False, "only_use_pos_embedding")
flags.DEFINE_bool("output_dim_is_one", True, "output_dim_is_one")
flags.DEFINE_bool("lus_add_bias", True, "lus_add_bias")
flags.DEFINE_bool("freeze_bert", False, "freeze_bert")
flags.DEFINE_bool("undersampling", False, "undersampling")
flags.DEFINE_float("sampling_rate", 0.333333, "sampling_rate")
flags.DEFINE_string("data_checkpoint", "data_checkpoint.txt", "data_checkpoint")

flags.DEFINE_bool("read_local", False, "read local")
flags.DEFINE_bool(
    "use_warmup", True,
    "use_warmup")

flags.DEFINE_string(
    "validation_data_dir", None,
    "")

flags.DEFINE_bool("run_on_aether", False, "Whether to run training.")

flags.DEFINE_string(
    "input_training_data_path", None,
    "")

flags.DEFINE_string(
    "input_validation_data_path", None,
    "")

flags.DEFINE_string(
    "input_previous_model_path", None,
    "")

flags.DEFINE_string(
    "output_model_path", None,
    "")

flags.DEFINE_integer("train_line_cnt", 0,"input data instance num.")

def clip_grads(grads, all_clip_norm_val):
    # grads = [(grad1, var1), (grad2, var2), ...]
    def _clip_norms(grad_and_vars, val, name):
        # grad_and_vars is a list of (g, v) pairs
        grad_tensors = [g for g, v in grad_and_vars]
        vv = [v for g, v in grad_and_vars]
        scaled_val = val
        clipped_tensors, g_norm = tf.clip_by_global_norm(
            grad_tensors, scaled_val)

        ret = []
        for t, (g, v) in zip(clipped_tensors, grad_and_vars):
            ret.append((t, v))

        return ret

    ret = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

    assert len(ret) == len(grads)

    return ret

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

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None, pos=None, perc=None, weight=None, pos_bias=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.pos = pos
    self.perc = perc
    self.weight = weight
    self.pos_bias = pos_bias


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the test set."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    if FLAGS.read_local:
      #dst = "./train.tsv"
      #if not tf.gfile.Exists(dst):
      #  tf.io.gfile.copy(input_file, dst)
      #  print("Download file Done!")
      #__tmp__ = hvd.mpi_ops.allgather(tf.constant(0.0, shape=[4, 1]))
      for i in range(int(FLAGS.num_train_epochs+1)):
        with tf.gfile.Open(input_file, "r") as f:
          reader = csv.reader((line.replace('\0','') for line in f), delimiter="\t", quotechar=quotechar)
          for index, line in enumerate(reader):
            if index % hvd.size() == hvd.rank():
              yield line
    else: 
      with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        for line in reader:
          yield line


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""
  def get_train_and_dev_line_cnt(self, train_data_dir, dev_data_dir):
    train_line_cnt = 0
    dev_line_cnt = 0
    if FLAGS.do_train:
      #train_line_cnt = 10000000 #478585308 #TODO
      with tf.gfile.Open(os.path.join(train_data_dir, "train.tsv"), "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for _ in reader:    
          train_line_cnt += 1
    if FLAGS.do_eval:
      with tf.gfile.Open(os.path.join(dev_data_dir, "dev.tsv"), "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for _ in reader:    
          dev_line_cnt += 1
    return train_line_cnt, dev_line_cnt

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_test_examples(
        self._read_tsv(data_dir), "dev")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""

    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      #print("origin line:[%s]" % line)
      guid = "%s-%s" % (line[1], line[2])
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      label = tokenization.convert_to_list(line[0],float) #Teacher Predict Emb29
      pos = tokenization.convert_to_unicode('ml-1')
      weight = float(line[5])
      perc = float(0)
      pos_bias = float(7.50)
      yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, pos=pos, perc=perc, weight=weight, pos_bias = pos_bias)

  def _create_test_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""

    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      #print("origin line:[%s]" % line)
      guid = "%s-%s" % (line[0], line[1])
      text_a = tokenization.convert_to_unicode(line[2])
      text_b = tokenization.convert_to_unicode(line[3])
      label = [float(0)]*29 #Teacher Predict Emb29
      pos = tokenization.convert_to_unicode('ml-1')
      weight = float(1)
      perc = float(0)
      pos_bias = float(7.50)
      yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, pos=pos, perc=perc, weight=weight, pos_bias = pos_bias)



def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, pos_tokenizer, last_checkpoint, rng=None, save_checkpoint=False):
  """Loads a data file into a list of `InputBatch`s."""

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  if FLAGS.do_train:
    data_checkpoint_file = os.path.join(FLAGS.output_dir, "train_" + FLAGS.data_checkpoint)
  elif FLAGS.do_predict:
    data_checkpoint_file = os.path.join(FLAGS.output_dir, "pred_" + FLAGS.data_checkpoint)
  
  vocab_words = list(tokenizer.vocab.keys())

  for (ex_index, example) in enumerate(examples):
    #print("rank:[%s] do [%d]" % (hvd.rank(), ex_index))
    #if ex_index % 10000 == 0:
    #    print("ex_index:%d" % ex_index)
    if ex_index < last_checkpoint:
        continue
    if save_checkpoint and ex_index % (FLAGS.save_checkpoints_steps * FLAGS.train_batch_size * hvd.size()) == 0:
      with tf.gfile.GFile(data_checkpoint_file, "w") as writer:
        writer.write("%d" % (ex_index))

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

    if FLAGS.multi_task:
    #TODO
      (lm_tokens, masked_lm_positions,
       masked_lm_labels) = cpd.create_masked_lm_predictions(
             tokens, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq, vocab_words, rng)
      masked_lm_labels = tokenizer.convert_tokens_to_ids(masked_lm_labels)
      masked_lm_weights = [1.0] * len(masked_lm_labels)
      #masked_lm_positions, masked_lm_labels, masked_lm_weights
      while len(masked_lm_positions) < FLAGS.max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_labels.append(0)
        masked_lm_weights.append(0.0)
      lm_input_ids = tokenizer.convert_tokens_to_ids(lm_tokens)
    else:
      (lm_tokens, lm_input_ids) = [[0] * max_seq_length] * 2
      masked_lm_positions, masked_lm_labels, masked_lm_weights = [[0] * FLAGS.max_predictions_per_seq] * 3

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    pos_ids = pos_tokenizer.pos_convert_tokens_to_ids([example.pos])
    assert len(pos_ids) == 1
    if FLAGS.assume_pos_is_always_one:
      pos_id = 0
    else:
      pos_id = pos_ids[0]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      if FLAGS.multi_task:
        lm_input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    if FLAGS.multi_task:
      assert len(lm_input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # label_id = label_map[example.label]
    perc = float(example.perc)
    weight = float(example.weight)
    pos_bias = float(example.pos_bias)
    guid = example.guid
    instance_ids = example.guid.split("-")
    assert len(instance_ids) == 2

    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("guid: [%s][%s]" % (instance_ids[0], instance_ids[1]))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("lm_input_ids: %s" % " ".join([str(x) for x in lm_input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      tf.logging.info("lm_tokens: %s" % " ".join([str(x) for x in lm_tokens]))
      tf.logging.info("masked_lm_positions: %s" % " ".join([str(x) for x in masked_lm_positions]))
      tf.logging.info("masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))
      tf.logging.info("masked_lm_weights: %s" % " ".join([str(x) for x in masked_lm_weights]))
      tf.logging.info("pos: [%s]->[%d]" % (example.pos, pos_id))
      tf.logging.info("perc: [%s]->[%f]" % (example.perc, perc))
      tf.logging.info("weight: [%s]->[%f]" % (example.weight, weight))
      tf.logging.info("pos_bias: [%s]->[%f]" % (example.pos_bias, pos_bias))
    yield guid, input_ids, lm_input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_labels, masked_lm_weights, pos_id, perc, weight, pos_bias, example.label
    


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 num_labels, pos_ids, percs, weight, pos_bias, use_one_hot_embeddings):
  """Creates a classification model."""
  dtype = tf.float32
  if FLAGS.use_float32_to_float16:
    dtype = tf.float16
    print("[NOTICE]: change fp32 to [%s]" % dtype)
  print("[NOTICE]: [%s]" % dtype)
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      dtype=dtype)

  dtype = tf.float32

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()
  sub_train_vars = []
  hidden_size = output_layer.shape[-1].value
  count_hidden_size=29
  with tf.variable_scope("bert/cls", reuse=tf.AUTO_REUSE):
    #[29,768]
    bert_output_weights = tf.get_variable(
        "bert_output_weights", [count_hidden_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=dtype),
        dtype=dtype)
    #[29]
    bert_output_bias = tf.get_variable(
        "bert_output_bias", [count_hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=dtype),
        dtype=dtype)
    # output_weights = tf.get_variable(
    #     "output_weights", [num_labels, hidden_size],
    #     initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=dtype),
    #     dtype=dtype)

    # output_bias = tf.get_variable(
    #     "output_bias", [num_labels], initializer=tf.zeros_initializer(dtype=dtype),
    #     dtype=dtype)

    #pos_weights = tf.get_variable(
    #    "pos_weights", [num_labels, FLAGS.pos_embedding_size],
    #    initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=dtype),
    #    dtype=dtype)
    
    #pos_bias = tf.get_variable(
    #    "pos_bias", [num_labels], initializer=tf.zeros_initializer(dtype=dtype),
    #    dtype=dtype)
    
    #if FLAGS.only_pos_embedding:
    #  pos_embedding_table = tf.get_variable(
    #    name="pos_embedding",
    #    shape=[FLAGS.pos_size, num_labels],
    #    initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=dtype),
    #    dtype=dtype)

    #else:
    #  pos_embedding_table = tf.get_variable(
    #    name="pos_embedding",
    #    shape=[FLAGS.pos_size, FLAGS.pos_embedding_size],
    #    initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=dtype),
    #    dtype=dtype)

    # if FLAGS.freeze_bert:
    #     sub_train_vars.append(output_weights)
    #     sub_train_vars.append(output_bias)
    #     sub_train_vars.append(pos_weights)
    #     sub_train_vars.append(pos_bias)
    #     sub_train_vars.append(pos_embedding_table)

  with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    bert_logits = tf.matmul(output_layer, bert_output_weights, transpose_b=True)
    bert_logits = tf.nn.bias_add(bert_logits, bert_output_bias)
    # logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    # logits = tf.nn.bias_add(logits, output_bias)
    
    #pos_output = tf.nn.embedding_lookup(pos_embedding_table, pos_ids)
    #if not FLAGS.only_pos_embedding:
    #  pos_output = tf.matmul(pos_output, pos_weights, transpose_b=True)
    #  pos_output = tf.nn.bias_add(pos_output, pos_bias)

    #logits = tf.add(logits, pos_output)
    # if FLAGS.output_dim_is_one:
    #   percs = tf.reshape(percs, [-1, 1])
    #   pos_bias = tf.reshape(pos_bias, [-1, 1])
    #   weight = tf.reshape(weight,[-1,1])
    #   if FLAGS.lus_add_bias:
    #       logits = tf.add(logits, percs)
    #       logits = tf.add(logits, pos_bias)
    return bert_logits, sub_train_vars, model


def CosineEmbeddingLoss(margin=0.):
    def _cosine_similarity(x1, x2):
        """Cosine similarity between two batches of vectors."""
        return tf.reduce_sum(tf.multiply(x1, x2), axis=-1) / (
            tf.norm(x1, axis=-1) * tf.norm(x2, axis=-1))

    def _cosine_embedding_loss_fn(input_one, input_two, target):
        similarity = _cosine_similarity(input_one, input_two)
        return tf.where(
            tf.equal(target, 1),
            1. - similarity,
            tf.maximum(tf.zeros_like(similarity), similarity - margin))
    return _cosine_embedding_loss_fn


def create_loss(logits, percs, labels, num_labels, weight, dtype):
  if FLAGS.output_dim_is_one:
      # labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.float32)
      weight = tf.reshape(weight,[-1,1])
      # per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
      # per_example_loss = tf.multiply(per_example_loss, weight)
      total_weight = tf.reduce_sum(weight)
      # loss = tf.divide(tf.reduce_sum(per_example_loss),total_weight)
      # targets = tf.ones([FLAGS.train_batch_size], tf.int32)
      # per_example_loss = tf.reduce_sum(tf.multiply(CosineEmbeddingLoss()(
      #    labels, logits, targets), weight))
      per_example_loss = tf.reduce_sum(tf.multiply(tf.square(
         labels-logits), weight))
      loss = tf.divide(tf.reduce_sum(per_example_loss),total_weight)
      return (loss, per_example_loss)
  else:
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=dtype)
      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      if FLAGS.undersampling:
          weight_0 = tf.constant(1.0 / FLAGS.sampling_rate) * tf.cast(tf.equal(labels, 0), tf.float32)
          weight_1 = tf.constant(1.0) * tf.cast(tf.equal(labels, 1), tf.float32)
          weight = weight_0 + weight_1
          per_example_loss = per_example_loss * weight
          loss = tf.reduce_mean(per_example_loss)
      else:
          loss = tf.reduce_mean(per_example_loss)
      return (loss, per_example_loss)

def create_pred(logits):
  if FLAGS.output_dim_is_one:
      pred = tf.math.sigmoid(logits)
  else:
      softmax_all = tf.nn.softmax(logits=logits)
      pred = softmax_all[:,1]
  return pred

def restore_vars(init_checkpoint):
  tvars = tf.trainable_variables()

  if init_checkpoint:
    (assignment_map,
    initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
        tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

  tf.logging.info("**** Variables ****")
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for GPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for GPUEstimator."""

    dtype = tf.float32
    #if FLAGS.use_float32_to_float16:
    #  dtype = tf.float16

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
          
    input_ids = features["input_ids"]
    lm_input_ids = features["lm_input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    pos_ids = features["pos_ids"]
    percs = features["percs"]
    weight = features["weight"]
    pos_bias = features["pos_bias"]
    label_ids = labels
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_labels = features["masked_lm_labels"]
    masked_lm_weights = features["masked_lm_weights"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      global_step = tf.train.get_or_create_global_step()
      optimizer, tfvar_learning_rate = optimization_multi_gpu.create_optimizer(
          None, learning_rate, num_train_steps, num_warmup_steps, None, dtype)
      if FLAGS.opt_cpu:
        #optimizer = hvd.DistributedOptimizer(optimizer, device_sparse='/gpu:0')
        print("communicate with fp16 [NOTICE]")
        optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=True, compression=hvd.Compression.fp16)
      else:
        optimizer = hvd.DistributedOptimizer(optimizer)

      if FLAGS.use_float32_to_float16:
        loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)

      (logits, sub_train_vars, model) = create_model(
          bert_config, is_training, input_ids, input_mask, segment_ids,
          num_labels, pos_ids, percs, weight, pos_bias, use_one_hot_embeddings)
      (classify_loss, per_example_loss) = create_loss(logits, percs, label_ids, num_labels, weight, dtype) 
      loss = classify_loss
      hook_map = {"classify_loss": classify_loss, "learning_rate": tfvar_learning_rate}

      if FLAGS.multi_task:
        (lm_logits, lm_sub_train_vars, lm_model) = create_model(
            bert_config, is_training, lm_input_ids, input_mask, segment_ids,
            num_labels, pos_ids, percs, weight, pos_bias, use_one_hot_embeddings)
        (masked_lm_loss,
           masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
           bert_config, lm_model.get_sequence_output(), lm_model.get_embedding_table(),
           masked_lm_positions, masked_lm_labels, masked_lm_weights)

        loss = classify_loss
        hook_map = {"classify_loss": classify_loss, "learning_rate": tfvar_learning_rate}
        #loss = classify_loss + masked_lm_loss #TODO
        if (FLAGS.with_mlm_loss):
          loss += masked_lm_loss
          hook_map["masked_lm_loss"] = masked_lm_loss
        elif FLAGS.with_w_mlm_loss:
          loss += FLAGS.w_mlm * masked_lm_loss 
          hook_map["masked_lm_loss"] = masked_lm_loss
        elif FLAGS.with_trainable_w_multi_loss:
          w_classify_loss = tf.get_variable("w_classify_loss", [1], initializer=tf.zeros_initializer(dtype=tf.float32), dtype=tf.float32)
          w_mlm_loss = tf.get_variable("w_mlm_loss", [1], initializer=tf.zeros_initializer(dtype=tf.float32), dtype=tf.float32)
          classify_loss = tf.exp(-w_classify_loss) * classify_loss + w_classify_loss
          masked_lm_loss = tf.exp(-w_mlm_loss) * masked_lm_loss + w_mlm_loss
          loss = classify_loss + masked_lm_loss
          hook_map["w_classify_loss"] = w_classify_loss
          hook_map["w_mlm_loss"] = w_mlm_loss
          hook_map["masked_lm_loss"] = masked_lm_loss
          hook_map["classify_loss"] = classify_loss

      hook_map["loss"] = loss

      grads = optimizer.compute_gradients(loss, var_list=None)

      train_op = optimizer.apply_gradients(grads)
      new_global_step = global_step + 1
      train_op = tf.group(train_op, [global_step.assign(new_global_step)])

      restore_vars(init_checkpoint)

      logging_hook = tf.train.LoggingTensorHook(hook_map, every_n_iter=10)
      profiler_hook = tf.train.ProfilerHook(save_steps=100, output_dir=FLAGS.output_dir, show_dataflow=True, show_memory=True)
      hooks = []
      if FLAGS.profiler:
        hooks = [logging_hook, profiler_hook]
      else:
        hooks = [logging_hook]
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          training_hooks=hooks)
    elif mode == tf.estimator.ModeKeys.EVAL:

      (logits, sub_train_vars, model) = create_model(
          bert_config, is_training, input_ids, input_mask, segment_ids,
          num_labels, pos_ids, percs, weight, pos_bias, use_one_hot_embeddings)
      (total_loss, per_example_loss) = create_loss(logits, percs, label_ids, num_labels, weight, dtype) 

      loss = tf.metrics.mean(per_example_loss)

      pred = create_pred(logits)
      restore_vars(init_checkpoint)
      
      hvd_pred = hvd.mpi_ops.allgather(pred)
      hvd_label_ids = hvd.mpi_ops.allgather(label_ids)

      auc = tf.metrics.auc(hvd_label_ids, hvd_pred)
      eval_metrics = {
                        "eval_loss": loss,
                        "eval_auc": auc
                     }

      logging_hook = tf.train.LoggingTensorHook({"pred": hvd_pred}, every_n_iter=1000)
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics,
          evaluation_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.PREDICT:
      guids = features["guids"]
      
      (logits, sub_train_vars, model) = create_model(
          bert_config, is_training, input_ids, input_mask, segment_ids,
          num_labels, pos_ids, percs, weight, pos_bias, use_one_hot_embeddings)

      # preds = create_pred(logits)

      restore_vars(init_checkpoint)
      
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={
            "pred":logits,
            "guid": guids
          }
      )
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def undersampling_filter(input_ids, input_mask, segment_ids, pos_ids, percs, labels):
    acceptance = tf.cond(tf.equal(labels, 1), lambda: True, lambda: tf.less(tf.random.uniform([1]), FLAGS.sampling_rate))
    return acceptance


def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to GPUEstimator."""
  def input_fn(params):
    def _decode(guids, input_ids, lm_input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_labels, masked_lm_weights, pos_ids, percs, weight, pos_bias, label_ids):
      return {"guids": guids, "input_ids":input_ids, "lm_input_ids":lm_input_ids, "input_mask":input_mask, "segment_ids":segment_ids, "pos_ids": pos_ids, "percs": percs, "weight":weight, "pos_bias":pos_bias,
           "masked_lm_positions": masked_lm_positions, "masked_lm_labels": masked_lm_labels, "masked_lm_weights": masked_lm_weights}, label_ids

    batch_size = FLAGS.train_batch_size
  
    d = tf.data.Dataset.from_generator(
        features,
        (tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32),
        (
          tf.TensorShape([]),
          tf.TensorShape([seq_length]),
          tf.TensorShape([seq_length]),
          tf.TensorShape([seq_length]),
          tf.TensorShape([seq_length]),
          tf.TensorShape([FLAGS.max_predictions_per_seq]),
          tf.TensorShape([FLAGS.max_predictions_per_seq]),
          tf.TensorShape([FLAGS.max_predictions_per_seq]),
          tf.TensorShape([]),
          tf.TensorShape([]),
          tf.TensorShape([]),
          tf.TensorShape([]),
          tf.TensorShape([29]),
        )
    )
    if not FLAGS.read_local:
      d = d.shard(hvd.size(), hvd.rank())
    d = d.apply(
      tf.data.experimental.map_and_batch(
        lambda guids, input_ids, lm_input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_labels, masked_lm_weights, pos_ids, percs, weight, pos_bias, label_ids: _decode(guids, input_ids, lm_input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_labels, masked_lm_weights, pos_ids, percs, weight, pos_bias, label_ids), 
        batch_size=batch_size, #num_parallel_batches=2,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        drop_remainder=True)
    ).prefetch(2)
    #if is_training:
    #  batch_size = batch_size
    #  d = d.shuffle(buffer_size=batch_size * 2)
    #  d = d.repeat()
    #  if FLAGS.undersampling:
    #    d = d.filter(undersampling_filter)
    #d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder).prefetch(4)
    return d
    #guids, offer_ids, input_ids, input_mask, segment_ids, pos_ids, percs, label_ids = d.batch(batch_size=batch_size, drop_remainder=drop_remainder).prefetch(1).make_one_shot_iterator().get_next()
    #return {"guids": guids, "offer_ids": offer_ids, "input_ids":input_ids, "input_mask":input_mask, "segment_ids":segment_ids, "pos_ids": pos_ids, "percs": percs}, label_ids

  return input_fn


def main(_):

  hvd.init()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.compat.v1.enable_resource_variables()
  if FLAGS.validation_data_dir is None:
    FLAGS.validation_data_dir = FLAGS.data_dir

  if FLAGS.run_on_aether:
    FLAGS.init_checkpoint = args.input_previous_model_path

    if args.input_training_data_path is not None:
      FLAGS.data_dir = args.input_training_data_path

    if args.input_validation_data_path is not None:
      FLAGS.validation_data_dir = args.input_validation_data_path

    FLAGS.output_dir = args.output_model_path
  FLAGS.output_dir = FLAGS.output_dir if hvd.rank() == 0 else os.path.join(FLAGS.output_dir, str(hvd.rank()))

  processors = {
      "ads": MrpcProcessor,
  }

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_eval` or `do_predict` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))


  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()
  train_line_cnt, dev_line_cnt = (FLAGS.train_line_cnt,FLAGS.train_line_cnt) if FLAGS.train_line_cnt > 0 else processor.get_train_and_dev_line_cnt(FLAGS.data_dir, FLAGS.validation_data_dir)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  pos_tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.pos_vocab_file)

  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    data_checkpoint_file = os.path.join(FLAGS.output_dir, "train_" + FLAGS.data_checkpoint)
    print(data_checkpoint_file)
    last_checkpoint = 0
    if tf.gfile.Exists(data_checkpoint_file):
      with tf.gfile.GFile(data_checkpoint_file, "r") as fin:
        last_checkpoint = int(fin.read())
        print("train_last_checkpoint:[%d]" % last_checkpoint)
    #train_line_cnt -= last_checkpoint

    num_train_steps = int(
        train_line_cnt / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    if FLAGS.num_train_steps != -1:
      num_train_steps = FLAGS.num_train_steps
    if FLAGS.undersampling:
      num_train_steps *= FLAGS.sampling_rate

    num_train_steps = num_train_steps // hvd.size()
    
    if FLAGS.use_warmup:
      num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
   
  output_dim = len(label_list)
  if FLAGS.output_dim_is_one:
      output_dim = 1
  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=output_dim,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_one_hot_embeddings=False)

  session_config = tf.ConfigProto(device_count={"CPU": 4}, allow_soft_placement=True)
  if not FLAGS.profiler:
    session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  session_config.gpu_options.visible_device_list = str(hvd.local_rank()) #"0" 
  session_config.gpu_options.allow_growth = True 
  session_config.gpu_options.allocator_type = 'BFC'
  session_config.gpu_options.per_process_gpu_memory_fraction = 0.98
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      keep_checkpoint_max=3,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps if hvd.rank() == 0 else None).replace(session_config=session_config)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

  rng = random.Random(FLAGS.random_seed)
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    train_features = lambda: convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, pos_tokenizer, last_checkpoint, rng, True)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", train_line_cnt)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = input_fn_builder(
        features=train_features,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=hooks)

  if FLAGS.do_predict:
    data_checkpoint_file = os.path.join(FLAGS.output_dir, "pred_" + FLAGS.data_checkpoint)
    print(data_checkpoint_file)
    last_checkpoint = 0
    if tf.gfile.Exists(data_checkpoint_file):
      with tf.gfile.GFile(data_checkpoint_file, "r") as fin:
        last_checkpoint = int(fin.read())
        print("pred_last_checkpoint:[%d]" % last_checkpoint)
        last_checkpoint = max(0, last_checkpoint - (FLAGS.train_batch_size * hvd.size()) * 100)
        print("fix_pred_last_checkpoint:[%d]" % last_checkpoint)

    #predict_examples = processor.get_dev_examples(FLAGS.validation_data_dir)
    predict_examples = processor.get_test_examples(FLAGS.validation_data_dir)
    predict_features = lambda: convert_examples_to_features(
        predict_examples, label_list, FLAGS.max_seq_length, tokenizer, pos_tokenizer, last_checkpoint, rng, True)

    tf.logging.info("***** Running predict *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)

    predict_input_fn = input_fn_builder(
        features=predict_features,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)
    
    result = estimator.predict(input_fn=predict_input_fn, yield_single_examples=True)
    output_predict_file = os.path.join(FLAGS.output_dir, "predict_results.txt")
    tf.logging.info("***** [%s] *****", output_predict_file)
    tf.logging.info("***** [%s] *****", FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    with tf.gfile.GFile(output_predict_file, "a") as writer:
      tf.logging.info("***** Predict results *****")
      for i, result1 in enumerate(result):
        writer.write("%s\t%s\n" % (str(result1["guid"]), ';'.join("%.6f" % res for res in result1["pred"])))
        if i % 50000 == 0:
            tf.logging.info("Predict [%s]->[%s]", str(result1["guid"]), ';'.join("%.6f" % res for res in result1["pred"]))

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.validation_data_dir)
    eval_features = lambda: convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, pos_tokenizer, 0, rng)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", dev_line_cnt)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)

    eval_steps = None

    if FLAGS.num_train_steps != -1: #just for debug
      eval_steps = FLAGS.num_train_steps

    eval_input_fn = input_fn_builder(
        features=eval_features,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, hooks=hooks)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    print(output_eval_file)
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        if key == 'softmax_all':
            pass
        else:
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))



if __name__ == "__main__":

  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")

  tf.app.run()
