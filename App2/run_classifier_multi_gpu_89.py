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


# Joint Training pAdjust Abacus 202001 89 counting feature
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

flags.DEFINE_integer(
    "extra_embedding_size", 89,
    "")

flags.DEFINE_integer(
    "ori_layer_num", 3,
    "")

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

  def __init__(self, guid, text_a, text_b=None, label=None, pos=None, perc=None, weight=None, pos_bias=None, extra_vec=None):
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
    self.extra_vec = extra_vec

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
      with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader((line.replace('\x00','') for line in f), delimiter="\t", quotechar=quotechar)
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
        reader = csv.reader((line.replace('\x00','') for line in f), delimiter="\t", quotechar=None)
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
      #print("origin line:[%s]" % line)
      guid = "%s-%s" % (line[1], line[2])
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      label = tokenization.convert_to_unicode(line[0])
      pos = tokenization.convert_to_unicode(line[5])
      weight = float(line[6])
      perc = float(0)
      pos_bias = float(0)
      extra_vec = tokenization.convert_to_list(line[7],float) #Counting Features 89
      yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, pos=pos, perc=perc, weight=weight, pos_bias = pos_bias, extra_vec = extra_vec)

  def _create_test_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""

    for (i, line) in enumerate(lines):
      #print("origin line:[%s]" % line)
      guid = "%s-%s" % (line[0], line[1])
      text_a = tokenization.convert_to_unicode(line[2])
      text_b = tokenization.convert_to_unicode(line[3])
      label = tokenization.convert_to_unicode("0")
      pos = tokenization.convert_to_unicode("ml-1")
      weight = float(1.0)
      perc = float(0)
      pos_bias = float(0)
      extra_vec = [0.0]*89 #Counting Features 89
      yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, pos=pos, perc=perc, weight=weight, pos_bias = pos_bias, extra_vec = extra_vec)



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

    label_id = label_map[example.label]
    perc = float(example.perc)
    weight = float(example.weight)
    pos_bias = float(example.pos_bias)
    guid = example.guid
    instance_ids = example.guid.split("-")
    assert len(instance_ids) == 2

    extra_vec = [v for v in example.extra_vec]

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
      tf.logging.info("extra_vec: %s" % " ".join([str(x) for x in extra_vec]))
    yield guid, input_ids, lm_input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_labels, masked_lm_weights, pos_id, perc, weight, pos_bias, extra_vec, label_id
    


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


NNWeights = [[-0.791979013010597,-0.15622240064476,-0.101862853816375,0.0879211801799581,-0.0181342748859648,-0.0344231589099116,-0.0297592540260954,0.0109205216105441,0.107812706864039,-0.000911141296844013,0.0240257913472878,0.0469759781167462,0.0813341824139268,0.0379854399855664,0.0660819413457389,0.0279534687333264,0.0537354065220251,0.00753821131132911,-0.132148730962326,-0.0324780283735975,0.110363921759878,0.100845892709482,-0.094495769697247,-0.0990424520264477,0.550076230719135,0.379934859070005,-0.232557277922649,-21.9922982674543,0.355109366196113,-0.0105163062112299,0.154151280614906,0.0152951727499505,0.333676964771797,-0.37564204492795,0.307993897596047,-0.197318284483654,-0.334067291293184,-0.480069196699271,-0.0363101093776268,-0.0374013960915883,-0.306128468759077,0.111956028646421,-0.146965974075299,-0.106224589969696,0.374589831793059,0.258724417813534,0.644762593743139,-0.127006448808999,0.421681498812334,0.688376508674861,-0.0230986672216575,0.233340224396648,0.0171469066691171,0.0819211891535077,-0.136308027287282,-0.111470726578596,0.590753130223232,2.31993403237758,-0.273567350807049,1.0250881603421,-1.10757815307715,0.191761763700433,-0.00626380917095708,-0.0034709369541156,0.151580663634153,0.516586046496377,1.84870312493031,0.0544046060603841,0.230734440442125,0.319925402693396,-0.497174838541543,-0.338707005029656,0.101979487454472,0.0466025098834531,-0.118126513923808,-0.339654604123408,-0.526830935763844,-0.323400753563649,-0.389388069945347,-0.484761408177774,-0.200884735139005,-0.372592386401829,-0.557795223687649,-0.571776454886106,-0.631189922555283,-0.30949366922465,-0.265219286244246,0.147714128398394,-0.405560505386815],
[0.254340024490222,-0.0866959010206766,0.00636102117640549,-0.29705968736414,-0.0262183446764398,-0.170343939713357,0.0375270721404872,0.0603262436009911,-0.0474813026249319,-0.00753139830004739,0.0691006949059189,0.0434631400330382,0.129960325651486,0.0281698742538849,-0.0228995236399194,-0.0288132277579698,0.00706488878767898,0.134554059188827,-0.028151262108999,0.162404183250895,-0.0428368147915883,0.139042128898137,0.0360392628490842,-0.0958397638322968,0.107121419221827,-0.224531253350614,-0.374368039543613,-5.22754133918786,0.483559647665229,-0.608670003056307,-0.0985808173162734,-0.245916735427692,-1.18843886522201,-0.244807679420865,0.0530792218612833,0.0452606492582595,0.459941748524799,-4.2634062631195,0.177159483895974,0.156121677934447,-0.421690252912388,0.616159858106395,-0.513221073085221,0.453950479642514,-1.82111270442792,-0.917051633039302,0.606553534604444,0.256753164961265,0.23308290123842,-0.258554525930668,-0.706921897036678,-0.905323869170512,0.530162366015745,0.561462947430527,-1.0585703583766,-0.696229389971364,-0.806825364335374,-6.1020807590984,-0.948735434125036,0.571254968265074,0.416301850437255,-0.502221064828244,-0.0422350285011375,0.13915711406576,0.356067794500258,0.252224614206872,-0.927879477291793,0.471003163978461,0.683010813921788,-0.0932190260657207,0.141745826589133,-0.493246072537924,0.602966521445704,-0.573978826294653,0.117486642056073,0.515225898462722,0.186217662563625,-0.0657492658909355,0.191528047765735,0.228270679881786,0.374552512538777,0.258772527409644,0.0885075190839945,-1.41922115095796,-0.935799180687015,0.546327294947777,1.11645927226435,1.96225512503636,0.400752226279148],
[-0.609835835277802,0.0572980869435847,-0.0802485111886583,-0.0383663161122149,-0.136840227464186,-0.0268960633567917,0.0494139463180252,-0.0807847268050105,0.0672464824464445,0.0129098613299158,-0.162413336774324,-0.126804812484309,-0.0486435221756457,-0.0405421866808375,0.101564365166289,0.200045269696466,0.0528128396956876,0.133400326746508,0.00109692475565671,0.0737744763594375,0.0326051284365757,-0.0360895800134789,0.066800384631972,-0.0331220995898559,0.00526146470407633,-0.176081216443723,-0.0967294278685661,-2.48682341061363,-0.340999701090084,0.139188376125809,0.0840951871581897,-0.0192528793476885,-0.0425141345705044,0.0199453713412982,0.0703507489779394,-0.0328318028712236,-0.0189787602674528,0.0421491025563898,0.222036455168835,0.0673133595563238,-0.148109751876715,0.0991560691503026,0.17502695789927,0.0231817274026714,0.16271911480943,-0.0162590153001957,-0.0643382804102881,-0.0321200820782762,0.30052358033488,0.0842583455523097,-0.0117379167758083,-0.0516232320809312,-0.191431390907248,-0.0591220197373949,-0.133487781601247,-0.0972427276584561,-0.145342295258242,-0.17911601539377,-0.0802971283675224,0.00531732794595985,-0.235321564607514,-0.0495675410813254,0.0100214220813653,0.0111288783863274,0.115345079994312,0.0162973302663296,-0.0563571422152056,-0.0741954218152055,-0.247723774373289,-0.124194335274143,-0.445307595161059,0.175273891774047,-0.307448273694077,-0.0889760295726627,-0.240194693632716,-0.374578353866026,-0.139123506347659,0.0236992065775284,-0.118951970429612,0.165559244028175,-0.113571746759205,-0.202933209172287,-0.431063429923768,-0.143705537610091,0.159542138017401,-0.0131102096358973,-0.227416682692783,-0.415442580620945,-0.121906921740301],
[1.10935143665611,-0.105434292787549,-0.189504275172311,0.0682863418240012,-0.0458986401900154,-0.322719909637635,0.0286773718088468,0.142694667090781,-0.086003050677843,-0.082566246355616,-0.37587779326434,0.477331310299886,0.313600349928284,-0.061978447441289,-0.150907915971063,-0.152913525740059,-0.0238734879522606,-0.176192330353651,0.0809127786800589,0.0750821713779923,0.186785880063522,0.165911896211184,0.0720222638104436,-0.0323608704039092,-0.345470212489282,0.45688499915223,-0.253088266742511,-9.44757538781895,1.06471292032892,0.274681226986172,-0.177621135438414,-0.451408643716577,-0.590590533611074,1.23161340181421,0.181930309766407,0.664107018100783,-0.364604462257391,1.34020429666755,-0.759010914096667,0.785119319059119,-0.712315110446729,-0.439740332517847,0.721571653401983,0.392268941372673,-1.75467020622787,-0.2213749175907,-0.0175863291876848,0.37560675982626,0.0366740270415207,-0.14240691822347,-0.33643536230756,-0.849177364402896,0.532315716221213,0.738520480031476,0.119044872445849,0.170434267662497,0.826022473789405,-0.593780191487114,-2.9918450903935,-0.781096733791614,-0.0737786660376927,-0.791456429424206,-0.0390480063109125,-0.0224351514195883,0.0629907636538452,0.230688424447765,-0.063941810625628,0.581793241115627,1.5504381269293,0.491583115981897,0.388239381668272,-0.292743852224142,0.77003842054764,-0.031626345258535,1.02978459468462,-0.129253070784329,-0.285470341373657,-0.378529248479851,-0.334348286930258,-0.164164023160328,0.0635974215733463,-0.100914795654587,-0.515292452620028,0.198642435797655,0.0875590604101108,0.389683762314703,0.204139942806241,-1.88945908294421,-0.491593053282436],
[0.272109355891571,0.147818850473037,0.396097031860501,0.10839183848722,0.259194297766235,0.15561232008297,0.0226748264120261,-0.0933213082143138,-0.0977451979979459,-0.0611911834170684,0.364754427333798,-0.181595356144667,-0.241152737091341,0.0751030458267203,-0.0918633046949368,-0.0676188866107549,-0.00673601818393743,0.0526057783557257,-0.21239901406162,0.0284254241215231,0.0123936470012148,-0.0238418866868403,0.172822626251936,-0.0473105792248166,0.122680316594532,0.193401784180967,-0.0481830120711865,-12.6300277753375,-0.605075253258779,0.163122270896556,0.147181711851115,-0.0476431034697467,0.20080195632918,0.216143650941772,0.0189364795287365,0.0220745248319347,0.116564267220124,0.148202516091003,-0.492493245512126,0.202962372274631,-0.0871862023356198,-0.0765746405659414,-0.968457160579379,-0.306502438984849,0.742856271725067,0.618593000172076,0.643511095655219,0.0709707521177867,-0.602870460243493,0.14269323345698,0.381341047328108,0.132782316106857,0.305515835654129,0.27000178075209,0.362017802364925,0.228261174457164,-1.55272014285225,-0.495040686393661,-0.222040224145163,0.98583530359342,-0.290819930980726,-0.551743308235384,-0.0879368384659652,-0.594453625520369,-0.169403073687103,-0.024246769944191,-0.964597475558946,-0.217425259822115,0.315867094630275,0.0684650496878715,0.0685915529154123,0.253847639082814,0.246117994329744,0.543587453165317,0.102895733187702,-0.105396044710596,-0.0950487529893114,-0.384918527666573,-0.287808531099537,-0.278938266223328,-0.144190849774216,-0.0444814194869312,-0.300220616748204,-0.569227860786622,0.0348506427624761,-0.0762249469580871,-0.347238164506418,0.135947001270652,-0.0655841449492352],
[0.0701822542825773,-0.00901946486484008,-0.188343805943702,0.0210758008922448,0.149640879385449,-0.138971544701027,0.028269806517119,-0.150211879007219,0.0149915076736683,0.0531677696258015,0.266658741173857,0.125940382250135,-0.0841510856764918,0.0554863287030845,0.054926930357479,0.058890372932305,0.0506916307103123,-0.00418501161202022,0.0612405947502622,0.0430726013440037,-0.088385073827888,0.353634745915969,0.0042186696593722,-0.0124105139251351,-0.104867517193458,-0.743107566149238,-0.323590034729764,-3.10846533399286,0.132003541336113,0.522984825137769,0.514136470942276,-0.208790826816172,0.311678261312429,0.658309637407654,0.0380814875492566,0.180538606153861,-0.577021465578399,2.91487067952053,-0.0613932941845909,0.348368556082804,0.159198046138017,0.332055078643399,1.19024305960752,0.246776176404753,1.33842485372305,0.365553783441778,0.271095523350429,0.276336622886631,-0.531843570484778,0.277799597712493,-0.201952140996438,0.846952773673511,0.399890379239007,0.845237700273799,0.217041719224982,0.315408497567737,0.232326641895571,0.357172957487962,-0.381835673213031,0.206169945820264,-0.672047169995051,-0.197612513029131,0.0150315810671001,-0.436336268476113,-0.386918773696078,-0.526301565890215,0.029252405085894,-0.201527909552075,-0.181720218698705,0.498653879517098,-0.673198095103513,1.30439041126667,-0.572329306762624,0.242660515355953,-0.918160823658763,-0.235359401087736,-0.162638493501052,-0.467517401869987,-0.369644945674088,-0.124669292077385,-0.407380426001147,-0.455020372572118,-0.384153682794807,1.26447967780644,0.661122930017403,0.0991355311964764,-0.665075068223795,-2.78716330777099,0.148420873990925],
[-0.0530193334562833,-0.338133841415594,0.0187257733148119,0.0590833350121735,-0.0146029223107722,0.489309001992518,0.122246172402276,0.0481278812403671,0.167095115754636,0.0525532364840218,0.0318795358537265,-0.0901168731940601,-0.150712364641067,0.227716212592816,0.0667859428843814,0.116169450988352,0.039134999916813,0.290019416516796,-0.046302856021868,-0.157269338634503,0.038334282548325,-0.0627237367818135,-0.0403575098948113,-0.0260581262005133,0.237307059696083,-0.634237931418164,-0.219057455188972,-5.65944955552996,-0.105999908894481,0.360177964417673,-0.111672901378083,0.173129356466387,0.130436128247515,-0.225805559229532,0.13888640734398,-0.0902903081641893,0.326797998043673,-0.0245980603057098,0.36660568869757,0.208717117009676,-0.229913880155906,-0.195946042926894,0.44138255669972,-0.0578701314021837,0.0197325982737619,0.358860397595872,-0.759517936877176,-0.809196180753689,0.670758200752271,0.451309905562485,0.52628809543106,0.155154546411042,-0.598544229732655,-0.331714489020478,-0.161134267802794,-0.0530379565315935,-0.32217284618563,0.305199062139989,1.30793502484962,0.0250176157107999,-0.42395323459723,0.786734862188285,0.624240541493966,-0.405110213679528,-0.360767871396417,-0.389703059766006,0.369408880061936,0.000849298984022617,-0.0346760632812456,0.148828547340173,-0.00646088926650621,0.214234400171354,-0.940799108060218,-0.0933951681010199,0.0157056087241411,0.130301228610096,-0.147565937040928,-0.564132681732171,-0.65443476797498,0.535196516472702,-0.190981760801629,0.196247885910855,-1.40596940385076,-0.571379133896113,0.0808960271500663,0.724099698087585,0.427508255200702,-1.78593090651582,0.363231469784722],
[-0.574429856731717,-0.1691666549899,-0.139381539443009,-0.143209031480085,-0.254724559129588,-0.133123530666232,-0.301526015414136,-0.189380145221465,-0.136966688016534,0.0412333103718977,-0.245623115791649,-0.060241769032852,0.00579621415006715,-0.00162624632606304,0.0133449380397237,-0.0565602842486078,0.00678600580231341,0.17550350326863,-0.0183731075542227,-0.0626408642092737,-0.000429806800661823,0.0481751881962177,0.0185280111096263,-0.0471439482694488,-0.0707834347407422,-0.230179545234382,0.124834247700266,-3.4577635376656,-0.523646634508698,0.147203927466539,0.0369497537215615,-0.0852637222442294,0.113280703643346,0.140113543838716,0.147802415525497,0.00886390068030572,0.0756839953783993,-0.0083009061572132,-0.10003003807926,0.164369170651526,-0.263866045083183,0.232673057577659,0.366251115530419,0.169945578191784,0.3740582616513,0.121040838549998,0.207669696517396,0.233216451645346,0.577060528860121,0.282439506749121,0.0217796844452286,-0.1243841578699,-0.0145710253403465,0.104156595303089,-0.0174819175054498,0.113134815671859,-0.073181209212917,0.245610815488338,0.283716182775161,-0.0936370268437176,-0.311783768872522,0.255537393076043,-0.0192894434170339,-0.0557604132266971,-0.0455669252997356,-0.0566994932912744,-0.114302860697149,-0.245166264258829,-0.407376761959304,-0.308380918168598,-0.38285571949469,-0.0629197126965583,-0.419686696843813,-0.382204247747289,-0.0223560504538448,-0.357265892220083,-0.314622141038456,-0.109907199885277,-0.0735445818045182,0.195355662261524,-0.194384448602754,-0.153182428748611,-0.646962840178965,0.127947047694748,0.376182292708144,-0.143463174246447,-0.16696354917747,-1.03461682681132,-0.338344454158497],
[0.243984397723267,-0.332128474560716,-0.15519570394219,0.0379781272241808,-0.0574425106360319,0.0791010748218766,-0.0657684088367822,0.0697228041533872,0.0204358063155936,0.00950351529800512,0.073537644354207,0.0825243810045464,0.227166524783966,-0.00687318190429966,0.0381868809981686,0.169058703647266,0.0438377769131823,0.23248267412225,0.00968199832466558,0.167112660372553,0.0710671694832192,-0.070676124482401,0.172052148350726,0.0376111822213294,-0.947576595946062,-0.168930427876331,-0.189033191223281,0.495708442273166,0.438997229038548,0.116500759265398,-0.247710536153989,-0.201631810523413,0.113999178943515,1.40193296110878,0.10118872275042,0.611408231773708,-0.561078636805207,2.15291088186463,-0.639130895733745,0.586621936973167,0.139336960938441,-0.156795470513392,-0.265358083488549,-0.179192937289082,-2.31801152170563,-0.674104293659713,-0.103472135521443,0.0866944275239489,0.00887606993187249,0.563764339945709,-0.860577031711742,-0.236892723029108,0.308774969685619,0.30041198385402,-0.195253133687143,-0.194093651078205,0.440661077402948,-0.91835534054996,-1.0203606408634,-0.00962226684715326,-0.225715896071841,-0.351271412718256,0.260699943161771,-0.170392137822718,-0.257231274687745,-0.190891380094726,-0.725815882606226,0.36347161231279,0.413071019043699,0.205082680046579,-0.405058968916511,-0.496129163506915,-0.118367678844854,-0.472073393069605,-0.342489730536916,-0.237448532073757,-0.362709396236376,-0.473350871845274,-0.649930299440217,-0.579635632028037,-0.219269997787591,-0.173200009307616,-0.80496548800255,-0.522715863313375,-0.909645141652406,-0.792845427337197,-0.488794051503362,0.877853534842369,0.181362096716256],
[-0.831037064709163,-0.0873685805400654,0.106524773993206,0.173630452119978,0.251568900087303,0.195044369547635,0.330494499955005,-0.205697005268164,0.0652679950759467,-0.0848207702232722,0.102353802177896,-0.152904071892186,-0.0251488765278171,0.294529345782936,0.155841769932142,0.175118030583207,0.0444405853215581,0.0332524800524575,0.0430939735863274,-0.0985348913792953,0.104442117746907,0.224348526367147,-0.0187229345652766,-0.10926475203872,0.00164589789044577,0.338662781691179,-0.386999031966915,-8.90656686085475,-0.378867724296087,0.126556859545237,0.130319173813882,0.171333603256323,-0.0044723080978646,-0.422975754595997,0.126130816115877,0.0827823476324938,0.5753330383797,-0.529725859274568,-0.253640281890604,0.189639418888834,0.179681175698885,-0.524923238904662,0.624828532708831,-0.782091455310142,1.40878155456117,0.15170681644617,-0.0493151286751955,-1.7477096017584,0.606070422867762,0.313012803919416,1.54550427587771,0.250055694127567,-0.362952389522192,-1.10566537021601,-0.0875844008784196,0.0266935655893722,-0.576538144967304,0.609192571215876,-1.00339935087291,1.39489765048588,-0.232209765373308,-0.179771337196183,0.415534630088717,0.428881110106764,0.936976105645965,1.07847831492989,-0.635797410487353,-0.550729666330766,0.863898019615336,0.071434925991896,0.296508892943074,-0.365524823605405,0.975172179536083,-0.160232542887506,1.10530135231471,-0.206655236916993,-0.111562124965958,-0.131663675251367,-0.566052594817731,-0.35649512987261,-0.160962669942237,-0.294770752114844,-0.311130187025018,-0.69771891060502,-0.821313198591826,-0.997991046261736,-0.562909079430495,1.89595823401685,-0.118831013870392],
[-0.508693014227529,0.0285337870414811,0.230895138605238,0.0818030615054431,0.0223462582938745,0.0189408868445673,0.0102774036809558,0.0740037196724426,0.00852694172261379,-0.117824812580357,-0.0397136236682975,0.0654125906307261,0.0474485676151332,-0.0345483185311536,-0.0263547374718503,-0.00752910349676055,0.132106224089454,-0.00337360204127427,0.128177115723443,0.0177668239877796,-0.0144221121868834,0.0107784176419434,0.16318514642834,-0.130537129240676,-0.0257153228158688,-0.0665205349733357,-0.395091031249822,-30.1325870058031,0.474048216678957,-0.171892269634959,0.00994719012621654,0.0912221894173441,-0.371960675154248,0.328598334153601,-0.253078802605859,-0.22142894600329,-0.187429084667252,-0.0487408740743484,-0.0376218272754425,-0.156410860934565,-0.394674174549601,0.326301192130993,0.0259004473387299,0.353040904245554,-0.467142297434212,-0.0564231656053299,-0.378793978499084,-0.102771702944374,0.00650536992075345,-0.398639761827602,-0.0464163070961117,-0.032102667882398,-0.371030065682995,0.299700983479493,0.177105655191013,-0.173312428756497,-2.99275311479123,1.10266514680004,-0.376067292781276,-0.545750168796427,-0.238706175650707,0.562788089204324,-1.85042123725432,0.0810854525784321,-0.403855419212171,-0.518708359106216,0.95435729406151,-0.00469364572383011,-0.192992716345371,-0.148741435931587,-0.202321878345163,-0.388564281909685,0.428592247437554,0.274115462076105,-0.928460819004483,0.0662492244990946,0.157228154141639,0.0585017579105411,0.0830343090306668,0.265399464864996,0.0161628523727112,0.217993929578549,0.220368023325608,0.2233211502036,0.146946413548384,0.0871722017717953,0.127415761435335,-0.143512380327141,0.211586754585368],
[-0.693349909183677,-0.302065006282585,0.136025650651388,0.0069053956693834,-0.13512830639751,-0.00698295798262078,-0.0166699183967213,0.00241206154896067,0.0473294179852008,0.054207041762203,-0.0800928474950608,-0.0592306552775539,0.0486148742606203,-0.0456409708261185,-0.0270336839218159,0.0312669874921975,0.0123926212132944,-0.326477834789878,0.0613652975822855,0.081182617606811,-0.17459730173993,0.116828516829711,-0.00855249136685605,0.0658418254849119,0.0096349434113642,0.185950973113125,-0.354184044104513,-14.2614872131334,0.0326802620883315,-0.0760907540718571,0.566226322013206,-0.0356325459811322,-0.33761249468685,0.299527214999164,0.128520227640377,0.339613130523874,-0.289593337102039,-0.838771129636,0.0578468319426818,0.250960811692429,0.102508649768323,0.145381850682476,0.215385585263135,0.159363908132049,0.872113847280107,-0.295387531844757,-0.217140523709163,0.147523124779454,0.536225364259013,-0.183909283169041,0.626467492263669,-0.136765486579744,-0.279543856293338,0.420300146953039,-0.33327403699432,0.112117352154019,0.621818557419123,-0.678830394421694,-1.05765686994407,1.19185816143502,-0.624708307700771,0.000382773800552448,-0.135385000669577,-0.279292999258492,0.748684814602513,0.172469103012031,0.993572288572048,0.165516496797945,0.464080267582277,-0.389028016344464,-0.185476452576911,0.416540982161544,0.254904213666098,-0.924488520096643,-0.267222664327376,1.89722484025484,1.94209566171134,2.02699798502072,2.25904200208285,2.26979842301871,1.65698615814215,2.11157564316682,2.24486194404767,1.73248906357612,1.38591252762776,1.76366430224886,1.89374781038359,1.30039435010309,0.138929923590111],
[-1.05530981564459,-0.260040055013491,-0.122790851545237,-0.0986462342906593,-0.13191718400707,-0.168912688791994,-0.647128231993102,-0.012838756426046,-0.72904835164229,0.0168955887487978,-0.206522378891856,0.0296484129608604,-0.0526187933846003,-0.28103719613776,-0.563320219397522,-0.558026619823985,0.0266594281371452,-0.668188228518049,0.0552925054462604,-0.0584134492103945,-0.0503161902159289,-0.350526192495376,0.0937457918482747,-0.00939668625495965,0.0897739073919173,0.44058937498979,0.351228321889722,-1.76072843425353,-0.50864114470786,-0.120326104115819,0.239243553193421,-0.0201034133953829,-0.385537304042453,-0.052598801482104,0.199928783728161,0.0238649686352436,0.00396395777062845,-0.0995269891185773,0.049542834919131,0.0907050343299712,-0.09904739912222,0.119291524994289,-0.196425324184359,0.262477233138854,-0.0608672615269678,-0.170460285014797,-0.204178326645869,-0.201052969212823,-0.0784807405196554,0.156045910959698,-0.194529502414902,0.295249993967317,-0.0437907282469439,0.0355829581173116,-0.0195210006563424,0.137266448616881,0.281250326409077,-0.0985869383869322,0.0255016925991032,0.360305158984042,-0.329152210793974,0.25904174771773,0.0656410334935172,0.397599658704363,0.359608036327654,0.198659939680191,-0.109961699588666,-0.219167639345357,0.123037382107436,-0.0190996073518552,-0.382035704154837,0.271199301121306,-0.417072742220393,-0.129897220026725,-0.809569768623775,-0.500558076983557,-0.0333720880868746,-0.0614023088351847,0.288877975387911,-0.522264381753098,-0.143025990727559,-0.188031243685677,0.452420494430358,0.052602602770054,-0.151475011421927,-0.207516781547904,-0.142404545195281,0.102837855552919,-0.132995258750509],
[-0.168247494490283,0.0298881998975249,0.200300602496526,-0.0963040394384725,0.0314080598728466,0.0879003442155797,-0.0896203264502353,0.149529160878877,0.161644084871729,-0.0297013404187133,-0.00649472900216201,0.39274980882983,0.0291522355157837,-0.165775285886858,0.113041549534648,0.196311690553152,0.00298329177700953,-0.21788643458868,0.0595796270071521,-0.0517873489996583,-0.0990482467847677,0.112180764876831,-0.00513639567092139,0.0108499728239693,-0.0105479562745254,-0.0729151104081979,0.692762912096935,-5.57102727897362,-0.333823638634386,0.293432747382169,0.134339297193889,-0.0671155189189902,0.185406657260875,0.658995074838529,0.151685189600179,0.0792406175486731,-0.0431393107469042,-0.062920647187955,-0.288893704154143,0.168233297920415,-0.0325991952818092,-0.267744708728595,0.699790694756477,-0.0423704271826549,1.33386100122851,-0.184792913063081,0.18715099228921,-0.659089745068792,-0.218362968849444,-0.0295730337604539,0.445170441236882,-0.112648378883062,-0.344403906533309,-0.376804428946476,-0.0603871475229583,-0.0191648679895433,0.752148040884962,0.661584803833507,-2.04537826675008,0.501847042686746,-0.288584465287026,-0.161484050553058,0.241126748733085,-0.0470431193237601,0.506683242179343,0.42549020280084,-0.452043962580702,-0.283920588646842,0.23746086183368,0.326682270435169,-0.958076810752463,0.287953247095472,-0.0203485989872547,0.464639199148425,-0.783836095085275,-0.475117559868041,-0.142734893903488,-0.0845629596949384,-0.325870507530869,-0.872711522315653,-0.454071470687305,-1.19136407703577,1.50325633319532,-0.255569334811944,-0.133482084025892,-0.441992859604353,-0.401595785247544,0.411938396638529,-0.00677790672540973],
[-0.860490156385713,0.0861449413336553,-0.149108382561551,-0.0681970065883772,0.101585624613129,-0.212171358530997,-0.445969719743675,-0.273440416673839,-0.171330303940327,-0.00451672154116561,-0.194943949832273,-0.123635589624828,0.018043968440013,-0.127041957658186,0.0791472416775934,-0.279716051779887,0.0570517904165504,-0.0750224524969852,0.0461085701737275,-0.115170043490003,-0.0696123574913775,-0.19345079792449,0.0155209411070759,0.034373971019754,-0.272029868689366,-0.330878395552697,0.392555437215017,-2.08095567319841,-0.267941827349676,-0.209522151741946,0.059923307542473,0.0595445361759714,0.0472708181405405,-0.145777340526394,0.23062071415919,-0.171944166130345,0.0637388949412942,-0.115372317121953,0.0265008043054949,-0.0159394863371897,-0.273631049158258,0.0716559660892459,0.0249351124899679,-0.00754527305174887,0.326208816149931,-0.0682340282626963,-0.343550450436782,-0.08328592355056,0.0491780083244288,-0.0139529698149363,-0.14129277518966,0.100575894935889,-0.319495705665738,0.0529171785518234,-0.078042724341932,-0.127718837245651,-0.102312116078319,0.0132667621981208,1.12585207019343,0.252434492028351,-0.63235781962045,0.545118402868767,-0.00155577379953767,-0.0241340607864821,-0.200907577659414,-0.240526957200993,0.21300392526004,-0.230979417815665,-0.298376482795522,-0.255630393068269,-0.211914806846023,0.355017883148117,-0.494297173886582,-0.269457738600139,-0.277526103648022,-0.556979274769068,-0.0905551988050087,-0.115098900018825,-0.188292441198784,0.252826013016748,-0.330412777527134,0.0316549078309204,-0.875998281071722,0.455839075428106,0.356810885195538,0.0493257644677628,-0.0936078755537267,-1.4827207835119,0.773701707089479],
[-1.94198690942464,0.149406558604627,-0.0508930899124874,0.163302241890664,-0.107783445331338,-0.0191426687219554,-0.149143475632276,-0.193100655486838,-0.0262154511408312,-0.0218957817633542,-0.361132288180769,-0.215693434990719,0.00349153030979985,-0.0316482091348568,0.0452186268033863,-0.096251686106756,0.0191399399254881,0.188943546001826,-0.0488833803085511,0.096006293653933,-0.0194270996187674,-0.224040210659494,0.0245595953837036,0.0101466706649569,-0.0336338206746796,0.249280659345761,0.307723771392428,-4.26210890751922,-0.179410435850963,-0.179583568734871,0.36678509138292,0.142230988189978,-0.819923361976748,-0.42227303708367,0.250728108981337,0.135909499566228,0.282002757157032,-0.510121152058244,0.0471405944198568,0.178884915432324,0.0614979129974415,0.477167581798926,-0.575804923122612,-0.290814176791476,0.16677942317168,-0.29073912179334,-0.0831183708838888,-0.794444634089439,0.327453122551259,-0.0290264060274903,0.224137553715418,0.228112306955675,-0.074014912144037,-0.367182569573491,0.0293336368017706,0.266299717896427,0.254702459844111,-0.0572616842525746,-1.0829705586322,0.441103104504144,-0.0409554854843737,0.0292491724693402,0.386328257882156,0.672120670782482,1.02804965088032,0.92685212437528,-0.12937418444833,-0.0769338165703274,0.437148153331973,0.242298042162829,-0.446096803551002,0.0722208833342444,0.101997573661104,0.00511106529131227,-0.654837380672333,-0.254619893956043,-0.126936308501227,-0.198231143961163,-0.196370706208972,0.0959762769330631,0.00589273279486482,0.13729901157375,-0.810190906920895,-0.731722325411617,-0.955931282618286,-0.765265254293169,-0.11752606397211,1.9442146588539,-0.400219991313624],
[0.0408608729754911,0.0265356562614954,-0.0280268064556849,-0.117823507175949,0.19961540734903,0.270151900448923,-0.0158871551210786,-0.11992749750903,-0.24186766523022,-0.0117186744627637,0.122782227885149,0.158280291492114,-0.138820003853838,-0.000522680306660021,-0.262182813748022,0.0085990694266535,-0.00639841529095956,-0.445314992870637,-0.0664438607860596,-0.125435274947537,0.0132011862523188,-0.23246227492328,0.0731084124398178,-0.0132421142456295,0.521127744107629,-0.466855131854016,-0.0282207931830892,-5.40111947805683,-0.911729646934217,0.0546553761572808,0.074894024534385,0.0962470040880047,-0.21012388596729,-0.115658386692994,0.217414069830198,-0.045058925370828,0.221406154002214,0.22737727002388,-0.164514514721702,0.106596768913695,-0.191396046688671,-0.0776055762252799,0.290703245515055,0.0861810057788169,0.20808613183191,0.0486468929812796,-0.414328916115662,-0.552028545496779,0.288510228499839,0.481914168491003,0.10399112263442,0.399538476663416,-0.445251560889937,-0.130201251747528,-0.190932332085777,-0.160622882643641,0.242708571693745,0.823640739974577,0.745530593345643,0.486763201204937,-0.514943661933077,0.892457623307311,0.348835666461852,-0.0503712424544325,-0.247514921169139,-0.240966588424391,0.0956858913245469,-0.507874223778897,-0.294676781942984,-0.0662988980591536,-0.430571503146309,0.39922947839983,-0.723445481987286,-0.0938614379834494,-0.51839187223167,-0.0876435689314873,-0.238796171611153,-0.387379760264354,-0.340227857024488,-0.466001256084471,-0.0340478806441704,-0.499964917193831,0.0577300586527808,-0.731763118174599,-0.109101173922132,0.505359550209711,0.355523379619188,-1.0681601400467,-0.779576093623638],
[-1.69773554447754,-0.177295322692833,0.284040932486724,0.31826430132152,0.321670019571862,0.170384769171696,0.135207017347685,0.147911349878855,-0.0586977912200044,0.28127600040962,0.223702402588741,0.297176948260368,0.101334366612655,0.157420250029998,0.208069044269625,0.123029294361325,0.0065007025107957,-0.548156605597898,-0.334922762541548,-0.140941289660243,-0.0798593032428047,0.166722632170719,0.214884364869642,-0.0542115767250623,-0.049003200011884,-0.416169762095634,0.113057146544078,-4.89593151099786,-0.0277171138636673,0.893790842531229,1.24067608506267,0.107879684076637,-0.653230032533668,1.17866206175338,0.256559655814212,1.02285677382678,-0.621175917125395,3.43898895060896,-0.0974727883703795,0.832128760744962,0.212310522854593,-0.233746411855107,2.13739686772273,0.360841269099554,1.69170802479665,0.337243059098626,-0.508082945444424,-2.18132659185814,-0.597920905234725,-0.113843510569414,1.4063885727838,2.18202345053148,-0.410029813248487,-0.510608399323501,-0.325698571852184,0.0211013084446375,1.34782861161066,0.0995796001641174,-2.87956323003698,2.07879491541284,-0.15745514045911,-1.05229403219371,0.975934230723788,0.783645722759845,1.46796181391549,1.18059720451708,0.0208087914593594,-0.620221425206435,1.15207433393238,0.939185348235791,-1.46337626885618,1.70349608191775,0.405143822690297,0.214010125675525,-1.65170830829029,0.219377195254157,0.382894302417093,0.469438398165607,0.546310476553205,0.510039833694326,0.40984168005407,0.338056867387612,0.290392260317252,1.46755714425938,1.18619727988042,1.30509179766849,-0.337085295748113,-2.4628422659762,-0.697327915076412],
[-0.543284769485913,0.0962294715889678,-0.0906162575857758,-0.139157965736062,-0.223584899457704,-0.189111990691376,-0.0595933782894377,0.219884325324772,-0.0335485995051444,-0.0458228643582374,0.291656505379434,0.0308059458115356,0.119587871403019,-0.19831967332552,-0.214120821358579,-0.290467829583684,0.0155030022923212,-0.31829468715182,0.0568406236497174,-0.0215033949773963,-0.00551589609214776,-0.0549637366287105,-0.0432319892499742,-0.00987752129905706,-0.32011430676257,-0.057613740838135,0.0937862756918955,-1.57159684120225,-0.136986955187441,0.167699068064988,0.0326930311823603,-0.0359709901325528,-0.02701209571604,0.486101930331247,0.177121583180226,0.257330740229345,-0.125066572501241,0.767958926967353,0.104634001488207,0.300698414844704,-0.0540556963812126,0.150350075620646,-0.000860011261294497,0.478547546099805,-0.296690941188768,0.00372778087227032,0.320343391610154,0.115307841725229,0.476977921577613,0.278017822918361,-0.364781050358251,0.043332699050586,0.119928062901079,0.196484455147297,-0.0382790996391548,0.0363768710556252,0.563696154980586,-0.194160594467151,-0.930582127882501,-0.0661908885292476,-0.0700653934227446,-0.361286762078084,0.25460507546024,-0.0467807382385276,0.0552759799545593,-0.0589739804385555,-0.287051783240153,0.373229994500022,-0.134543563338968,0.33464905956412,-0.603998385290643,0.0705257603478601,-0.217827470836599,-0.00216810182076849,-0.630179164330121,-0.257298736135597,-0.290538598713271,-0.0857052848893878,0.226530047816491,-0.0115814330695984,-0.113082974112087,0.235028657078775,-0.822181267704538,0.380563850857262,0.120149055075474,-0.87378798421628,-0.725071807331339,0.680122854949838,0.0681110146390479],
[-0.775866286356966,0.164953684626621,-0.209841014875536,-0.0699981919272634,0.189082448892889,-0.27656688822556,-0.366512368929059,-0.0612105322570933,-0.248322295307332,0.0211346030002247,-0.214324808405531,0.0161293072430107,-0.155413362625649,-0.172012472007578,-0.248845007244658,-0.308301893104223,-0.0125760944964463,-0.398862698883156,0.00486723845207035,0.119138402420908,0.0195458658088041,-0.196694621758459,-0.0280788505735363,0.0514335890733181,-0.48550053279475,-0.30362425105881,0.287506234204705,-0.969897537283084,0.274118734897371,-0.16418497240536,0.18558158040676,0.0885112499183462,-0.0261720690737863,-0.200464703285431,0.299830983367539,-0.356914260153223,0.29205300421703,-0.531480561353068,-0.0858022648869482,-0.118550971429007,-0.342930405742048,-0.0444859169830118,-0.25777541833107,-0.039486195727106,0.118901684784059,-0.362108987197679,-0.445927382186741,0.00642478210007101,-0.704319309137318,-0.457713897333111,-0.277817773316372,0.081760362228893,-0.181802386793458,0.183694078152403,-0.199804316578314,-0.237803793402694,-0.108459167241868,0.0137417987518754,1.44700994042397,0.48889920690279,-0.826826176777829,0.418045784842965,-0.225535995373332,0.100895030281872,-0.270029091504374,-0.292539127112049,0.385573596875322,-0.243400521415804,-0.00967260059421043,-0.049144580962983,0.127404350357757,0.857901483912983,-0.253877614482335,-0.215850456553693,0.109181534102246,-0.473129182854744,0.0949218803378238,-0.17370074164282,-0.212730657473034,0.0512179950868001,-0.226868604052727,-0.130465299821236,-0.356448200938213,0.892828682198501,0.706111630278825,0.220434988142406,-0.28103973072006,-2.22277847980149,1.07678368886346],
[0.700071899062265,-0.0318785972777479,-0.0701093266794105,0.029246340739651,0.0543192761399065,-0.0193564244651582,-0.0143948896387197,0.0713769738356267,-0.110196676782499,0.0846727001139177,0.0955635855518603,0.170137036610458,-0.0288137549156467,-0.0482638397280653,0.0496630762197808,0.0108360259145766,-0.00237404763609612,-0.260887438123451,-0.188350442526682,-0.0976854525287306,-0.242713982468295,0.0517306738375553,-0.0888452324849035,0.0300766791715742,-0.0898845108856282,-0.387640996375745,0.739332058771053,6.51024586916516,-0.106007979964818,0.127268740782567,0.503896145416601,-0.0220508560912104,-0.419238014524899,0.847213229239626,-0.241061453866693,-0.29832011852634,-0.72438045362196,1.63404487298719,0.0109146061832475,-0.429327584097839,-0.389452729433173,-0.151676388502864,0.767095330166453,0.21734996095483,1.85780473987508,0.0811821049815993,-0.738030013942662,-1.43888386002568,-0.785099431404223,-0.814352987448845,2.01740086764731,1.64028388071316,-0.393584147412943,-1.04167910209214,0.13453271001931,-0.199362833673746,0.597062744491088,0.0558393821808565,-1.06983781039139,-0.491883069229615,0.611042534533257,-0.382557241844464,0.258273770663982,0.415749115931862,0.352562588135031,0.847009619974972,-0.260040384141358,-1.25204758324426,0.0939371513525802,1.69458409191049,-0.759416035370871,-0.82000047306015,0.436051517234083,1.98236782690818,-0.997921205508393,-1.03129056597401,-0.890062132044564,-0.754379294463447,-0.760969749288888,-0.758135130818881,-0.795634867459699,-0.717893409922438,-1.12429453638683,-0.402251287531934,-0.408877147939697,-0.780452076843638,-0.779807381837863,-1.07732654740012,-0.617607952825628],
[-0.288151756340485,0.280993941785211,0.0679227723220527,-0.221762641374436,0.29261033456034,-0.148291625363834,0.0612775286330455,-0.21232548559399,-0.0460661561062011,-0.0066945700131543,-0.511749082703177,-0.25898714945612,0.132649623246163,-0.002431300954029,0.198863383400109,-0.105123438170325,0.04465856305612,-0.0659947726131954,0.140735254222418,0.0909707193999115,-0.152746166358606,-0.0750544106066988,0.0580929942293819,-0.0197515112173478,0.585864658238398,-1.23643613670803,0.104476622960545,-4.73648431605964,0.974439530926647,-0.067144738858142,0.269759872871275,0.123987709688584,-0.0745832867618227,-0.423034574676975,0.259765851181621,-0.43042191262582,0.420326537662816,-0.922129081535334,0.0584815455233113,-0.0805280015841467,-0.591245101741882,-0.180767278955328,-0.0812599950270346,-0.342382533450041,0.982507142681299,-0.387525360971975,-0.896942693692709,-0.684926239679068,-0.039211893604819,-0.140569940103842,0.990807004245189,0.654106599451281,-0.42052118547139,0.102268954112226,0.0723078814613237,-0.0429664298997061,-0.525492888059241,0.104994609026212,2.35081226202559,0.106860860677085,-1.06528865148438,-0.0467204713436374,-0.588895750876882,-0.0934317327070336,-0.602472424665263,-0.724102464591037,0.695982210226857,-0.538289263966693,0.0939535233497102,0.125951481354931,0.69389722645226,0.815652246991385,-0.850851635261388,-0.00424928738982227,0.608134879690939,-0.168604883189853,-0.504328198218082,-0.163586528214342,0.206926184478737,-0.231745249677722,-0.212832612239154,-0.0544025994980387,0.137477332683871,0.703183439538269,-0.0629612058556627,0.550331522045036,0.985418388817286,-2.62739384640333,0.0901929370775839],
[-0.342085347335845,0.168907030805486,-0.0276110495680675,-0.0759649940719852,0.164000123636407,-0.0134207573184926,0.489279282040948,0.0932886681382371,0.26664643919863,-0.0378980382466148,-0.159625688347578,-0.0595372678443564,-0.00396068847433686,-0.136336238886082,0.272059364230861,0.612155679311364,-0.0031060094175393,-0.174135388374229,0.0212913675309333,-0.0731162271810527,0.158347727413684,0.137389750485071,-0.0292911083902715,-0.040785408632212,0.282256216313196,-0.676827842313402,0.0885421984643491,-2.5147425249077,-0.302027539987735,-0.0436518915174271,0.204012269815088,-0.0492690726190848,0.0358953437324217,0.0371786340934872,0.166544783919278,-0.22256285187976,0.0389811706221502,-0.317586775971375,0.274456711071952,0.0130299019355645,-0.270157671486952,0.0245157023988457,0.661053038682721,-0.103463090678914,0.494768067255122,-0.0461540125553885,-0.0193601550211887,0.207955924341906,0.397334722283921,0.066637064648605,0.0957500813867859,-0.0347217801637975,-0.451606461353685,-0.0366155927004409,-0.313930698679073,-0.271565703683595,0.140564496069344,0.478948586341785,0.18346378680444,-0.0156098835356026,-0.582891999963454,0.300758563871328,-0.0804804199308774,-0.182843702559539,-0.21426030792628,-0.286580380024667,0.229350900316649,0.0742086953277475,-0.499572508023037,0.1599812238849,-0.482630175842096,0.130961520183995,-0.575279904015671,-0.218435758455225,-0.38733994135254,-0.624238255920477,-0.319706243681261,0.189755689964925,-0.336348689594958,-0.0989308785604708,-0.302804162642274,-0.310687977182493,-0.532886854805156,0.386900321300717,0.52256827320398,0.151756702087272,-0.825824418765404,-1.41032713970819,0.350981641692008],
[-1.26037288130216,0.235266361904083,-0.0355652327610198,-0.247761249027355,-0.285610033492891,-0.246415631036106,-0.126115990618485,-0.109155556839962,-0.173850882707557,0.0245280817491527,-0.295040873272981,-0.00731116162799157,0.16934783971733,-0.197421292991476,-0.36162937011967,-0.36336459262988,0.0340696994448938,-0.0128886301329647,0.062467360845801,-0.0519158716864136,-0.0238058558365015,-0.161131134196151,0.0401638318317439,0.0278213540390989,-0.766229910727427,0.0166575799026663,0.66849904379097,-1.43685408400307,0.232209283930874,-0.0480719276582635,0.272917696193029,0.0120702847089358,-0.0537367082459627,0.157188898271622,0.185436100904365,-0.00179371664581123,0.0145066599152759,-0.123575290284971,0.138851124805199,0.0942877934995481,-0.188983508799168,0.11347659119544,-0.271080892717664,-0.109470727215993,0.22891666034581,-0.126211296999118,-0.0120092367392549,0.0243968008823451,-0.160984647059759,-0.341965053531932,-0.283394098896203,0.0315044718499067,-0.0429413791819463,0.0248626618695749,-0.115628062437337,-0.102775995766271,0.154875562363337,-0.337926049140856,-0.290248751040122,0.0625178681705715,-0.332523871415917,0.179207890500413,0.0512689557726409,0.246109613337492,0.226792747551812,0.220633734146736,-0.0601532010076586,0.0142681588337256,0.0248911397996594,0.161829036397599,-0.252225452942855,0.37074604298185,-0.236150891204886,0.0140965799108897,-0.471749522906623,-0.741632745280121,-0.238472769640234,-0.0132941915387256,0.304115943300924,-0.132120145717132,-0.247820517988246,-0.313812108649451,0.127612354946644,0.998179355572541,-0.00772367230697209,-0.691231285093055,-0.775434482747138,-0.254902148547171,0.482373070965984],
[-0.677363356440853,-0.0805138045570705,-0.141412115119895,-0.0850532812858028,0.320194310407417,-0.12626048866759,0.129706097013804,0.346090306840216,-0.173865996717076,0.0510749168362771,-0.368143360585455,-0.0201640568991412,0.126483682792378,0.0141877343070465,0.303931692689061,0.140647666329112,0.0212345918565593,-0.179412433691495,0.122004035110933,0.168127545172092,-0.0164325123553589,0.0834238566826314,-0.0775274570464501,-0.0371709541382039,0.479583809962726,-0.175085823278294,0.144834559165847,-1.48877861704857,-0.00333522549435277,-0.0466373200399741,0.401390394780402,0.0873911454969484,-0.313573656230018,-0.333728244746996,0.161230394056741,-0.301960194097457,0.156115948231919,-0.747089436965952,-0.09655917362049,-0.149450700872875,-0.367971548437601,-0.018945264099428,0.818751477747229,0.183331350136094,0.84792558311814,-0.128748065103991,-0.154392832062316,0.0644064137392708,-0.0359123440326996,0.0450887491794448,0.164841233642202,0.25807505356073,-0.364342112454182,0.0750122734565236,-0.167043132890504,-0.239834442500774,-0.108832653712527,-0.0740004745907158,0.201787905364809,0.0726470829910685,-0.689947474764827,0.115610399573092,-0.325798474684347,0.199329164497353,-0.0554036726562063,-0.294724220983822,0.388979607888033,-0.243556924438588,-0.277490005564082,0.039331767497007,-0.366466340911518,0.414309289386773,-0.49564827578452,-0.33432438209563,-0.677290976746375,-0.598758924737016,-0.0442880025459189,0.269657982214252,-0.0113928423201526,-1.20936168295184,0.0557279569354733,-0.267536248114888,1.15448249901708,0.273685053900184,0.641550055922167,0.429280446064089,-0.367063194955914,-1.48119406188542,0.430012870505255],
[-0.179322245755626,-0.0950868056951754,-0.195529793499263,-0.136423623007099,-0.148447988297063,-0.0460346814145862,0.0137181746656758,0.139072965398059,-0.256514404939367,0.0155733603188721,0.0492605934048056,0.0153197567219195,-0.140704625433978,-0.0128528646502336,-0.355082541698456,0.0632729917555929,0.0486805834312358,0.0607342172028409,-0.00895962114553791,-0.0915410436518231,-0.0390042371620097,0.0612904764854884,0.0146958071474553,-0.00775629303080753,-0.343497953179649,-0.0726575474604206,-0.42682034315031,-0.664925590033497,0.100297175333601,0.373082448493589,0.0110557635923956,-0.0924156836393901,-0.110736575058455,0.106161350756391,0.154192033103256,0.0737381608946102,0.149932007612819,-0.110393766836195,-0.144838836392494,0.124292054646642,-0.0188757987197123,0.094288289313197,-0.15651741476635,0.165131186046751,-0.871538098050096,-0.139799582824507,-0.0860604237712585,-0.119330759067686,-0.0253512960923565,-0.172852502294998,-0.574683210067108,-0.1901783466462,0.020546798886108,0.0768390858675786,-0.178326772405866,-0.0167690692321227,0.335716195859424,-0.603913111322143,-0.153283309328956,0.161128410926709,0.0276322968916318,-0.45647241623038,0.145553544015707,0.0166402882339481,-0.0290066698207563,0.0134146491696972,0.0323263004867079,-0.0356561101381377,0.173075527375644,-0.181970212072144,-0.180575521493662,0.20967419389563,-0.0806193609926113,-0.07741966197674,-0.16639988770097,0.264193069467904,0.00932664520243859,-0.154729945083174,0.0201100426320638,-0.066128104812083,0.25347782796573,0.241096150459648,-0.288357272428092,-0.218417717700896,-0.0182137925437317,0.10483067469845,0.29645375548214,-0.00169006768309426,-0.714522036628101],
[0.0472479430918416,0.0333352916102843,0.216917965653071,0.314514594931887,0.0390169123588444,-0.112135853013782,-0.147109289695123,0.00422546703335256,0.220592097341583,0.0398883723317494,-0.684890126794886,-0.259205495558842,-0.0685229470825637,0.141095878927375,0.00490809082603812,0.102283005776679,0.0760203899259482,0.473402369762329,0.110235787820973,-0.162289224178918,-0.0455092146329419,0.0289628000349693,0.176772185240004,-0.0249060005464387,-0.052817710542846,0.142947652985663,0.50898160611272,-7.29476201754264,-0.157451267738364,-0.309917819254168,0.176920888174608,-0.0521575910262524,-0.105414309816135,-0.238721768181803,0.0666534287814131,-0.0720823912336715,0.144363054828568,0.172490594802739,1.02906415156651,0.0539573962374647,-0.135300207010692,-0.504491477818907,0.436957557089866,0.0109543516101678,0.796303114233502,0.262007651631043,0.208335717332198,-0.47873698850133,0.242707440504423,0.359604342054007,0.742180854902646,0.22358828724176,-0.491332887907938,-0.536117224911264,0.918169519044153,0.655540865899969,-3.04456818252507,-1.77092335455869,-1.18381032446451,-0.0444003174038652,-0.529338325127637,-0.63621735318996,-0.216790966144238,0.127983677371729,0.0429114636167245,0.00800464241345899,-2.58754109899097,-0.159058987910857,0.521093347644275,0.146191363690501,-0.151086512403323,-0.360395652100935,0.164618975720769,-0.0761626851502982,0.177301067537797,0.539333773334365,-0.411705090592846,-0.618016419637855,-0.189704013030081,-0.252876282052337,-0.286149714842837,-0.532986679202891,0.34250847654256,-0.29548583076623,0.0727802952134673,-0.179756215143879,-0.305387901303539,-0.306564634623744,0.136636833915814],
[1.65389589112413,-0.0457986010982941,-0.199497046450137,-0.0363432081805522,-0.0814140581802127,0.0384630241998522,0.00318900585475855,-0.235511390986,0.0534152432833321,0.24864193909099,-0.222574155097834,-0.113259633974754,-0.0728485775158663,0.0242933597874555,-0.0462492812676793,-0.0879680099851983,0.0343924238693018,0.193133369907332,-0.0186714313808584,-0.00450896052558032,-0.126576037745645,-0.258777214202619,0.00669020618406812,0.0758704384821299,0.102225865855644,-0.822233508819307,0.22609106972406,2.10383204853718,-0.0550259008011574,-0.752354755986439,0.238013421283356,-0.0125068584738269,0.848334423283029,-1.67221467061518,0.267225834905843,-1.80702537196577,0.470262977852724,-1.48152709568653,0.0350368729548894,-1.08951496791097,-0.783844424280315,-0.106537032769015,0.659735485870336,-0.558472938540935,1.48881517335946,-0.511195880434598,-0.16674558614925,1.50160683481035,-0.324318029456027,-0.436888210411266,1.90531997352833,1.07930865050953,-1.33032647020858,0.656764305277797,0.699148704377024,0.125655561030889,-2.70325018910614,2.12469606126393,4.19757994316649,-1.48043964920934,-2.85625902046081,2.41420949828858,-1.83159811535187,-0.805412214319542,-2.36271964328385,-2.38017338546626,1.3768071413103,-0.0300542422605482,-1.03754701761442,0.582644512616105,0.969530695217655,0.450417768986458,-1.86846519912314,-0.880888156648297,1.58670113492262,-0.449971492396216,-0.892995817799104,-0.929764818100679,-0.911633211684206,-0.864569007265025,-0.651848055056636,-0.794824260236797,-0.703643412979434,1.56937491255461,1.22637391169865,0.694315905677877,0.0493929952450723,-6.44201225060405,0.536937294125204],
[-0.621415813130834,0.0661389142432988,-0.122864802758189,-0.176113481266162,-0.215247125622258,-0.181320581008255,-0.00427951629414045,-0.0205112175425536,0.154988793836895,-0.0558633392802013,0.0907359906549024,-0.140184243531199,0.0460076327451355,-0.203398380521207,0.142565917990775,0.103515140059163,-0.00898910741267741,-0.270879293155564,-0.0220950608793634,0.090492361548232,0.0403062731522056,-0.0298045103744261,-0.0399309243737191,-0.0354910306926322,-0.0155737215496716,-0.152839790977289,0.0773273398781,-1.41801715672328,-0.35157984794657,-0.0620587455033097,-0.00515711455295868,-0.0999999400551944,-0.00436222457032273,0.351330434788922,0.117145012940657,0.0613408568170146,-0.156043647478069,0.224008419011037,-0.0597386115893863,0.15484997102822,-0.180433415416209,0.319358522274999,0.286578093569972,0.0490754288144892,0.0798228886104512,-0.175262052117057,0.418856948022431,0.460593684127354,0.358264575228691,0.176473954133907,-0.381314230708318,-0.270287091864332,0.099958008598027,0.296034605877613,-0.10734188549975,-0.107264280176634,0.303500586564875,-0.0600416967036311,-0.239587269994191,0.0738825223252194,-0.223251932484369,0.0776673977773674,-0.000521018668051115,0.0157998785331187,0.153246065407782,0.0818053096606437,0.00839553113847706,0.266509862939663,-0.475018717772636,0.0566390724985925,-0.615983733594966,-0.058314831138544,-0.292492974717144,-0.290172201096798,-0.463822281682837,-0.834399497334173,-0.241383409894849,0.168094766103005,0.0715787342535464,0.270344844952808,-0.257651014280785,-0.0902044682991547,-0.81155800042541,0.394453577684039,0.30600249998728,-0.40557457649111,-1.10329497041069,-0.195879637860327,0.0251339390573808]]

NNBias = [-3.81801205125905,0.635073432310189,-0.927315216320636,-1.75701291111713,-1.41287471760515,-2.08508567738106,-1.29653752677098,-1.15136117527802,-1.86316228212421,-1.99674207738639,-0.468813343304098,7.55037800794737,-0.535889058504504,-1.15555167407057,-0.917632531345475,-0.739921502057983,-1.09240419189483,1.50284426157577,-0.892337908599424,-0.66371736806918,-2.18479179419075,-0.927049395633377,-1.02993896417004,-0.792858021710501,-0.818976936355358,-0.0242236908298587,-1.06408991449678,-4.35756254710854,-1.10175836391695]

HiddenWeights = [-3.99930642115496,-2.7264113780783,-0.432240027597847,-1.15895564898658,-2.50556741124323,-0.618489726541215,-0.808861613467262,-0.679877767282853,-1.23191205909307,-0.856633343672475,-6.94840265241881,-2.87898818716362,-0.576246629670126,-0.538822630968929,-0.636515845193935,-0.764642991777678,-1.0063083051819,-0.934564278629149,-0.575302643936789,-0.715640669444299,1.34756362845961,-0.683908084460194,-0.55868711524906,-0.550531131230242,-0.640870671412236,-0.883267302040309,-2.26725616637394,-1.73962555121171,-0.442169237827234]

HiddenBias = [3.34233150862306]


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 num_labels, pos_ids, percs, weight, pos_bias, extra_vec, use_one_hot_embeddings):
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
  #bert_hidden_size = 30
  count_hidden_size = 29
  count_input_size = 89  
  with tf.variable_scope("bert/cls", reuse=tf.AUTO_REUSE):
##[30,768]
    #bert_output_weights = tf.get_variable(
    #    "bert_output_weights", [bert_hidden_size, hidden_size],
    #    initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=dtype),
    #    dtype=dtype)
##[30]
    #bert_output_bias = tf.get_variable(
    #    "bert_output_bias", [bert_hidden_size], initializer=tf.zeros_initializer(dtype=dtype),
    #    dtype=dtype)
#[29,768]
    bert_output_weights = tf.get_variable(
        "bert_output_weights", [count_hidden_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=dtype),
        dtype=dtype)
#[29]
    bert_output_bias = tf.get_variable(
        "padjust_output_bias", [count_hidden_size], initializer=tf.constant_initializer(NNBias),
        dtype=dtype)
#[29,289]
    count_output_weights = tf.get_variable(
        "padjust_count_output_weights", [count_hidden_size, count_input_size],
        initializer=tf.constant_initializer(NNWeights),
        dtype=dtype)		
#[1,29]
    final_output_weights = tf.get_variable(
        "padjust_final_output_weights", [num_labels, count_hidden_size],
        initializer=tf.constant_initializer(HiddenWeights),
        dtype=dtype)
#[1]
    final_bias = tf.get_variable(
        "padjust_final_output_bias", [num_labels], initializer=tf.constant_initializer(HiddenBias),
        dtype=dtype)		
		
  with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    count_input = tf.reshape(extra_vec, [64, count_input_size])
    count_input = tf.saturate_cast(count_input, dtype)	  
#[289]*[289,29]=[29]
    count_logits = tf.matmul(count_input, count_output_weights, transpose_b=True)
##[768]*[768,30]=[30]
#    bert_hidden_logits = tf.matmul(output_layer, bert_output_weights, transpose_b=True)
#    bert_hidden_logits = tf.nn.bias_add(bert_hidden_logits, bert_output_bias)
#[768]*[768,29]=[29]
    bert_logits = tf.matmul(output_layer, bert_output_weights, transpose_b=True)
#[29]+[29]=[29]
    total_logits = tf.math.add(count_logits,bert_logits)
    total_logits = tf.nn.bias_add(total_logits, bert_output_bias)
    act_logits = tf.math.sigmoid(total_logits)
#[29]*[29,1]=[1]
    final_logits = tf.matmul(act_logits, final_output_weights, transpose_b=True)
    final_logits = tf.nn.bias_add(final_logits, final_bias)
    #pos_output = tf.nn.embedding_lookup(pos_embedding_table, pos_ids)
    #if not FLAGS.only_pos_embedding:
    #  pos_output = tf.matmul(pos_output, pos_weights, transpose_b=True)
    #  pos_output = tf.nn.bias_add(pos_output, pos_bias)

    #logits = tf.add(logits, pos_output)
    if FLAGS.output_dim_is_one:
      percs = tf.reshape(percs, [-1, 1])
      pos_bias = tf.reshape(pos_bias, [-1, 1])
      weight = tf.reshape(weight,[-1,1])
      if FLAGS.lus_add_bias:
          final_logits = tf.add(final_logits, percs)
          final_logits = tf.add(final_logits, pos_bias)
    return final_logits, sub_train_vars, model, total_logits

def create_loss(logits, percs, labels, num_labels, weight, dtype):
  if FLAGS.output_dim_is_one:
      labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.float32)
      weight = tf.reshape(weight,[-1,1])
      per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
      per_example_loss = tf.multiply(per_example_loss, weight)
      total_weight = tf.reduce_sum(weight)
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
    extra_vec = features["extra_vec"]
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

      (logits, sub_train_vars, model, total_logits) = create_model(
          bert_config, is_training, input_ids, input_mask, segment_ids,
          num_labels, pos_ids, percs, weight, pos_bias, extra_vec, use_one_hot_embeddings)
      (classify_loss, per_example_loss) = create_loss(logits, percs, label_ids, num_labels, weight, dtype) 
      loss = classify_loss
      hook_map = {"classify_loss": classify_loss, "learning_rate": tfvar_learning_rate}

      if FLAGS.multi_task:
        (lm_logits, lm_sub_train_vars, lm_model, total_logits) = create_model(
            bert_config, is_training, lm_input_ids, input_mask, segment_ids,
            num_labels, pos_ids, percs, weight, pos_bias, extra_vec, use_one_hot_embeddings)
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

      (logits, sub_train_vars, model, total_logits) = create_model(
          bert_config, is_training, input_ids, input_mask, segment_ids,
          num_labels, pos_ids, percs, weight, pos_bias, extra_vec, use_one_hot_embeddings)
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
      
      (logits, sub_train_vars, model, total_logits) = create_model(
          bert_config, is_training, input_ids, input_mask, segment_ids,
          num_labels, pos_ids, percs, weight, pos_bias, extra_vec, use_one_hot_embeddings)

      preds = create_pred(logits)

      restore_vars(init_checkpoint)
      
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={
            "pred":total_logits,
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
    def _decode(guids, input_ids, lm_input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_labels, masked_lm_weights, pos_ids, percs, weight, pos_bias, extra_vec, label_ids):
      return {"guids": guids, "input_ids":input_ids, "lm_input_ids":lm_input_ids, "input_mask":input_mask, "segment_ids":segment_ids, "pos_ids": pos_ids, "percs": percs, "weight":weight, "pos_bias":pos_bias, "extra_vec":extra_vec, 
           "masked_lm_positions": masked_lm_positions, "masked_lm_labels": masked_lm_labels, "masked_lm_weights": masked_lm_weights}, label_ids

    batch_size = FLAGS.train_batch_size
  
    d = tf.data.Dataset.from_generator(
        features,
        (tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32),
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
          tf.TensorShape([89]),
          tf.TensorShape([]),
        )
    )
    if not FLAGS.read_local:
      d = d.shard(hvd.size(), hvd.rank())
    d = d.apply(
      tf.data.experimental.map_and_batch(
        lambda guids, input_ids, lm_input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_labels, masked_lm_weights, pos_ids, percs, weight, pos_bias, extra_vec, label_ids: _decode(guids, input_ids, lm_input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_labels, masked_lm_weights, pos_ids, percs, weight, pos_bias, extra_vec, label_ids), 
        batch_size=batch_size, #num_parallel_batches=2,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        drop_remainder=False)
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
  train_line_cnt, dev_line_cnt = processor.get_train_and_dev_line_cnt(FLAGS.data_dir, FLAGS.validation_data_dir)

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
