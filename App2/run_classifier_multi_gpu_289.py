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


# Joint Training pAdjust Abacus 202001 289 counting feature
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
    "extra_embedding_size", 289,
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
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

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
      label = tokenization.convert_to_unicode(line[0])
      pos = tokenization.convert_to_unicode(line[5])
      weight = float(line[6])
      perc = float(line[7])
      pos_bias = float(line[8])
      extra_vec = tokenization.convert_to_list(line[9],float) #Counting Features 289
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


NNWeights = [[-0.0348846523058289,0.0514530466869951,1.11869811480199,-1.61033015900343,-0.741133902046601,0.303850786302748,-0.263291680103796,0.380113806992101,-1.49630275135475,-0.709137863732675,1.491590902092,-1.77968163240162,-0.523282611354038,0.981113260594474,-0.150417090786917,1.02169637062905,-1.21148436441859,-1.2077255543,0.638982239519681,0.101559251164723,0.00763860315521281,0.117669891803827,0.121845975043843,0.176454031374186,0.25118417762677,0.441900174700637,0.500032848804514,0.0609253371223538,-0.0677123589688039,-0.155637058066938,0.0194496718566565,0.0975803049738668,-0.131685436528451,0.129413535759446,0.410053560343439,-0.0287117822886054,-0.214391109060121,0.307544766738172,1.94767414450378,1.28686339094356,-0.193588318205072,0.92923235948235,-0.456740766395437,-0.588714640947623,0.579485897807662,-0.36209238440707,0.368106602843902,0.0563316480048247,0.60965405809304,0.23416307782551,-0.109279462194008,0.100012955579885,-0.193050678833704,0.100699853057978,-0.103805765518432,0.258574357965595,0.11617450083149,0.230945244050063,-0.103138801317477,0.0319629936280381,0.0562648808548898,-0.00579579295127408,-0.370865189573817,-0.0662531319619838,1.5934233008143,-0.872630318496569,3.06593545151249,0.197530659889942,0.804114614052212,0.351945499702652,-0.141051495702987,0.601920801053132,-0.434596781632482,-0.335768584959146,0.975451887487206,0.375543965566789,0.0863150201871221,-0.11307688153075,0.30424142527972,-0.0215669092115193,0.27794179796808,-0.321255368960641,-0.0610615024695976,-0.344480279636998,-0.760422521533813,1.13099441885305,-0.726087169841321,0.350423518905053,-1.63668421250902,1.22659236577151,-0.721409929681096,0.616986627603616,0.13328307376023,0.0249045296945555,0.137536705715495,0.0438274649680198,-0.0908811939492065,-0.0889890054277613,0.828580031573104,0.429173474317276,0.360875396567264,0.505016067853276,-0.854349264782162,0.102180520273055,0.213355619124418,0.0693447789023042,0.0924378975247452,0.139668768508787,0.315514377762311,0.0652240176026349,0.2038924121433,1.2489588212164,0.417782970808961,0.0641932796105838,-2.47361636752975,1.0829993794543,0.0143957170181108,-0.380414958524745,-0.399993660615127,0.0284372235696729,-0.598324659722641,-0.252805171641835,0.151499364960557,0.378813432952126,-0.0633578349922943,1.33339699526295,0.418803005264421,0.0597788903868097,-0.180340219636728,-0.168479364430977,0.314158147094291,0.228886686568469,-0.32152579625111,-0.0196142969379386,0.158183517024274,-0.262172238843879,0.345903018344814,-0.0991024947476852,0.460943932190987,-0.0914974710220044,-0.0451103025691185,-0.211760878190675,-0.100219581806692,0.337397174224144,-0.146632494805549,0.388066270938308,0.166828127665896,0.459774073475024,0.00575106491700695,-0.043624940385612,-0.0772949642568562,-0.267050487631137,0.117364695501682,0.072958376097581,-0.0481933102655021,0.757697000550706,-0.113905272016545,-0.0792758424695224,-0.101002162686644,-0.158437310591364,-0.060198515809928,-0.112892882024898,-0.0778290291929648,0.183889184653007,-0.0646883908803828,0.0881881488132415,0.287444350042975,0.0654715399460427,-0.0931895736050154,-0.0194739441734735,0.0777392093641892,-0.112957641094783,0.116603294640518,0.0408273921619814,0.140915396143711,0.0165100619901598,-0.0636194799200339,-0.233232364950992,0.0774544804140205,0.435408601580493,0.0379269104307021,0.0854062869364922,0.25620238511203,-0.188322666950029,-0.0183305636060423,0.0153082535209472,0.590522806856192,0.695423180194095,0.155576678058246,0.665822151312401,1.36512031415307,0.629912128848126,1.12977875413303,-0.0464316241960024,-0.323284199496912,-0.0305590674950233,-0.34805095023159,-1.03153621376595,-1.53694612539004,-1.31160449518109,0.345026326209269,0.573679491906616,-0.536617621993524,0.615507101090761,1.03454555636299,0.945934922620435,-1.01908526357027,0.176986757246399,1.07594922205647,0.438591330166714,0.195685604457269,0.0311799405434155,0.486953739938968,0.696406030646103,-0.592918128438916,-0.0652904563141385,0.0512211816598375,-10.4074911245715,2.48678315678777,-3.33107809198809,-0.929828883461026,1.06172924573045,-0.681925688539707,1.61892109239744,-0.483362310066131,1.3927397841059,-0.363054606483264,0.103793077760889,-0.191447621314677,-0.485331580364105,-0.172842261656814,-0.336784770784024,-0.853463331829253,-0.69908985675889,-0.82156178841786,-0.270745842703153,-1.53593859242069,0.203427678752703,-0.609291165882027,0.0363300895738348,-0.026618613583674,0.0889517795928919,0.0288645419610765,0.0747042806420181,0.149338237699334,0.150958442037178,-0.163804016206385,0.325488383051967,0.182205605193676,0.135695074218418,0.0140967697538853,0.0271974554844781,-0.00750525084400568,-0.00298669044318245,-0.629525914883985,-0.197207113700407,0.00983386553445563,-0.0506542997580772,0.0759986219145716,0.171006266106528,-0.000873928179530128,0.0903887488973364,-0.488908573946801,-0.306803214476217,0.473824747340855,-0.353574328132998,0.0822494284835144,-1.12554192024566,-1.7209396086833,0.652067517330328,0.10252912514743,-0.839288061398893,0.560239450817358,-0.0445022847229892,-0.710249838290157,-0.421333779804358,-0.231528900622564,0.13110816248774,0.855916418072915,1.17295905610666,-0.423113912927195,0.119304655168621,-0.0172365971267693,0.688598306975162,-0.174380365992831,1.10060417324922,0.0763536637899895,0.0897585821047554,0.565223655527298],
[0.0264952770061011,0.113601283756444,0.313400993663534,1.30733173964573,0.0646580565156639,0.420813002503798,-0.111357879064093,0.0626478014528321,0.922263524272486,-0.147721784511063,0.210936121602533,0.173703539307224,-0.0977003502412008,0.0521635907024109,-0.074899989282887,0.565972451838723,0.498801962416113,-0.133828179663427,0.235919102721082,-0.0930664851273588,-0.243590266444703,1.16769743243799,0.0863246905515362,-0.0744046029931574,0.0659167105664739,-0.0101774043518877,0.33996483292094,-0.00938737254074342,-0.588522690274151,-0.00473231780073601,0.0189186606015573,0.0623953740488817,-0.13599249658562,0.474610999709814,0.261399761606555,0.115033367784521,0.00246001644019147,-0.163190572104153,0.00839290432808214,0.127508731010942,-0.568481451340533,0.493897366225339,0.625367395118118,-0.0982002640907456,0.308058645222106,-0.141506578213766,0.291338550303849,-0.529914919332477,-0.700612649110669,-0.100821841752509,0.295324923298958,0.247273873560404,-0.244869055168602,0.046650548403271,0.131129579599908,0.388349178713109,-0.235925908005792,-0.255036072544225,0.31116815336542,-0.0255662720573435,0.0699199200524155,0.0197604553311372,-0.655692449422774,-0.166927612399423,0.741570814359304,0.137140696153063,-0.117799197080736,0.00949695505268394,0.617908487716694,0.311136597662986,-0.259338518226539,0.0206337873560369,0.0487016242853626,0.280877360215952,-0.0223840233229115,-0.328850703787328,0.402606893419538,-0.0910484873533748,0.826172364799251,0.273525050654006,0.359282761966949,-0.100759846687819,0.220953930930622,0.172146488380612,0.435812807437941,0.135982754296546,0.24989603312417,-0.155737363285274,-0.401409087074931,0.106897503985562,-0.0379867867698985,0.360559119327215,0.0100275882790221,-0.310214943381943,0.0505813351312037,-0.470817193547799,-0.0621579287455316,-0.13733734477383,0.643385626482547,-0.00557766232438477,0.496585620703844,0.27690710819585,-0.185668249131301,0.243885149027296,0.293062202488185,0.121462042498084,0.335204644159152,0.125671892334239,0.757214145099138,0.150387698080712,-0.737225594525549,0.539096747412284,0.530970266056843,-0.282150330139683,-2.73272788219972,0.610979639650883,0.22364651778972,-0.0715275972152202,0.133210081946925,0.0274325775644224,0.0704291691652044,0.117516694939517,-0.828341923540361,0.0233685000330843,-0.161671747966493,0.199616358707109,0.161315525614312,-0.193739254222841,0.422647889658271,-0.188420532613786,0.0931519321848795,-0.166346933179708,-0.125528805017089,0.0097941627145025,-0.521797738086085,-0.0565538543480861,0.224010239717887,0.532115005948104,0.0539549906022435,-0.100859785032119,0.236442406541713,-0.0331559434372522,0.205133341736509,-0.147029731810899,0.259348951116512,0.243344848057291,-0.019780403735769,0.00383430322622614,-0.0175287674493698,0.152682921227626,-0.233474039147879,-0.109731417549103,0.151219505117119,-0.0347430793807349,-0.00941762478282668,0.0245124017271243,0.103818956627866,0.117593466474736,-0.0695056499969882,0.14538132638491,-0.0704508822815868,-0.073809847095583,0.0128398246980917,-0.185245291232771,0.0366646059777302,0.104113181793769,0.0998531774518331,0.0896084364772697,-0.0672238120754444,-0.0741230215545662,-0.0810429996177139,-0.0279999634034165,0.0181429174199042,-0.100906072485221,-0.101524183842155,-0.0473278536799909,-0.0275714668289974,0.196949539060111,-0.0185433866078628,-0.0753880737731387,-0.214593727408095,-0.031071166712063,-0.0614349634207035,0.0856096190178872,0.0438570255579906,-0.011127805931216,-0.0773456199700025,0.0945126024133526,-0.0766843024128025,-0.0784631496415285,0.122066503630985,-0.167460435359316,0.0427792116470471,-0.133932250804302,-0.18099880657702,-0.07146745776347,-0.396271496797458,-1.31131357361684,-0.535974134714686,-0.580570068328175,-0.350112464441607,0.263010313108292,-0.965566238500176,0.37005199773532,0.329527072686905,0.382396145391335,-1.09229554163677,-0.0394265053620375,0.589393746068831,-0.0269939840001186,0.0495580088710629,-0.18266836160884,0.329650264037472,-0.245381679663375,-0.532280047874688,-0.0715621890922529,0.0408243613784765,-3.54112448063776,-0.0421673561109168,0.480635231800021,-0.0649130735264948,-0.04857623077281,-0.638615702824141,0.340062109274718,-0.253687203863726,0.484887253961854,-0.292999533899327,0.00847758981948946,-0.0842863814342899,0.272154877126758,-0.121538346547739,-0.330403913107445,-0.233665030563206,-0.0196253709600561,-0.258912107114665,0.741596147795396,-2.30164790303763,-0.52749894039181,-0.179489336692782,-0.0203835376464394,-0.1106543442211,-0.0304404821388299,-0.0385115371101557,-0.508292962665595,0.0198573556333701,-0.0961239960476218,-0.0263880309775941,0.120411873747993,0.138418182155966,0.133017941277686,-0.348321521513099,-0.0554848309025615,-0.230593555286715,0.047974596976384,-0.00597921169004217,0.0330017831609653,-0.0461568555312419,-0.025175537445998,0.186599332847923,0.0163135189229646,0.0480718270367523,-0.298340524701774,0.0519490804072507,0.0625625169988715,-0.0139759552109863,-0.163418547801315,0.21859852010297,-0.316116532416114,0.114366883990619,0.134157752794499,-0.592630860154648,0.25337221581934,-0.658818622422365,-0.475161030476527,-1.41822795022746,-0.262392634276299,-1.31219734492933,0.241320468692056,-0.174439393416827,0.124932330948291,-0.441142732116078,-0.692269456917836,-0.37344971112903,0.0343059252642092,0.0247518940379068,-0.0152374969381585,0.645512814839639,-0.386163960807477,-0.205714041468928],
[0.00271641969941997,-0.775559495150016,0.597523401221924,-2.55139781018127,-0.516874070627946,-0.796618257221799,-0.320703990194313,0.0833112874989343,-2.87871104428221,-0.400882945642809,0.00865488142285285,-1.67845870851472,-0.313856469291721,-0.00651939558393151,-0.11597907941998,-0.143187386422427,-1.73254401433609,-0.847015658272069,-0.0955873833862079,0.127212163683382,-0.0306050799463738,0.217767893333984,-0.123744017453235,-0.107110670740338,0.116906259445179,-0.0568585905574667,-0.241578600093513,0.0600638831180824,0.161104342283174,-0.523040222451174,-0.0290272623206249,0.0504309551324058,-0.00698649452481591,0.139362080877521,0.0999869435321509,-0.0671487151981891,-0.00323676407656701,0.099644663278944,-0.0185972226308296,-0.0107988251956239,0.927674362798891,-0.606348548162508,-0.446524504916606,-0.624198743191565,0.274053249266853,0.0187915155160912,-0.0488033428107234,0.922324915253121,0.933606064035769,0.142568553769409,0.168248742103241,-0.0879463126577378,-0.000191975542251681,0.0367150970275197,0.156035371999306,0.23094183610145,0.0342818672138577,0.295639990759503,0.107961673588487,0.158976684088189,-0.00266961870190618,-0.217197305124231,0.62682709454371,0.0363901105053457,-0.241582770511448,-0.484549454628545,-0.0876636258136023,0.268516267928332,-0.148228559876116,-0.0302038099865888,-0.564132145474568,-0.769866016478872,-0.57371155922895,0.0330933967496121,-0.413697729779107,-1.05664323502184,-0.118452358575071,-0.071368479719046,0.341146428064299,-0.0793197228135224,-1.66496340639664E-05,-0.235521708632261,-0.0942340441147471,-0.0116428219112156,-2.75139210867569,0.967121042456118,-3.13383170196755,0.362755620895124,-1.64154598918467,-0.163519079737225,-1.92935821992317,-0.249645731238685,0.0594125465529322,0.0559118508531284,0.156272852332095,-0.0208641020302365,0.199871387717316,0.0233778602310766,0.252915574442598,0.0184280804451693,0.280522702118573,0.128623998684815,-0.272411133928476,0.0730327883652031,-0.0859350416334177,-0.14764373198919,-0.181826420256875,-0.0532686495863852,0.0984982753069746,-0.205120033022006,-0.271027867372707,-0.138959561322427,0.0541585038257536,-0.0761306116931498,-0.926947741858714,-0.020943571989654,-0.251156979879077,-0.242942953125387,-0.455755705071263,0.276049624776872,-0.58000475160475,-0.00898335001973401,-0.203357715997875,-0.439479352504844,-0.450184487666471,-0.351550612260502,-0.374318875422571,-0.591556503074577,-0.196689952246199,-0.0439244393771837,0.266587081370944,0.309984722073341,-0.120326974526163,-0.0233637247652417,0.0818223534844042,0.222287066165627,-0.0193481926077991,0.111085633589397,0.411080096245565,-0.145821407007165,0.0709628495872935,-0.193301916491778,0.0233360209819529,-0.168596974640963,-0.0111092168153162,-0.187247842942704,-0.011110542383419,0.0189280121904719,0.0894836421627794,0.0661768914858642,0.0379533725580552,0.134742817532712,0.0987978351293535,-0.0532187869649592,-0.214578963646891,-0.111278291786667,-0.178358248967411,0.0517905411936739,-0.226063005122182,-0.00222035555917755,-0.155247161261278,0.173155866345233,-0.0454956138265076,-0.0731942262540789,-0.0776080799992496,-0.0268555150165475,-0.195635474471453,-0.0304401998516337,-0.237614230514777,-0.0583424856960105,-0.0882276358463369,-0.025344018652936,0.0640116196356861,-0.0300922800060866,0.140747286488908,-0.0114501120279047,-0.0397423536540244,0.134760259095659,0.00715759050720347,0.0602760942488982,-0.0742709741854465,0.0270079141038666,-0.0710028053472541,-0.200905158987919,0.00305101345740431,-0.0395329948901346,0.035145564857547,-0.156133369216083,0.0490579323370998,0.0501444863047181,-0.0366538584380375,0.0612059493520535,-0.132371544701732,-0.181402813542449,0.0941014849670561,0.239927048606269,0.0726800301764571,-1.44550134245969,-1.57441266506698,-1.06578936900707,-0.16853235366341,0.328139625501864,-0.384991453945729,0.368901936466021,0.264468278009013,0.381416897089971,-0.809951586587026,0.943319469479097,0.483243699580616,-0.232335151430219,-0.0922076617841383,-0.267801045333672,-0.00393376085233848,-0.131164745783071,-0.32643417256432,-0.0577979370779308,-0.0119670443912255,-0.110512585998822,0.604114853122484,0.00809956637773399,-1.51192502294362,1.29807036730541,-0.527611216081072,0.921066194728822,-0.619233462361463,0.722267268872828,-0.0665103684160515,0.386603871898828,0.463918660188258,0.540807858033432,0.42650696853547,0.32244402708623,0.272651348923741,0.116801258310172,0.193847653109424,-0.130822327038691,-0.0391107602732829,-0.271229040577539,0.391488340412895,-0.0255839524188792,-0.02169009048522,0.0498738716043192,-0.210586385488517,-0.0898949154937548,-0.128776579389848,-0.126682709951919,0.126876743928527,-0.0961080476676567,-0.134967006933103,0.102116889900206,-0.220584320570702,-0.00503876560048321,-0.12012849891023,0.0173660001687324,0.197602397645738,0.122838187873553,-0.061236686361573,0.147133821228293,-0.171979352959292,0.203461767842588,-0.00794176928602055,0.484695130583114,0.266671543586878,0.163285390142276,-0.417302767243689,-0.232194222724357,-0.0974057171111326,-0.776219655944311,0.421915601873269,-0.428823479265019,0.135458990126971,-0.250283120708786,-0.0635283576359329,-0.10242624511019,-0.362394885616377,-0.114818783785038,0.203774846468935,0.0178636275047969,-0.363511726807147,0.280113902725649,0.0751977960536437,0.45282310800218,0.27171769534583,0.393991869336738,-0.0324229956798275,0.162869587027194,-0.0273799737889777,0.340683459294474,0.29642363376924],
[0.0172389340078767,0.111622336451915,0.817743103821446,-1.19361262749321,-0.0866355355995723,-0.553692593361207,-0.134247143428125,0.486574847239378,-1.21029711417055,-0.0456969196141685,0.453134406195613,-0.833118769737519,-0.149914784060758,0.0436893838125055,-0.296803035733947,0.461847936331955,-0.54041292060179,-0.483492406752792,-0.226589624611079,0.0294664195841122,-0.0360658848194962,0.714093760067737,-0.11266727770447,-0.0777972897571266,0.0108146526300141,-0.525644108838188,-0.144315117191426,-0.00513687239662652,0.13579729237567,-0.51062722296948,0.0321231780658854,-0.13292808989321,-0.0622787321020622,0.0339222603748328,0.123477514430021,0.0648790382451241,0.0453010733734872,0.0829756155446607,0.337983874027603,-0.28096033649035,0.67991204607959,-0.306498059137811,-0.940914354142493,-0.886816752062633,0.282566776193743,-0.316962109588194,-0.107291490162938,0.794711659351436,1.37916160676082,0.139395727408667,0.0166455564256257,-0.00575584623215243,0.0402580655719529,-0.1451301226682,-0.199648367223522,0.491459879201098,-0.187326127449175,-0.188829696630329,0.360403289431091,-0.0198534919280569,0.174696905713003,-0.116859258698565,0.116433785537452,0.441792263250597,0.0635305485205664,-0.532765174276393,0.270403362600627,-0.101717869857597,0.085043181269493,0.134002960375555,-1.1386373769822,-0.610090624595242,-0.478130088107474,-0.273851581437022,-0.340785976361858,-0.309020408940437,-0.0184990765053635,-0.0733129586526544,0.120704984950758,-0.122912928356605,0.0536435642081896,-0.32336571892536,-0.12266652986176,-0.103431410624843,-1.76142511643664,0.945877906832176,-1.8555462436097,0.592607425827972,-1.16732865354367,0.362591101643589,-1.22364508937531,0.401954389871704,-0.126552487982126,-0.0833446467043844,-0.0172789294456194,-0.201977360347465,-0.1451871197214,0.0706652302230271,0.137144518472961,-0.207295880232825,0.115019746987332,0.119569864402805,-0.598864154908374,-0.075640529923215,-0.0396401493968831,-0.0137926029804021,-0.0702874502647636,0.101748981407563,-0.15782816592559,-0.176621942850169,-0.37649766550277,-0.142626626720951,0.072259197709166,-0.259316396739926,-1.7992327185383,-0.0724611179994435,-0.18176003725482,-0.163136799806812,-0.604971698479637,0.419989716849218,-0.560219918921932,0.180240588453392,-0.451318519074841,-0.452404022822485,-0.658197002556294,-0.20634157205846,-0.0391509771429578,-0.157564490852199,-0.563068582345598,-0.0220923424377163,0.276915759871947,0.0894675752213598,-0.0943602968009778,0.0150254616173988,-0.0584350952187779,-0.268066464826484,-0.0575117435553673,0.106249002715664,-0.119614741286979,-0.266887732922451,-0.11344871105516,-0.215265681974778,-0.0613896133523307,0.0684112782166905,0.0495725259272834,0.0706423535562347,0.0420461761310617,0.146430648947208,0.185888142886525,-0.180879602989141,0.283610713686767,-0.13029335754652,-0.174229629972014,-0.231635170314776,-0.171278833844257,0.0798940992295714,-0.0558033733947027,-0.0473785475404742,-0.180143012127633,-0.0561907175875441,-0.0571164647431236,0.0980536448243144,-0.0255816914985083,0.0332109534682938,0.0968470772424597,0.111733424586494,-0.092033196730688,-0.215123636536829,-0.0228583238290984,-0.0741847350297152,-0.09221552558129,-0.146555314508048,0.0360350934426326,-0.078217589176903,-0.0355947987621418,0.0163010942280107,-0.0133078630457389,0.412086826162246,0.0675308740738001,0.0715333855352051,0.154495657823399,0.0264739664152453,-0.064433302987631,-0.150979977964319,0.0531278015597368,-0.0102673607581345,-0.260401035453893,-0.197771195817047,0.214287585665336,-0.11418470588688,0.00828738450936401,-0.175483440035593,-0.0110746028330991,0.0526248477303327,-0.0399378372177944,0.276688764787711,-0.102679713984547,-1.91818670653747,-1.65136521922865,-1.42145671458634,-0.31400511400914,0.505620359822445,-0.532645769115917,0.581488974098753,0.461784589522911,0.480598278269146,-1.14843119966216,0.449164255807794,0.748393789805679,-0.553301015601317,-0.139594689861617,0.111498695592162,0.453205494008832,-0.209767946911521,0.146012471423912,-0.0533847268238649,-0.0145292469269764,-3.27643187379869,2.33387472058625,-0.804107732201513,-1.39218002371373,0.775500434601053,-1.0058616463243,1.62818939412342,-0.864694791810223,1.53175954643766,-0.454530167256214,0.220710095203275,0.19212991810606,0.222347697317097,0.459541558742696,0.208166902473905,0.167499051179037,-0.0081994924243711,-0.0533384485293914,-0.260139401340782,-0.816811051921441,0.315231596601432,0.0862362460295612,0.162344551763478,0.476645276077504,0.109105845211245,-0.111628363408973,0.099893366046004,-0.280497714520968,0.206325290303254,0.0155250175762042,-0.659655108880249,-0.232515926943746,0.0882560595091539,0.359619118144729,0.0795871336257979,-0.0398162540724556,0.0295533846891007,0.134637403558225,0.205583913949247,-0.041059443401888,0.0474979818733445,0.146261846628542,0.23266398145188,-0.0329490362977405,0.511175344966388,0.0512847552267652,0.0448361162822017,0.122945130082635,0.23380321792446,0.390018723327633,-0.764836830800598,0.673055052604805,0.290498913462955,0.347542681080646,-0.206745181875183,0.640358217988064,0.366627639097841,-0.8243621520198,-0.0464144894300555,0.0939894723240767,0.111632943025485,-0.309026065379981,0.498028083173314,-0.421730552121346,-0.319465000256795,-0.137854759343427,0.794997565255451,0.256936240458819,0.630906638951417,-0.0957693572073794,0.25483517909904,0.346722184715099],
[0.0166019012585242,0.0166651744435973,0.19804936996785,-0.221067444661488,-0.0751077015897103,-0.028506416916629,0.222768077490021,0.016395865865502,-0.128843966961961,0.885896887853983,-0.0446544001246654,-0.339920046512242,-0.0826821297076101,0.0657470355057061,0.604231403797703,0.10364089595082,-0.431025838869965,-0.219643774902264,0.104588804138945,-0.0106416195123509,0.0199823386635967,0.466275621471557,0.0281365326176912,-0.0844077805540452,-0.0142399295963134,-0.103293043339902,-0.27923034118384,-0.334988692616248,-0.284418553518512,0.189126190845522,-0.0419619453982435,0.0477773318766708,-0.0875466307046684,-0.0390820334041106,-0.0107224856463479,-0.024684777310975,-0.0880834883784251,-0.183962605621254,0.039216993560689,0.243558313775534,-0.143909896536737,0.000759536126004031,-0.437024964613789,-0.305205765474216,0.181752367370761,-0.370331640561303,-0.111395258475916,0.0194330970960783,0.479788427688781,0.0563773987194448,-0.15152672380401,-0.127832454552778,0.00952329100960384,-0.20864521651191,-0.124398997409221,0.0450083060636367,-0.159926888558823,-0.623383616846679,0.281639527630968,-0.0479050805642138,-0.161895522093656,-0.102760452055602,-1.04667659750582,-0.639971505961583,0.456373526464297,0.0324845663875897,-0.248278161045758,-0.38092821653173,0.170752974383935,-0.00316413788243325,-0.758017757647767,-0.274769700090899,-0.348082555166193,-0.275968648488399,-0.0252725854224388,-0.227162957684412,0.0806016551176466,-0.0750635950190493,0.285927150874459,0.067224207154467,0.131115420918812,-0.213766047336301,0.0650210623798704,-0.0331017047081836,-0.522662347757213,0.229565763311079,-0.528071865513646,0.0204312518447132,-0.537745354736234,-0.0856796912934063,-0.592543962106348,0.0179911404154196,-0.0820611196704758,-0.118711718637123,-0.0298173081405481,-0.229280809952042,-0.0605198646928956,0.178499964514033,-0.00717397553816462,-0.198014948991431,0.244950353204132,0.261148634638329,-0.37471098212554,-0.246568579375445,0.0407235954496909,0.0657073850153272,0.0380327791950629,0.170199372227654,0.032028952293279,-0.0949052575198405,-0.528556656512158,-0.0591148586069184,-0.175511506153092,-0.397976278855022,-2.12874281394949,0.0953475310603963,-0.024063570886692,-0.119196627922849,-0.544871202653629,-0.293501910871242,-0.242593626122926,-0.0687075382228272,-0.618606387066602,-0.32679408746197,-0.16103854736315,0.225995582891534,-0.248073622752776,-0.132268892961657,0.165418493941702,-0.0238579800748595,0.00892259746577922,-0.315315512610912,0.0352081567802657,-0.00211883510432414,0.264031252541039,0.00272079609958746,-0.182879663711106,-0.253659935861289,-0.30326010398488,-0.164094489799331,0.093929072250824,-0.180726685019911,0.0510589373461902,0.554994015280074,-0.122204449178425,0.0950465502738835,0.0201452792137811,0.0421294876328886,0.102461012267581,-0.0234521119803799,-0.0743440521881295,-0.0516101776839152,-0.18985673409519,0.0736110641872854,-0.0411471302266896,0.144234730651142,-0.0294570232039502,0.129368440083711,-0.105229346931534,-0.0411187591897263,0.0296160141954651,0.064031731748379,-0.0400418259592343,-0.0622319813255838,-0.0100955217181998,-0.05426683068843,-0.067042574282804,-0.133848777891875,-0.00271385443485426,-0.0550827264699801,-0.131693799580099,0.223469379585554,-0.010050751255313,-0.0930446296757524,0.0524044920509837,0.017955631886741,-0.0585491773253044,0.159811093546517,0.03061446805041,-0.000169102240515455,0.156758062809879,0.0271710072216656,-0.0684828998204409,-0.0995684594215631,0.00398923980560131,0.0289517414898433,-0.0571902235529337,0.0275599854148689,-0.00418500591530099,-0.0675802091653811,0.189646865053415,0.0372266738701173,0.104087026325706,0.0739050025675498,-0.0725956764503343,-0.00400104176074777,-0.202995305444642,-1.13386786072194,-0.706259038019293,-0.66710334879681,-0.192057124490737,0.275283659777536,-0.414446776335266,0.466789993712567,0.298492348764971,0.375600360510091,-0.828903478749106,0.156156683985339,0.527250834631587,-0.645666111033163,-0.448329930815984,-0.0561604406090689,0.315074864297254,0.561511515644596,0.429484213040287,-0.0882632425119967,0.0180874441033487,-2.50957373067885,1.24925608373653,0.401089131784454,-0.381046127482387,0.413461707676425,-0.658330449180362,0.864587078264581,-0.445557714108618,0.930681897136682,-0.276362713719529,0.1747257686038,0.131116603640788,0.104291649232833,0.247240663850987,0.11640496908925,0.0621553440893429,-0.285122943961179,-0.351443949685641,-0.147274338730271,-0.648219175553277,0.0965370212664596,0.233919129169292,0.167042738904174,0.261879308461911,-0.120752792457352,0.0856111259826802,0.151420790707683,0.00827041647983782,0.153323282267767,0.0119971887030194,-0.834653131168409,-0.013481690159544,0.0247352963993489,0.0441155287875739,-0.0665589095992553,0.203251353416484,0.035981490376209,0.29412364027498,0.136462007407066,-0.189451396281719,-0.0700752685110306,0.0511028710081553,0.0708373146952366,-0.00281542970039324,0.0226213652001553,-0.119561211826085,0.090298134175575,-0.112386752057656,-0.490034089441797,0.153513410886697,-0.407777276514008,0.329133520202235,-0.0673031008696603,0.0335559634936355,-0.167796304489254,0.198290333122843,0.662482211102177,-0.462770862591395,0.131066488692761,0.172409695133138,-0.29702460465415,-0.41450516301972,0.071810641350107,-0.409128858054092,-0.29320551546857,-0.181753159886758,0.0380994426736917,0.273006541199947,0.141869984024324,0.0450704024408384,0.113508546697093,0.224345033004458],
[0.0269338165882507,0.0874764226466988,0.253442359818297,0.904439252156193,-0.102819900756066,-1.51402743506925,0.00152379461786353,-0.19658999521407,0.858516065752143,0.0780496709932319,-0.059448646497213,-0.513201891797638,0.0506222391604831,-0.113003736035477,0.0942593231547235,0.209479277437678,0.136505550246058,-0.124169747702605,-0.631523544247171,-0.0490225961399643,-0.0800284015077744,0.438128326913351,0.267987690309918,0.169265930441,0.0537885781894871,-0.195364808361703,-0.459926473009844,-0.293616924563619,0.0745806093403661,0.140489087486689,0.0710811160428781,-0.0672014688317636,-0.0144179816309022,-0.0622452147256158,0.551086558841464,0.0604515268159096,0.162291035427527,0.160658491625206,-1.61219744405304,-1.52701673225505,0.0815433339677991,0.256859170236186,0.960778218591494,-0.329494496295588,-0.497946909825685,-0.186498884408874,0.00581070457392808,0.138415789282138,0.230976254471354,-0.120741366060719,0.307932193558335,0.353833956700549,0.200839757979952,-0.217300045012897,-0.00811321299676042,0.00281284117355219,0.374932228847259,0.221938176950028,-0.140710999110011,0.309992358853689,-0.0449883834420525,0.149834099449229,-0.475981989913912,0.0913965452230181,0.00372556195767327,0.420706303570785,-0.90242735254977,-0.228917906003211,-1.12032726017754,0.080482351915454,-0.744433867505833,-0.306402100751373,-1.12983731043456,-0.187246225935602,-0.206349318511697,0.393090603481548,-0.0967123892346771,-0.0456370176166493,0.247475242727069,0.0660909217346066,0.014317736952078,-0.115263683165052,-0.00569805196444199,0.165164937839977,-0.945063125046731,0.083694748815752,-0.824864762013823,-0.424512043972321,-0.980156666405307,0.0342954955552764,-1.05213394476517,0.183269443166561,-0.0132035184963211,0.236237293359128,0.0821677295442493,-0.0397718310882753,0.0177112714284573,-0.203262018560266,0.787523831249278,0.093261466049013,-0.00493161375150415,0.392521313678176,-0.171948302090829,0.13292537342556,-0.134701766482739,0.0808408825760442,-0.173140320007395,0.166678427449091,0.400325628070138,0.206727825055749,-0.822433762694003,0.00865083995292866,0.649212863695088,0.302071336859474,-2.25350896209581,0.0817203295762399,-0.104454603283941,-0.114659252738863,-0.0485308174548625,0.244455126006144,-0.306771168501583,-0.0583435775295234,-0.869184616690582,-0.111025410986395,-0.616951402601796,-0.467807691901596,0.185253649388425,-0.239989035040795,-0.487349583198218,-0.00761242403412838,0.159002375939219,-0.204154652764856,0.0756240492947546,0.0741458345472606,-0.524212927846537,-0.22576776307449,-0.441095204371356,-0.341883361064946,-0.0847621840278863,0.288860678805595,0.0904157602584695,0.0418397021434828,-0.019154655521987,-0.0966304964533813,0.127138859203585,0.0447552675420159,-0.0283620928241307,-0.18764879802987,-0.0179151929716053,0.0326063505726213,0.0588425941593914,0.0588782546895329,0.195337907836791,0.0698273587846478,0.0646511337338897,-0.322354834595335,0.204399457912265,0.0886051877277373,0.0462427138159073,0.0217698794632425,0.0747061371856331,-0.0894338187624658,0.116251364235072,0.226231472859508,0.151354011054732,0.151617969638243,0.0396615648791451,0.0888821599107449,-0.0228668177156698,0.0101808799591811,-0.207233738257504,0.0605537259256461,-0.0387424069013496,-0.187446078062363,0.135523149011791,0.0121467557191223,-0.154938432171435,-0.0417705812612988,-0.0880999336959737,0.0318042738795586,-0.011698765052886,0.032558971134076,-0.0243074720296876,-0.139554492568416,0.0758969284194601,0.0371895883090753,-0.20527733183149,0.607330209395578,0.0618219887882218,-0.215053606601155,-0.128366483868676,-0.213220039772309,-0.0996012653645613,-0.183922145857836,-0.351584832765034,-0.100670761514628,-0.158964941288091,-1.46430240406522,-0.860322628046609,-0.849493499026471,-0.184654901698003,0.133552556681726,-0.371697239182673,0.318046906203989,0.240940318903198,0.102509055642495,-1.07169782828951,0.0331405349147056,0.333962607819445,-0.126493203394638,-0.260832366473502,-0.730937661925077,-0.00857124417744343,-1.04556110400854,-0.26754260486929,-0.0731801606169722,0.0384206592982682,-5.67350803300972,1.1160449114703,0.839483480988564,-0.713854577536283,0.182457539191285,-0.557396288181661,1.07424473412008,-0.711249007088117,1.09395617557439,-0.610294994040877,0.0640207204528702,0.231614160881331,0.148694645282441,-0.277966758552727,-0.261012706467336,-0.527515042750706,0.879568112790715,0.960926785697126,1.43977476085178,-2.39199285413673,1.35855799747754,1.12732523623377,0.329981337611055,0.422935300989118,0.48974903698153,0.758586332308653,0.741022842479579,-0.193504408796591,0.598104935430875,0.252471173760328,0.222206796081953,-0.0564318605100773,-0.194966864110809,0.849136967323792,0.351708915456506,0.393254370201317,0.0380843690038861,0.811456802476977,-0.299746201049951,-0.184112179657523,0.11349517334452,0.00695087120579865,0.240105862659512,0.0694745851195177,-0.789093010199562,0.0746579004822526,0.144356679538541,0.334041968872085,0.0454886936959618,0.307996084806832,-0.57982069533038,0.0753079175048133,2.52825335558515,-1.67232893003676,-1.93257651854947,0.22914187805872,0.420422351318422,-1.27269451209047,-1.67290902305503,-0.141743237807805,-0.518613385674091,0.477891505168934,-0.721154355408373,-0.292216309082091,-0.284245441273397,-0.160757840111114,-0.163399279123798,0.278357171923533,-0.363829619773068,-1.55681676594227,-0.172593072568896,-0.0267659909021517],
[0.0654477842720421,-0.323297147549899,-0.0643879417786255,0.982654538063901,-0.491683612946743,-0.359678260250737,-0.0248848668363588,-0.13100172146257,0.615767766000814,0.176294864378354,-0.126015301103361,-0.342339156762583,-0.0498149508809698,-0.104351314448001,-0.0794284405436034,0.347013464849202,0.274142046680106,-0.147406481446532,-0.14145751149975,0.130671808630761,0.0428069514207527,0.912145559606588,0.379754744085988,0.0602918827093028,0.00549497964248865,-0.723596810911634,0.211179548584902,-0.144998162025617,-0.617136996610855,0.264895057589979,0.071723791404959,0.0163676357711537,-0.0284745253674022,-0.0399858066457153,0.294929486632256,-0.0384898388647888,-0.103263256636169,-0.191499455829064,-0.214417771060223,-0.506573772918191,0.0445042361313457,0.295621704987442,0.59310213893047,-0.539840750709691,0.0566607069934828,-0.0504425463229055,0.0713468048726263,-0.0664334230158955,0.457382372153514,-0.183307593288427,0.310181936267923,0.195101865375264,-0.294273195121503,-0.0624833561916091,0.0937381815642361,0.464064018173706,0.0905724354003379,-0.295605242627276,0.312985882320464,0.0516476985028387,-0.304088290569737,-0.267473183292546,0.565171321534592,0.176347070135686,0.282790943133627,0.540534145784687,-0.333849888989967,0.0842270097857225,-0.124476610058513,0.566526302934078,-0.405684634564775,-0.209169251077103,-0.628415389616374,0.47981659238172,0.190559161518814,0.0954917067577786,0.281427996044991,-0.179885466804397,0.336943084817834,0.282039477982663,0.22659402714024,-0.165677347013127,0.0198332773371257,-0.0517060871624815,-0.503970298529691,-0.0971344285301368,-0.644668606270447,-0.233583243705479,-0.777573056597753,-0.0203221087041997,-0.730088232218743,0.320478372108942,-0.00135332342494392,-0.129883756472227,0.0527329672370254,-0.265155811022866,-0.188753662961409,-0.0930561084350025,0.131742750187922,-0.345254721676366,0.243134871322806,0.0149679161417363,-0.49218117744071,0.0196140160219492,0.176389538177536,0.0745695684066527,0.164124601581937,0.0648325402596873,0.17362113865747,0.194499877566682,-0.582357045664027,-0.0819218466767395,0.0939944814736935,-0.0464882918211979,-1.41608554319721,0.0181487105287385,0.185354944382776,-0.0563729358274694,-0.373979847121908,0.258722986101962,-0.00549433003751581,0.0777801311841851,-0.512449291880857,-0.12814047163285,-0.219446178533206,-0.126947287053652,-0.144591576892645,0.142112520362873,-1.17901814348954,0.000362567667205691,-0.161553049158253,-0.0622644835776673,-0.0124099361719288,0.151750854258969,0.226268404630766,-0.599808623430115,-0.178961925777921,0.0650673168718698,0.00959162930462327,0.0423626980818871,-0.25403762566255,0.427957434456126,-0.0933803029198037,-0.0712091492211416,0.132701758739575,-0.20748321393364,-0.0511402176608202,0.0772696877982612,0.000176456805364668,-0.00567141387875787,0.0503509167296211,0.0973843879389189,0.229579549689387,0.111890797889939,0.110024383023171,-0.123561541131503,0.254571635427167,0.116412860082967,-0.00109567115076501,0.0453970477666339,0.0275175401370896,0.00973342216105233,0.0197296480959553,-0.0318099506707522,0.118586588357396,0.0379648709546952,-0.131662896511707,-0.535371946539076,0.012975652438355,-0.0549623426566823,-0.13710240001025,-0.16758230095745,0.0330700034870131,-0.133652194185082,-0.00210602093543627,-0.0270138311403979,-0.0947091424190786,0.0478163717602778,-0.00484134883867036,0.0542323999377361,-0.0102573122771032,-0.0372623302756438,-0.125800341188677,-0.0358556212815629,0.064075453111869,-0.0359911336496579,-0.314079986906221,-0.276827651111598,0.103964090369735,-0.0946659788385789,0.124820394365142,-0.218768566102202,-0.0496776390183774,0.396852140481345,0.271732907934122,-0.208722886920023,-0.429582615719867,-1.69260210743245,-1.11728359757339,-0.721767343561089,-0.451029033592968,0.365161076567799,-0.68879159577316,0.389053323051951,0.241659471587432,0.304790059950355,-1.11161104993041,-0.0623341791565885,0.310315425652394,0.0993938353046462,0.194239225923264,-0.118843102863173,0.498219616630772,-1.20762225177707,-0.380066191862721,-0.118147618629975,0.07911990983957,-4.69056127161567,-0.790444536959686,0.0971455515977266,-0.155534870332882,0.0189281958602737,0.0535453048455947,0.322967554435843,-0.259120997562577,0.618367995410662,0.0502721525423415,-0.033153891408917,0.0110471002037149,0.102171517538236,-0.132824483446596,-0.133279776014318,-0.184838870842504,-0.148978549735108,-0.0920204151991412,0.260279177402396,-2.93510654835498,-0.0371775572003331,-0.229140330083077,0.0531473885536261,0.0099581446702528,0.355471381644585,0.201786805992899,0.520514331615407,-0.303063945896796,0.375342352079864,0.0999256987506149,-0.569019937023111,-0.221480225782555,-0.00700517385174975,0.367648194196465,0.494498065065864,0.211410000057558,0.0565625301375561,0.342515225897258,-0.0407184090500524,0.307645100433974,0.115719238910122,0.083850220658605,0.0112333059246532,-0.029594820353692,0.156645551883574,-0.820363054024077,0.368110585018956,-0.269774252170418,0.142101177534123,0.123197336474799,-0.254192083653447,-0.101587719092908,0.394275581768654,-1.21264254010847,-0.777576250387908,-0.697727352559345,0.0767170395412635,-1.23342763014121,-0.622175704446075,-0.949142841549677,0.33871590762478,-0.233550675589718,0.692169942472452,-0.259411212605037,0.503280820643817,0.552476149331043,0.623580385341127,0.562583164020626,0.342414223920772,-0.0073769006918933,0.408873917570016,0.741184236150282],
[-0.00810020540728167,-0.311095200899324,-0.260990369001225,0.396609205003071,-0.318670781708343,0.351293741695896,-0.15416961313214,-0.151672492678739,-0.0457911075398061,0.184868434497827,-0.0561641989901886,-0.0123897729457879,-0.0168781654948491,-0.124072338970514,-0.169602829189166,-0.1748760013251,0.186678655214651,-0.293749796363486,0.262277219575241,0.0245539210074597,-0.116212183282036,0.495976694235994,-0.156987161907222,-0.0351589167883088,0.0996702653963559,-0.616022871330391,0.391949616570259,0.020770133884294,-0.368483742748386,0.311781587959609,-0.0207982182779738,0.0358927570968814,-0.135159808690945,0.152896877086621,0.0216730499051951,0.292704511053336,0.0309630622840208,-0.00394126251137645,-0.269075555860662,-0.58930011731641,-0.0440967022094704,0.581477355982061,0.52878842288816,0.31349514934088,0.779362337154717,-0.32484298686408,0.100971035811027,-0.437881774868942,0.187339798541714,-0.0844046810185222,0.205892948913316,0.0228122572938165,0.1657522952247,-0.119794377792832,-0.244969045987115,0.339225357807924,-0.268728464883346,-0.113241178068406,0.112398671994044,-0.342764341636516,-0.0112517725708566,-0.0441060206420256,0.758256866240961,-0.0529741993112864,0.086400902017137,-0.263272918939781,-0.0143166632291354,-0.333413561211667,0.0501344104690925,0.664046098814938,0.0627086971228143,0.259590638105266,0.0411573925937166,0.102158081133524,0.0692419500553138,-0.157373458074965,-0.0511585619810427,-0.192538603940943,0.149608861475689,0.046669177040117,-0.0690327786794052,-0.207342700461148,-0.432073720548967,0.0714070816442843,-0.225635680410823,-0.243025773731112,-0.372582641865917,-0.22217068492059,-0.483895307670388,-0.0403078548255278,-0.0556880619432448,-0.144598603485917,-0.342963680843654,-0.0548741187262438,-0.188908914377524,-0.0655539914088824,-0.383832524719861,-0.137618266950188,0.130536792064004,0.0700436174945803,-0.107431079086713,-0.12004554339702,-0.146414513425393,-0.353530573458584,-0.169058778046986,0.0139876889044687,-0.128305479862383,-0.00608158720788319,-0.0234038974217724,0.232686258374776,-0.634693448051241,-0.0409903185408395,0.170753017088409,-0.0427549627327757,-1.29959028852075,-0.0325449611263752,-0.142409205294187,-0.190627864396207,-0.0403909656713522,-0.0868886081945763,-0.252344065207896,-0.124623566579702,-0.206725422207192,0.0617690718487317,-0.197263914928589,-0.207934685105961,0.0110520767420414,-0.0573920274129057,-0.0348000832133999,0.114743694036771,-0.16715677067914,-0.275711550912717,-0.00536279745097333,0.0706416782509696,-0.0469670863641211,0.178239985990803,-0.111055689125218,-0.260639844396321,0.12472298169984,-0.21760556485902,-0.034255472232515,0.17720816637224,-0.0531352896994038,-0.0908222556162861,-0.325121345622442,0.115546748947055,-0.0615984362085478,-0.0426046786448512,0.125340865998107,0.151044967168268,-0.13849967095606,0.0623549544573513,-0.17718708830183,0.00483252676250992,0.0263265866233285,0.0487287603724304,0.151149455998735,0.0883184286856837,0.0182538791550553,-0.000623244981319794,-0.014431757521514,-0.0848479150125414,-0.00375096096316647,-0.00381527298758103,0.100963327381921,-0.00377576343156686,0.0275313024190454,0.100951224794126,-0.284401166475754,0.00970226109235871,-0.197412324372564,-0.0496222774248088,0.103515952416913,-0.118510125995922,-0.102275647265404,-0.0508544009729795,-0.109613503086267,-0.0366321154445397,-0.00365200296321374,0.00214457116233391,-0.0675452833094742,-0.0449441664217386,-0.125793449981679,-0.175966558456452,-0.0366537921046194,0.0212644229664796,-0.0164699300220029,-0.148653159514664,-0.069925720832264,0.097964971978533,-0.0198663284595978,-0.0382549267850585,-0.0339323884636254,0.08154854207056,0.0594343837744918,0.220938566388796,-0.0320268008456051,-1.62483551043331,-0.912212086750281,-0.56637443278193,-0.371278087207273,0.440982377984227,-0.608060445258925,0.492294906101387,-0.0155407189627782,0.228880831543872,-1.05041094596374,-0.393243521303213,0.30500522938594,0.158363269708137,0.720890566046664,0.351265140975883,0.677265147187782,0.115715748928825,-0.411950833512407,-0.128903058703006,0.0176699757410051,-7.31885395312635,0.49436318508024,-0.120746019904194,0.0507870055863438,-0.0212595752525725,0.319640500672845,-0.16681758927549,-0.115287705078792,0.104351501897472,0.315820605395755,-0.112164285967471,-0.11265617230437,-0.228163328660092,0.0520762180909059,-0.135971774099326,-0.150569135284918,-0.77704454862039,-0.598973083915929,-0.564817726278201,-0.397021056999213,-0.0296216599064643,0.124755400537752,-0.160137902197763,-0.172756428275063,-0.0783018560625668,-0.257486702087631,0.0754509375538214,0.0293602436234473,0.0694632282560845,0.0133103328018308,-0.0678973559450497,-0.0269592144762812,-0.0569547511828297,0.124568594705928,-0.000214601354566097,0.00402828338155267,0.0295547032051344,-0.0871425185786488,-0.00849227037119,-0.030394929312652,0.0910864138616406,-0.256835279000519,0.0403874943012448,0.0372569487715217,0.408252006623789,0.0931152757022972,0.229199098348391,0.196364578288675,0.64470310520292,0.10671683593936,-0.0548706068358687,0.310143350973378,-0.103815400653454,0.284436150504226,0.384619567882325,-0.0162887760025148,0.221112674205977,-0.605543450066652,0.331833089389193,-0.250696658340529,-0.484731943797365,0.15158818940239,-0.209544642181393,-0.0609168941948013,-0.0421195884056287,0.0461300676575924,-0.314786034816584,0.161050964124053,-0.158502259572471,-0.0222666234805713,0.125616136039416,0.265829279440222],
[0.0244758482075838,-0.222308570527149,-0.29883196926719,0.312211597051882,0.868189651177856,-0.0848579540971372,0.0569606061150696,-0.0194254489521151,0.00876369372032836,0.261779491048476,-1.05388213418576,0.904120182944764,0.473860021491802,-0.731396839640116,0.094765766635501,-0.637458809858399,0.510389256109142,1.03302039946907,-0.534441084081952,-0.159504609449956,-0.171425474300575,-0.211140067811347,-0.195335944114939,-0.361658795998161,-0.289677550458407,0.107341590091315,-0.687076109919025,-0.0777793353280181,0.244680801848983,-0.307254420943709,-0.113327774326359,-0.291134822007147,0.0863327361860041,-0.238130951288704,-0.443181021474623,-0.0500255864553286,0.275531202509295,-0.249308647397696,-1.26456362064333,-1.08641648818848,0.390125102766891,0.187272127750386,-0.011072011783364,1.21842611736144,-0.362131999144157,-0.270111049154783,-0.157127636011692,0.0731160469669243,0.0838416927079381,-0.0170467996928414,0.173252712756046,-0.308822608141385,-0.271352629849064,-0.0752960326125799,-0.146221871543309,-0.0670895767735631,-0.0847916983637153,0.393937704285708,0.0444735387116207,-0.103783565910842,0.0478055137738496,-0.174031973093802,0.132733614104308,-0.0919475159515021,-0.570303119921936,1.18315944019187,-1.99816939375561,-0.0398328517457804,-1.2364533838336,-0.558280100925767,-0.144841367240841,0.398219887957068,-0.167877024902231,0.111095866487058,-0.417079289603436,-0.775609957093619,-0.264245215430772,0.174308117351422,-0.393436363750672,-0.0339108290244325,-0.324089284928094,0.0214995306949474,-0.135839808439133,-0.033597617340101,0.00910179982927346,-0.170635655664872,-0.343552662996557,0.161357868079147,1.05087418588492,-0.846880457885716,0.274721598306028,-0.3197377915916,-0.14223113968258,0.102227579934981,-0.0964511897035848,0.136249328629989,-0.0771981648090743,0.0516964775066571,-0.795841938326888,-0.160748594601458,-0.421853164026175,-0.344237225794337,0.503230276391666,0.037401841016276,-0.340134041173131,-0.175023410157403,-0.257644237731272,-0.0049199404079919,-0.310865447608422,-0.165338930144696,0.0671378034058254,-1.26272727663596,-0.69834545850384,0.0788491723305328,2.65178857882011,-1.22402898445964,-0.0890436036672032,0.17815968052267,-0.262322446363953,0.215465056562946,0.19901761479326,0.296253825988813,0.125315102236388,-0.0930542804439848,0.175173304761956,-0.196097019005387,-0.34812104828897,0.014940139303557,-0.289171339787752,0.0363427490179174,-0.336215033731062,0.244977450091992,0.0644834662824737,-0.341279509799777,-0.0679964272719866,-0.338877102167109,-0.354378809926688,0.398694971982988,-0.249447312949584,-0.0566472567599541,0.218980740556497,-0.140522350965406,0.19323369397903,0.223959658445545,-0.104812454387571,-0.316792595284901,-0.110838534712886,-0.179375557087106,-0.146074539757083,0.140392618827802,-0.0731102715004318,0.282926760333888,0.117919617747652,0.0545642570693888,0.0149301400523287,-0.525945112147976,0.0381159654817851,0.174379656202376,0.134942920338834,0.142097886714674,0.0354939334559382,0.133644919612387,-0.0193048588458919,-0.306347448793363,-0.00752292507981856,-0.406934550752598,-0.236253690385018,0.0662834804642339,-0.0556190552982685,0.0169518051286906,0.0302558763424552,-0.288542175522457,-0.107734124557437,0.0362672756839898,0.014152464419585,0.0214636548634568,0.074005415816423,0.105065443363212,-0.0752998661579703,-0.161376199369997,-0.016574482691633,-0.0317749545009553,-0.0340175498520223,-0.0929596414958609,-0.0259068372384652,0.0410391021225154,-0.363064785034137,-0.603025930641057,0.0955508471783294,-0.752447058836664,-1.43685541602969,-0.746410080368703,-1.37118263954523,-0.240913785034905,-0.0861020323337087,0.24292713329409,0.654332380733703,0.716805881199823,0.973209626973113,0.736782931130247,-0.282313479941207,-0.418539455625521,0.625701214566301,-0.435696720356662,-0.377417025444082,-0.793133346086502,0.699066859846485,-0.0279385069553714,-0.972235201413336,-1.20111337545349,-1.00969569687112,-0.265820750053061,-0.512025723024921,0.659652229985094,0.033730253606514,0.0455115563137365,-0.0443301370183729,9.75758590554251,-2.15639039615741,1.9735542293255,0.0756505003494274,0.239859169903003,0.734594885269393,-0.335231389585029,0.683694375637262,-0.443522602109786,0.240154012337985,0.251718843136263,0.443253370338996,0.382446029181684,0.329537855090483,0.665852296567984,0.986772317237171,0.293141582534858,0.716000045838936,0.0394327257255721,0.770481399894226,-0.270423399339362,0.413301474363372,0.0411172692411839,-0.27797812064242,0.40739349405927,-0.0426644289235222,-0.0822838538483234,-0.153261076261379,0.0655468807466629,-0.135783062728183,0.601180130398143,-0.0170906831278584,-0.0479188207318082,-0.0215919407172876,-0.00241884003196759,0.114524029726681,-0.0280507229849777,0.338788851064371,0.0889513825209436,0.151406463448262,0.152791002844565,-0.219761064703051,-0.0624827543560605,0.0369716656656151,-0.286503952720624,0.0265638063640187,0.385241258717904,-0.244026562429458,-0.130449591634754,-0.279038204439698,0.866274907103655,1.36047617581521,-0.135284410070114,0.227506847886213,0.409687050279102,-0.209830125182589,-0.0980631305060236,0.755959029892552,0.0792674933459308,0.432795250767515,-0.422405876529698,-0.574245040106714,-0.87447175755685,0.243123740552853,0.244330709942029,0.0526224582428712,-0.183210821041119,0.172298990653308,-0.778494293255787,-0.314193911439909,0.303297719014937,-0.219522311486642],
[-0.0328013350794518,-0.446859486913369,0.168068570571435,0.394927128414336,0.165010569161567,-0.0730064382843901,-0.228500801328777,-0.315218033761594,0.595690215952246,-0.652455177859869,0.0725771509272759,-0.467396262305412,-0.04860151052939,-0.160982644793753,0.413286104266662,0.735216982221409,0.337250255466061,0.622886081679738,-0.225190208524242,-0.242266123814683,0.14584247312335,0.473744542323463,-0.415339407312337,-0.064360748412506,-0.560879443666207,0.935131532839352,-0.411424130322238,-0.478387105743446,-0.555210557718308,-0.0490405202862359,0.0139742659492113,-0.186521293988872,-0.0329461425426987,0.213907306439188,-0.234158872030949,-0.00345956003965122,-0.0161378634443028,0.104228496816048,-0.839999287159153,-0.640224629402395,-0.849687779134615,0.0197370567178668,-0.0293458556549109,-0.0460534839672999,0.254087320382621,-0.4990211827314,0.624591007069943,-0.113160377326033,-0.176753945489609,-0.0813487928300221,0.525202411271843,-0.510383937391633,-0.0273416645055434,0.215773518228292,0.0630396945235201,0.349708793863946,-0.271726670836424,-0.478500509778467,0.318735882620564,-0.217834259934711,-0.312097508166539,-0.103571997915264,-0.612826378484074,-1.12271577824422,-0.135476756620158,-0.755793268286583,-0.683381016422971,-0.137317656378958,-0.484131886063634,-0.343478966620121,0.0537035767116121,-0.0544965980713006,-0.366947707312725,-0.305132674935392,-0.459856248334599,-0.618571055246728,-0.19300088294452,-0.311970399738307,-0.267488059928406,0.333212067768634,-0.343420640125507,-0.335491484488045,-0.236539307988953,-0.0177050408557391,0.0345951934700652,0.166689311823811,0.207533776895281,-0.387771159930143,-0.584462855550669,0.0987305524229063,0.1531702535409,0.683555394265652,-0.255647761620155,-0.085482440104595,-0.0763194634283642,-0.134480043340749,-0.155920845290738,-0.411102642688471,0.0144122511039855,-0.295056388202901,-0.169888003883638,0.131945489321394,-0.124061804685002,0.0384940908899467,-0.365223904238872,-0.212511513043732,-0.28159644591396,-0.178787576984378,-0.171478960100532,0.293786003098248,-0.714030030800076,0.00382284238691611,-0.334862179057035,0.129937109280269,-0.823242776889357,0.227662704102776,-0.231501913875054,-0.360512136465487,-0.53654312124795,-0.0364311314927201,-0.225921558129573,-0.177410627973902,-0.171159815026383,0.395135422434408,-0.279349640420113,-0.118882117748782,-0.184211970473422,0.178340624638171,-0.327693070674531,0.626908982561355,-0.29673505662225,-0.338799711426414,0.157759743173103,-0.630612094759623,-1.16861848920465,0.0895649074185321,-0.475757429290722,0.473617808242992,-1.07405665958866,-0.0851336140508894,-0.0459099442375639,-0.0568800786969191,-0.0191444157093815,-0.581865425805835,1.45457556082738,-0.392351231754642,-0.00787579554376108,0.00928612829236842,-0.243920839129529,0.0812573854284505,-0.351003479600046,-0.914785977882191,-0.0187821579213625,0.357610490602536,-0.0907401229496791,-0.297244646919556,-0.00466369352899681,0.319638493302625,-0.0526905743220182,0.197112253631073,-0.0943824254414802,-0.222726086312132,-0.172627263756277,-0.755204183666341,-0.0666453311639253,-0.442706449645769,0.124049507854063,-0.00290360458398526,0.590045824376255,0.0573453882973682,-0.0716732337635941,0.83202750763394,0.115016849803575,0.0765285104184511,0.0768119590739358,-0.0254890838490077,-0.0544339725170227,0.250683860140323,0.0540597030916937,-0.030072638698076,-0.0584334974928961,0.0365585267634839,-0.00809951589863102,0.129427382401024,-0.0153768635750025,-0.0113251280855448,-0.301802211215112,1.65091601155756,-0.35645235567563,-0.58409277203727,1.22690110904111,-0.751525575549141,1.03112557684668,-0.618743207175997,-0.664612148952828,-0.65087359941134,-0.577839948679182,-0.708614682366934,-0.859146096069283,-0.37459731319091,-0.144231255790668,0.224425652471825,-0.21616549457876,0.334049245770179,0.567114629089006,0.110310605078392,-0.469996905016391,-1.6346633676771,0.2505587324576,-0.359920253976738,-0.0966140254898552,-0.519703869200208,0.193800459110261,-0.688138888880337,-1.81360031100399,-0.0494321268245153,0.0621356088972237,-3.74075687399848,0.0842093640293995,-0.156273381346372,-0.62180850658244,-0.380814265276285,-0.549867366420048,-0.00366800543688774,-0.717997834040466,0.151488844626646,-0.172146167837893,-0.343715064338775,-0.382242937900422,-0.181487878663716,-0.505653670766193,-0.730393055414312,-0.398996499923206,2.65056499766558,2.58061510642404,3.28451293546761,0.822039874814171,-0.466832277288215,-0.370801969514878,0.339238182876123,-0.252376326789807,0.611118923725532,-0.221193544501149,0.294314651678135,0.273998850803212,-0.333813866969528,0.279641297453618,-0.016868945555972,-0.354444641501686,-0.973982842988944,-0.132084826969264,-0.498879404688189,-2.31487611152782,-0.0356544874835689,0.845894560843274,-0.432537150436894,-0.685472621653404,0.0544825599905822,0.0815320515336768,0.429664596957884,-0.0390541466292011,0.468666242895239,0.351037110506481,0.450569077912411,0.601517984812282,-0.40734305149784,1.77094022769055,-0.180640661348191,0.154983084496844,-0.477793080981292,0.640830295075467,-0.0816354223652435,0.0286162253281013,-0.7417253317562,-0.543803457946578,-0.406395301582694,-0.775235418600025,-0.117683624298979,-0.470133282567616,-0.0898576388732299,-0.23321393428219,-0.485536258681742,-0.417562150417794,0.280214494398481,0.497423599455805,0.102575403337399,0.881819092260283,-0.508360018150992,-0.508229186954169],
[0.0355356182526679,-0.0826528475282699,0.0586285331165488,0.129237479826342,0.159657414345671,0.225777546658933,0.077122570479176,0.14020012464416,0.141243754464895,0.0955961423663499,0.0570601462079774,-0.0134535259198494,0.0478184290832931,0.112432831379088,0.318353949927869,-0.0764059179853317,-0.145852422695211,-0.183865111508605,0.274158928222256,0.163534012534943,0.0124939751644594,0.228347581720696,0.0579978926342365,0.047001947668862,0.225400999100001,0.384119012273736,-0.0452105708384798,-0.0701327412410108,-0.521228703053939,0.238949448105172,0.00528402037856035,0.118113310364624,-0.114116027309602,-0.183598203956605,0.191722687629474,-0.104859175160989,0.0315309185937051,-0.0302171450954539,0.176882974770956,-0.247222811244608,-0.0369927559805829,0.0890730305853865,0.0863297206441879,0.166340383192001,0.536060921263693,0.150330355765515,0.0976980697274355,0.262040744680082,0.09411540235734,-0.0570269091420545,-0.00326727842605433,0.204970991381921,0.0202687172537169,-0.0116166733308469,-0.0981659137684933,0.00168707826320825,-0.0269373706335182,-0.17889103502175,0.0392948048588333,0.0965136065802393,-0.0117791945106124,0.017113014525619,0.0407591649561327,0.0364869256481756,-0.017962787531938,-0.257849039623931,-0.0193797177111932,-0.208945102767405,0.0857029879955401,0.0404338052637983,-0.208382354703902,-0.0366094110171077,-0.0232453302044157,0.0481148216236208,0.117234213940709,0.380488901063133,0.0606072938822021,-0.110180570838221,0.524445981277125,-0.0130814642072152,0.115874177840701,-0.15532528779042,-0.0517230057296898,0.0885809861222438,-0.314221601787111,0.109468892604107,-0.345592438592365,0.08320352227553,-0.269426779018309,0.0746424877525231,-0.235681206189939,-0.104778623051362,-0.0447431506887967,-0.0184282539638301,0.0134137210670955,-0.120210036832558,0.121538169408994,0.318133309429418,0.525150180772312,0.162158625919857,0.373572062432952,0.354550172109261,0.0907289183187164,-0.124207293439828,-0.0143225340530644,0.0890972389190939,-0.0015332623159474,-0.00464228955464825,0.619891567164026,-0.432227255035267,-0.194689501108408,0.046320375909979,0.348100456479146,0.107683307591299,-0.76970978005178,0.00112037968506878,0.00846905346664707,-0.136530475914152,0.0855593016895293,0.148730160718665,-0.152014404677202,-0.0156410106552653,-0.0880401194928811,-0.271307579332496,-0.169202601952861,-0.179423395153346,-0.214557723020802,-0.26560508050444,-0.292769203560824,-0.191651720732824,0.207724320747528,0.0103578273512678,0.00254572285637153,-0.0237945067483943,-0.195762278534773,0.286062617329289,-0.084407754471697,0.086603935366219,0.081354941385128,-0.0953120740727029,0.24432323363265,-0.138538417813525,0.122191516819137,-0.00113804714201907,-0.0392266995111526,-0.0252296682294687,0.0602442362727592,0.0231665615447432,-0.0434407358126581,0.0556369286022982,0.00493543267045008,-0.0320116832243219,0.0495931082222018,-0.0636163309291933,-0.101665292257569,-0.176461983921467,0.0734448282268399,-0.0247929444583936,-0.00970230102161639,0.0257596395959217,-0.0444306969345469,-0.048430026460228,-0.0619696312263101,-0.0722636998729332,0.000217086284597797,-0.167251595767782,-0.0302553994072338,0.197231045515055,-0.0761060834268146,-0.0293294359034547,-0.0494766528089009,0.0115371374722954,0.0485706301105105,-0.0680983616419372,0.0144758084574393,0.020204878295785,-0.0416284659007875,-0.0244597036017464,0.0321789540432391,-0.0373510596394795,0.0971952725062983,-0.000739431158901303,-0.0455416225363839,-0.0939837341972351,-0.0338264982950318,-0.0224724045376896,0.0626737393708338,-0.0709573581022437,-0.0837497539599246,0.109804079141951,0.0894969298302885,0.119407497903951,0.0999106732453121,-0.143546452995424,-0.20462641693532,-0.206363641298451,-0.314821599456522,-1.25589511186552,-1.4012029129355,-0.558782294344573,-0.411327672988751,0.267202493551023,-0.541985828399637,0.329768651818437,0.168811489864697,0.261960227148791,-0.794593911427034,-0.0545703001923298,0.373563570315448,-0.046318701181834,0.614016990787955,-0.015263046019058,0.27492575303182,0.0921668298766834,-0.408728648659302,-0.109630951275589,0.0334619165376117,-2.18573660326433,-0.0123474712558626,-0.14084820798123,-0.0522403624591945,0.0871781930778557,-0.142282593616192,0.325956418899361,-0.140237666034342,0.489458670066693,0.0864611422946967,-0.0768595217620835,-0.147406975084376,-0.0119807692833441,-0.041184365207219,-0.174515531543542,-0.342374270916221,-0.382203195205457,-0.292953024168876,-0.0759777668913056,-1.39359678296042,-0.346859460028477,-0.843590142010031,-0.135857893292177,-0.348445719180022,-0.637640589956604,-0.337125021621085,-0.967798438853974,0.0146369136856644,-0.653449461950524,0.0570714082592118,-0.498555727054541,0.0191661675052389,0.0988535357954933,-0.499723891466244,-0.83465794808045,-0.761186876152768,0.0088199393395961,-0.545831491913516,0.00394797415024499,0.102181339615865,-0.0563341794754254,-0.297298999703348,0.00713537079631981,0.0244981227051995,-0.049962982421252,-0.195514562529709,0.0394746174667565,-0.0473807610093102,0.0462385838608816,-0.0127138493781549,-0.182007594522085,0.312430791634548,-0.378990538789429,-0.353748894195784,0.201410444095832,-0.660288771882158,-0.155535109218011,-0.798579844745863,-0.175998746265798,-0.570682214348234,-0.586496536231203,-0.0376991253072591,-0.853223468082468,-1.04735428729134,-0.164686141014461,-0.170412324701642,-0.65450730519719,-0.309003656895147,-0.363958764549145,-0.0328556734349742,0.142556939580895,0.150080657586063],
[0.0309688203154568,-0.0283329602733357,0.277736456916187,0.693944970313793,0.156644049294936,-0.978738292017507,0.142455758704753,0.18656327864501,0.444083080348221,0.0791610771258974,-0.0216168861799434,0.570137115317026,0.102069872822539,-0.536443856790643,0.0059067925386974,0.257269267812711,0.334817662968994,-0.135745098013051,-0.630985301510973,0.0253244844540313,-0.0969488243617955,-0.423482701425329,-0.181466579944871,-0.00958489410883073,-0.299367780598532,0.0310736284018075,0.115458432070372,-0.49900301642721,0.0726681413748571,0.0463504140334875,-0.0521898988669643,0.0431972188953589,-0.125401740463055,-0.307146900708973,0.0845330784228204,-0.132356240683052,-0.0490401187547388,0.0368417000195029,0.614463443138579,0.0500715478861729,-0.148254241148216,0.732132990852683,0.10412279599689,-0.0746772771546857,-0.58919549489952,0.0665435242388878,0.0590225286556169,-0.0843623338083427,-0.0446697426915185,-0.0639949739456192,0.0455347700212969,0.205760498784261,-0.0180780767927403,0.0171130402402881,0.163730306573071,-0.169056631715391,-0.119376046454344,-0.0988643339716476,-0.0893793583252135,0.0848890012661327,0.149654588266371,-0.00779223569400172,0.289936744983823,0.0830313509054322,0.271568388979453,0.370647956758526,-0.184434693907871,-0.156590067966925,-1.05941398466536,0.680656515554231,-1.0305427062648,-0.153288127634921,-0.0645343696464265,-0.216552882942312,0.135515613628467,-0.0942167164296119,-0.385520678967399,-0.526790536911353,0.0379788437027263,0.047595357024802,-0.366855793041853,-0.596193458499104,-0.956249759899596,-0.303893968931249,-0.775537197736931,-0.0948549825758849,-0.828729279200399,-0.275026882780414,-0.364202298182629,-0.0177001504214233,-0.54349110292706,0.0441080103290461,-0.713732203135849,0.137174789688502,-0.475247188788611,0.0395872899030337,-0.204972265724784,-0.0987146353392137,0.432438481775674,-0.103511377055416,-0.0964781197246431,0.174327652794275,-0.0556799758070668,-0.287578383097643,-0.574053614399496,-0.243813810203715,-0.664735266844319,-0.14806679404613,-0.506525205332033,-0.0615370999211344,-0.687555292072889,-0.0304798098120509,0.566939255789234,-0.0823787909832388,-1.7884823260941,-0.199301188862672,-0.584873075103733,-0.222105700265432,-0.0738108643684604,0.365618483819166,-0.61009991331205,-0.0215706981518911,-0.70638235104058,-0.0650237072770614,-0.289277121546297,-0.518829318810026,0.511897958174372,-0.0726020015891585,-0.0890643055929365,-0.218072671794517,0.216826423328167,-0.0296291575089031,-0.0764452824156458,0.0544074980853204,-0.0618897955625483,0.205118205417333,-0.141840269564459,-0.128088604170029,-0.118308641349545,-0.0997691060144853,0.0835507937199377,-0.138211482972997,-0.0873464937842641,-0.000106179109531063,0.0872548904285516,-0.0117603531048079,-0.166507612179943,-0.143242382585471,-0.176006364943537,-0.0516722908343735,-0.0693438043086111,-0.221056940536895,0.0939039689777563,0.0902796562378722,-0.239575792852101,0.481258732708912,-0.169219078690781,-0.134646980476994,-0.153679359543639,-0.0895074682718996,-0.157388141174778,-0.131879364279747,-0.0745092085798897,0.00952489271641119,-0.00650553433404324,0.0098252676233264,-0.166501586580576,0.108317040194344,0.0246061733919168,-0.0618131204325196,-0.304755723507685,0.0823984304509522,-0.0470570484034151,-0.285688390238103,0.0154065224571344,-0.062569497404645,-0.127586711338957,-0.0801722058070708,-0.0294182516700501,-0.0321260966098727,0.108462182068591,-0.0823369501211255,-0.107935370423549,-0.0144907089935303,-0.0059197265575811,-0.027060622202317,-0.123665464103881,-0.147837067229257,-0.303579188711521,0.0193204916590994,-0.220322602995065,-0.16649865172358,-0.266182666037973,-0.121332092815764,-0.190805201656907,0.841552497058661,0.225662400970678,-2.15473862056872,-0.16502858306142,-0.627641676809618,-0.25695833648991,0.246774332257982,-0.448176595623885,0.319431413405487,-0.245390956681825,0.00710489269956783,-1.46322818853643,0.35394996690794,0.244064355384753,-0.190769534818698,-0.328389265642746,-0.761921327345878,-0.170015932188853,0.399138134635553,-0.0978521841905921,-0.104756295138678,0.0457637692078979,-15.4011684049826,0.979254230993735,0.50687342942292,-0.356799573637685,0.191820568653547,-0.0932118613490776,-0.0611304188267967,-0.441982881382012,0.267116467763003,-0.125481736016438,0.474993976478612,0.251991617151403,-0.1420454282817,0.661027561574098,0.318342380644865,0.112707552724941,-0.995162943598392,-1.22968188450731,-0.897304285815162,-1.98003093906055,-0.00596624836820431,0.12566632533679,0.324266100934893,0.362240283315233,0.199491232962868,0.454720343324119,0.117280302773782,0.031263971954344,0.171305163742545,-0.0146800999414517,-0.100235479118156,-0.00384247254793604,-0.0301855252720155,0.363669059531162,0.0376315389327728,0.0210927010100759,0.0758577848287657,0.0623454668719763,-0.0145940645522874,0.0326582976589612,0.0960967276167721,0.217967649191991,0.333095779770718,0.0806419432845504,-0.0941517906979767,0.244381088386967,-0.291949064931344,0.0264309709693302,-0.0277099202413428,0.188442481137703,-0.277900279886024,-0.10937896745048,1.75847354937068,0.0384668613661531,-1.58115674539415,1.30413123836521,1.19746620905659,-0.413908562362094,-0.392019699820551,0.641162286269536,-0.441895755214697,0.431262543119778,0.144035602325705,0.941531823589811,-0.130029219947165,0.0416982423183295,-0.221588041275939,0.688950211605961,-0.27769719572339,-0.62912900825294,0.159213208585235,0.121742852824476],
[0.0291976901875868,0.0254406386666707,-0.0419028945208796,0.00970930777895725,-0.110389605068297,0.264791525972415,-0.124628307154649,-0.0942269044365478,-0.104062782502805,-0.295933038222144,-0.0145618263253252,-0.0306300235624925,-0.116049071757407,-0.0455845188187612,-0.259247447249551,-0.0043939250095404,-0.130241479089469,-0.191068749827096,0.276176181880396,0.0808696888206057,0.0438961286266896,0.536192819994378,-0.0830012662113223,0.0046297950924857,0.039349415047741,-0.281516604358152,0.0136815329621103,-0.251535993229087,-0.709280409280834,0.252588154060272,-0.0457856760615476,-0.176395950393017,-0.123445134864644,0.343906704277263,0.0814548722957799,-0.176849959083312,0.0658513390451801,0.0276103642915664,0.213441494653502,-0.191126414151394,-0.211702986347238,0.13569057140333,0.145241363118922,0.0544913932884671,0.581792817312683,-0.13316190412671,0.184062060966327,-0.0443448943729713,-0.222558544185675,-0.0582820488930662,0.170582755977539,0.0111038360599367,0.190520391121401,0.352473371218303,0.235904832318195,-0.0113033305769013,-0.0976791414344668,0.138464133429039,-0.0339485720021305,-0.0334712897170524,0.0842280943068738,0.396102753957459,-0.331125820324825,-0.0452689442797231,0.0591485914594853,-0.0803834155499785,0.105943371229903,0.150716749962094,-0.0377019991360536,0.125926970905155,-0.25972825365726,-0.026436563334335,-0.180216468185064,-0.0743386241023739,-0.12651090426892,0.0946825121050229,-0.114287365630771,-0.174030476599513,0.0833060463827544,-0.155717541196568,-0.0688278335379658,-0.238206288040755,-0.288692869146302,-0.029208727124056,-0.587660338806818,0.00540893055086847,-0.541358711015017,0.00162261091807943,-0.486156803545465,-0.00484065740925798,-0.357333309670969,0.0572919532837778,-0.20174023399694,-0.298557772987091,-0.145804663055664,-0.329863378666695,-0.215509659365099,-0.119195852546041,0.255499469266013,-0.00552061580246498,-0.0930691406997575,-0.00808575250671705,-0.194932164737939,-0.122661825161109,-0.166846623107227,-0.0220486499894845,-0.238890772562143,0.0669158005280709,0.139511378112725,-0.251715637857412,-0.505001540609228,0.0545281741458836,0.101809594207666,-0.262848281681595,-1.36684399516013,0.112086371477784,-0.178482815633714,-0.23868021550533,-0.187387876476753,-0.0748660943479629,-0.259158051103714,-0.0831627262572851,-0.349475677689194,-0.18857246941497,-0.116049996783446,-0.173696396053605,0.218820536513799,0.276509849905952,0.201923621490153,0.264207216747291,-0.050325526435912,0.214854758475521,0.139449060048856,0.00743338321529465,-0.158167699558723,0.118099455675706,0.0581168718220078,0.252954903103909,-0.00574932272619883,-0.068861200726128,0.0630125200595922,0.143054660629524,0.126013653580871,-0.27574281458693,-0.12854552535344,0.160239278923582,-0.025411212679519,0.0773598776158017,-0.142465354485458,-0.00883159547502061,0.153062556782645,0.139139858353365,0.0714354572250835,-0.104340747916151,-0.0533376819741952,-0.259898763410117,0.0378557068233835,0.0437555964558107,-0.0230257288966856,0.0315811425381929,0.0440695877676172,0.133539111674274,0.059872835156282,0.0533786955760515,0.0693354070018991,0.260639312370468,0.0220036167276889,-0.0735718104927359,-0.225650575598138,-0.026926773238048,-0.112885089818915,0.151446127752783,0.0304088167548246,-0.080412076706788,0.0379421763926812,0.0380988356550764,-0.0885820671194349,-0.198636804964908,0.0235159186491152,0.0246106195704023,0.0938433545808654,-0.0383193783474688,-0.124686395628527,0.240789915533078,0.00550314397110682,-0.00833630354713556,0.0705273723543911,-0.0481977208775032,0.0695838059580708,0.0622993548748112,0.0813587776845859,-0.000757749685169114,0.0117748838489588,0.119476883230427,0.0151597769283778,0.0278502952091882,-0.21550623821874,-1.5447477990009,-0.763094842559809,-0.490680820229675,-0.296298015001657,0.458889817905145,-0.573708508679927,0.589624280161608,0.163145427827955,0.469921859626815,-1.00571249207647,-0.0119487061854631,0.549939185004589,-0.209351754988138,0.432576632726901,0.258554584598148,0.651375841208027,0.8284662927523,0.221777725510974,-0.0675803173412785,0.0176812208569356,-3.71948987357076,-0.533613856037597,-0.270661503897962,-0.0259559520023594,0.23747693303403,-0.114986461120806,0.0373277659718207,-0.173828782590592,0.358401870138097,0.0259394551091782,-0.134223682422577,-0.13728846138916,-0.0935246177489116,-0.0310442290465258,-0.278479482819533,-0.199996962373895,-0.445235033057353,-0.349602347202854,-0.409389899981449,-0.716899179034835,-0.0575109856051584,-0.306565834887034,-0.227084129606813,-0.216428728686157,-0.142053006392348,-0.1707339777362,-0.247377210725111,0.0756173338571468,-0.450723603237613,0.0935898792578,0.0106927172245752,0.178468911250159,0.310791661637501,-0.136519475042039,0.0723835716610131,-0.115345428415138,-0.0168276276862665,-0.174874829820321,0.0147350838198244,0.230898496778531,-0.0263714177590404,-0.0875444350712869,0.0628309559685439,-0.0133630714361335,-0.239534356799267,-0.519867094311186,-0.256401671639277,0.625995609767146,-0.103175914769858,0.299455395949836,-0.203894958618931,0.361463750296362,-0.423515127559534,-0.318521115980567,0.340567048145159,-0.649701018383953,-0.194596255254026,-0.545875507745629,-0.267069093909695,-0.0945158710666239,0.983613809388704,0.616282555663583,0.441687317890762,0.272057880201829,0.163046404181903,0.200088451039598,0.398080707793412,0.561982350579139,0.0874686743754013,0.467079854809973,0.0662282605568669,0.2323955214697],
[0.0225851317432759,0.199810837515909,0.0306469982740467,0.0855971670585872,0.0967238668183439,0.288657893856232,0.242897207115087,0.0595700808503348,0.0384652914285693,0.409294525890732,0.0175085432961208,0.337032227243515,-0.0261645001295589,0.0775946923070851,0.371246446036188,-0.00610813879647954,0.0606981392661811,-0.0998384214026087,0.396563854572038,-0.00716223813139409,-0.0993641137486754,0.242592261181598,-0.0721025048638185,-0.0732883188393136,-0.0200688656208064,0.178813730501406,0.0912791989373898,-0.632917414298459,-0.296446998177307,0.416365476587922,-0.0296668835041297,0.0664565127273447,-0.18179314014385,0.228223509856403,0.117972922958172,0.138595974848136,0.239334877791887,-0.0304295220559339,-0.219166702019386,-0.413941561989209,-0.252393900992922,0.301493219812367,-0.171688691752781,0.275362488362939,0.649140352021403,0.39719696500784,-0.156805748092453,-0.192103472897107,0.124787402276449,-0.0681645785826987,-0.0644687993400734,0.0556690344688226,0.102141116383156,0.0678995227197016,0.099460907543438,-0.0154553432910536,-0.0991520277044566,0.0097950409382346,-0.052977302303189,-0.0375068591542564,-0.044086527802806,0.315055199438134,0.0480476820687347,0.0208814938410944,0.104234994506486,-0.100198018754327,0.0179004225925098,-0.319468727843319,-0.417788095874931,0.414392035296168,-0.617372448234225,0.0150668715398292,-0.214785059908051,-0.0867968393792758,0.0917562202448815,0.235084466917425,-0.0920804526936806,-0.0731630062844842,0.152678909805302,-0.0871369222620851,-0.134522329878495,-0.135312612407306,-0.233603506204399,0.019661467687526,-0.570005208374192,0.0744379442641459,-0.446045617004677,-0.00439412105544121,-0.150906992678365,0.0824885083612234,-0.119996005603058,0.00247477211088103,-0.231988483265572,-0.0585876114431739,-0.134472822389357,-0.0940594852171997,-0.16482807125338,0.154226542806237,0.395857225851732,0.364346452629696,-0.0870887450778007,0.243370460444036,-0.0588415713234006,-0.335408379670548,-0.209935844656644,0.106001110320394,-0.239085130466882,0.0742745532706491,0.299392436907721,-0.0297297833569138,-0.388235502895665,0.090973740918254,0.291946940157671,-0.0405811119683566,-1.25568039692776,0.0934330335534636,-0.179575893724255,-0.175777786629583,0.0130656324444395,0.087115663699052,-0.161547879580481,-0.0281106832391687,-0.393843854971365,-0.188114835440662,-0.206256491232195,-0.00264446755798621,0.135459732000261,-0.047080374847068,0.44251528377584,-0.0505702577716188,0.0641368928707202,-0.135044051246352,0.254349908420224,0.210645563133056,-0.151800550720724,0.403011614378528,0.0496648323055543,0.0320146759269871,-0.118229351420063,0.00439307171379372,0.188143843663169,-0.373439490161675,0.14724838850171,0.11734288490453,0.048909733266615,0.351716179445082,0.0663361492358724,0.00887928532602279,0.0667451049842561,-0.075387551179683,-0.257574157184783,0.152690357517101,0.245426786814687,0.154436708106098,-0.0761120977009433,-0.209754502194158,0.0387549224943257,0.0234238740836258,-0.0680057148757751,-0.0363186564071107,-0.0457872135039229,0.0429023956787495,-0.0584373592249337,-0.0678587316947448,-0.0146954101115218,-0.0392376125395646,-0.11785612377518,-0.0956797474438757,-0.03848050064439,-0.0752225770206529,-0.147917305391451,0.179009380781259,0.0671806359618394,-0.143236530802136,0.099387822927126,-0.0101106552084915,-0.031307665114251,0.0126958624923075,-0.0391152487875954,-0.0860246602466228,0.00228178415143846,-0.0366236352112895,-0.0736203524002712,-0.0459169991479846,0.00676237492865326,0.0125512312161788,0.0838868617038603,-0.123954760212755,-0.189718959867104,0.128020481560327,0.0895535751537836,0.0984588050137567,0.0943727294293368,-0.0327982552537605,-0.162039420145901,0.0474482146004435,-0.177496100764973,-1.49526487822661,-1.07592559431909,-0.746674739317722,-0.347848320325309,0.391102229233809,-0.524706708684645,0.54634953385891,0.122917039221245,0.391468792623574,-1.08713181280438,-0.00315335778776031,0.612020564053369,-0.792167817415425,-0.267443620308556,0.0818334021340634,0.422370032208526,-0.59311900235725,-0.272261899857218,-0.0840328137577831,0.0658513376851971,-4.17274125878348,1.01591963161529,-0.203567769010915,0.00231142268505603,0.0368569642368072,0.0145018695165357,0.204311867887636,-0.0315939221207309,0.509408558719647,0.202141745219141,-0.0263587719937926,0.0584997003318972,0.0588596709593832,0.0563422508361191,-0.100242724865789,-0.114910547181084,-0.548185120447519,-0.556013538538038,-0.349935250881743,-1.47845452760433,0.137959844718336,-0.34144995987887,-0.113533599102284,-0.301224954525306,-0.224095250138206,-0.203493032891361,-0.398480164609373,0.293427374407215,0.119126683828012,0.0205203277237233,-0.892016274776054,0.198073411306754,0.247161921120383,-0.258549785639101,-0.276741989474931,-0.224104144522998,-0.015048069530465,-0.374510679880623,0.0975351252297131,0.0554968375429744,0.0151544040537182,0.00207057277525436,-0.0155296255416503,0.0501107072421586,-0.064418321384194,-0.0119896538518756,-0.0792786329149518,0.328852617223085,-0.1853860437238,0.169950784556642,-0.0747283847121093,0.727124212691239,0.408989426447318,-0.341300218342655,0.0109503240237684,-0.690706712374189,0.499097205657952,-0.41480640626642,-0.171950737088716,0.0610504796690824,0.29101979287968,0.56774484539481,-0.37002565625471,-0.406328639748164,0.109328771346569,0.0861294034830558,-0.0247192405353291,0.12724939865218,-0.324424517776363,-0.302992536916094,0.194548597251171,0.293081698568379],
[0.078401251841932,-0.0293137762273338,0.522620497328903,-0.806158254208234,-0.371734636310568,-0.553762444242514,-0.0706406561435718,0.170863738024245,-0.689382286159387,-0.0787406212276479,0.329764368841285,-0.868806271095338,-0.193260851574006,-0.0221872880439125,-0.391109274953723,0.624281836829891,-0.430255821223125,-0.437283713455774,-0.256881086816541,0.0222557099486005,0.130396636672637,1.14079002894693,0.111051085851948,0.0922525444535768,0.16837763666782,-0.735974336592728,0.0729081134890765,0.421256085883126,-0.17796866966656,-0.565842514322617,0.140140298448446,0.296627500469421,0.0617070937973039,0.206262052035785,0.213392201781172,-0.0925496514181797,-0.155195230337241,0.0615250788072391,0.752303112308639,0.563626017856019,-0.114235467452647,0.235042027668313,-0.703039248836101,-0.718508627741161,0.0669859882186171,-0.325001463154256,-0.212547895024525,-0.0174796682978419,1.38477412027193,-0.103849061923274,0.171175123399993,0.087797375014939,-0.0631135888899872,-0.245502093199584,-0.144447083102119,0.680302257316299,-0.0030490616990806,-0.0957882328512074,0.45120337228932,0.0232656077307001,0.0626575501882817,-0.0986563348389365,-0.384987919723314,-0.282376661535843,0.914698218160234,-0.125045088783008,0.196609861503013,0.170486996499677,0.494730587442912,0.206238190756108,-1.42486351179209,-0.294400557248512,-0.486130247173261,-0.13676573159311,-0.318532426152005,-0.128036145708689,0.331452147044935,-0.175542217647914,0.417976508659979,0.113981516081876,0.418086513181194,-0.33139175399891,0.265142233998104,0.145083498709898,-1.5507609908934,0.540324208531451,-1.65169804082973,0.193338832696388,-1.24680731318451,0.160730626392604,-1.29201888748788,0.491632227797126,0.127141053027317,-0.0710904495041664,0.220108915616839,-0.314391911815225,0.0457868101309339,-0.147319737734024,0.607690811541981,-0.316185501379419,0.399829383914933,0.230525611452078,-0.674408690383841,0.280537320719676,0.227806867859999,-0.162045679661196,0.226149870928033,0.11158746883681,0.350275138095613,-0.261690525814959,-0.871076554461346,-0.10933217210121,0.298532855211034,-0.332953294734042,-2.93069809949552,0.159284313163015,0.128678160818247,-0.14213852997837,-0.40346239480292,0.13344451378495,-0.117212480964797,0.302225613950968,-1.07044730259765,-0.271980392034817,-0.725672420922508,0.356553493244234,0.00112287051838884,-0.501375672380835,-0.258859651468154,-0.465299579458703,0.23285265129426,0.224523609543112,-0.0232048644165582,-0.0745865107553844,-0.0410498800159652,-0.0212042089217201,0.0385678323008078,0.124576096998472,0.349782042207155,-0.192836243571997,0.220447171700743,0.245970572983208,0.166187110427149,0.439378369634647,0.219175095430358,0.137433212715306,-0.0289316872218301,0.0918509961474326,0.0905248714360282,-0.0775537161697533,0.0186243883512096,0.0744486091954797,0.199067918257132,-0.253681861532184,-0.356590189537498,-0.12139151956523,-0.167585104040223,0.0348001413726229,-0.235445572504846,0.263624830620766,-0.191107975535934,-0.309214456928289,-0.0774814830489835,0.0955789598479695,-0.0704218257110253,0.157508363824208,-0.151917223976851,-0.353819357533541,0.37630387685393,-0.0351786170173293,-0.0785413714217181,-0.109072099261965,0.0478313788204071,-0.0933166604989355,0.200286537189063,0.0333560586588202,0.00472288840718613,0.128160038756259,-0.0105013183750818,0.012326958781461,-0.555318518778057,0.00584506178708977,-0.104134340458788,-0.0405804761540351,0.0186975032638574,0.0312114095526225,-0.112196299830624,0.842385881813618,0.190546517024847,-0.253066200129656,0.68185409675871,-0.415667367002067,0.56516734537781,-0.0861635029964625,-0.0738299710732655,0.304321771890122,-0.0145420184836011,-2.11851148647246,-2.28526612671544,-1.60111989920251,-0.386457526244954,0.463779566472513,-0.866088238112649,0.517949003520379,0.477378041684265,0.533147432209902,-1.47234321752981,0.347379861434253,0.717770918673137,0.599558924393767,0.70056377446006,-0.504230073061102,0.197651277383624,0.779099043935912,-0.371837339470124,-0.165754741568908,0.0444098268947725,-5.33383553027562,1.1230281851521,1.05747614404143,-1.60548487517982,0.930544639214347,-1.0437893465737,1.88418283655885,-1.65301415605907,1.3392570521459,-0.878708165540956,0.184219446989275,0.111584680669325,0.456135986035875,0.0246611144753785,-0.0820686632794638,-0.126652922442148,0.57352852307017,0.472639093005645,0.910648142914724,-1.48448056799024,-0.197110564712849,-0.0217255712281306,0.234106590858267,0.0949313060471561,-0.364107633524869,-0.179643782685668,0.0335289247377357,0.172362480445907,-0.268702871423904,-0.0275352462928977,-0.0812663329754518,0.19019366585093,-0.0101229507826628,-0.0833018324425327,-0.159516408882334,-0.0917597844351173,0.0625511679916194,-0.0345203700866912,0.0149240095235525,0.0271075090682284,-0.0420874437852283,-0.186110343496137,0.266154376603074,0.00590199916460696,0.174514744617043,0.797267027825276,0.0997005418945895,-0.22709904159373,0.281171101703132,0.322231520963977,-1.14793621760343,-0.185446050890521,-0.285792665363515,-0.183976732059656,-0.318354991844493,0.0431997560537178,-0.432734792165493,-1.56713421755485,-0.158601609548338,-0.692513565095896,0.192664984793399,-0.596238100016304,0.658727153403393,-0.472515534198555,-0.233920173756325,-0.0892476208086989,0.503010952084531,-0.407863641716608,0.813682574687645,0.112572611364914,0.0217743107128701,0.15582148303436],
[0.0328316432983253,0.0937076349325347,0.186177450447396,0.644722505637491,0.0356537888117868,0.371028036685179,0.0511544235774253,0.0516564150710131,0.450361953571973,0.334663760582894,0.223362697262809,0.0680892472364779,0.0904770048778165,0.0594128177540349,-0.121947381023712,0.143962581142753,-0.0113397047668029,-0.256009591272232,0.569242904729149,-0.0967274685242283,-0.0246871456731324,1.25154368326636,-0.245751166861363,-0.00971171992612765,0.00977312357831228,-0.215315236615726,-0.0461615303233761,-1.35176844532581,-0.322781144067196,0.385074782893675,-0.054702073360522,0.233361116031826,-0.189848424358518,0.50084243710017,-0.151268593802003,-0.17171715616135,-0.149240910730396,-0.0263629485018734,0.967829558293325,0.568723226331597,-0.151586802987465,0.471562112822504,-0.0819371404915872,0.00442397120425483,1.00213999903822,0.101178776631631,-0.214190462897785,-0.201738817528502,0.071307376275531,-0.116709873210113,0.406157164123521,-0.0593435381273586,-0.198918072152657,-0.132050200951504,-0.186788567072426,0.594575437822258,-0.411748791019662,-0.183187343711837,0.418022565289635,-0.259430833103589,0.0229672129946208,-0.268062504742359,0.353255712413697,0.0520929567099521,0.388579545597488,-0.249306038578285,0.352829278639801,0.316865347404167,-0.279030534342661,0.27103963897922,-0.45883349922242,0.122902931371258,-0.745596716974994,-0.310590051355704,0.0182997288066319,-0.121372926311902,-0.0535248880139508,-0.130445583550483,0.190694951851548,-0.311894977950658,0.0263651931642034,-0.264170812701041,-0.0933675863036421,0.198784499461363,-0.35563053139297,0.15259910249123,-0.374787326431843,-0.0554174013352678,-0.657368208688081,0.144359848929441,-0.358016501555431,0.053260732741402,-0.315699865563472,-0.22101942253666,-0.242223474167585,-0.384237633697913,-0.432093243318165,-0.182307881383616,0.0992075579299481,-0.1290698036796,0.00877124568438098,-0.048223416929076,-0.464044374353095,-0.174001773589671,-0.118954092057117,0.0851860987541194,-0.186676487785156,0.153302334123467,0.0136825408138474,-0.571733777833109,-0.648552468611167,0.289745767590929,0.095002749049474,-0.347298140558828,-1.95577233372193,0.262695672353359,-0.0791540585834478,-0.248132388877187,-0.4788026778569,0.00879141436611455,-0.375993369163403,0.104933561443777,-0.6974119429826,-0.0595016950664415,-0.170111547495433,0.127710267946515,0.267100039050781,-0.0114768450780084,0.390947820766915,0.066563095194159,0.0827283745985526,-0.131597727007073,-0.115529582786629,0.268878376035325,-0.215966158617971,-0.215218237117298,0.119473314781818,-0.268326723142994,-0.126714913871811,-0.182803274816279,0.135913453433289,-0.282201248129052,0.269893825611452,-0.197372663544382,0.184987188270818,0.271437757913267,-0.0224325579543276,0.0815229682613041,0.0324296382119726,0.209064570323306,-0.0630565724226794,-0.0791157580058229,0.0365374234833469,-0.0375217685389575,-0.163909068955624,-0.226196617838424,0.136126357791845,0.060508881805752,-0.0963182366763608,-0.0842563389083273,-0.064251842072399,0.0942518751178889,-0.0397714066500984,-0.25612465946441,0.0336456864098788,0.111131627506892,0.0526838401000916,0.27761829893378,-0.297977205468229,-0.0329460521002355,-0.255868604761007,-0.103528624017875,0.0852445087892384,-0.197067454865741,-0.0616066741993365,0.0387569182759308,-0.0727546364541813,-0.253449923933579,0.0131990961185355,-0.0164889488046651,-0.012580539098071,-0.0424064645219415,-0.229438655811794,0.237312526236596,-0.0135844211464957,-0.075672683993584,-0.206699931336651,-0.1192963996325,-0.0283508609245855,0.00611401866047949,0.334740147553485,-0.14019077647883,0.204354753260228,0.158451776227902,0.0660357131822381,0.105142562437908,-0.253283396429128,-2.0572213250582,-0.928545454158258,-0.895943717787414,-0.191602058322514,0.795195647731887,-0.733809462564456,0.893043254184944,0.600436651998848,0.978997332528299,-1.47267029461497,0.595449588336151,1.09651575374313,0.281894330707339,1.18934793192105,0.505378609444194,1.07312697767819,2.3595594700059,-0.135984741124016,-0.21978699586709,0.054443185731667,-6.14060735752063,-0.0962816785972034,0.213133771038814,-0.0572511304359844,0.12822334458275,-0.172598137741959,0.513070496824059,-0.568766296285536,0.846983014833985,0.128179194525345,-0.23150233976597,-0.171911906605023,-0.14272244109565,0.0336980984022463,-0.257259792385725,-0.066751286659482,-0.72825643587577,-0.67031737133841,-0.788735427016482,-1.24927743419155,-0.0759213556515063,-0.369255122331267,0.0352844004772981,0.0417836117497437,-0.133847121110481,0.0451843578630476,-0.266617308234649,0.0485785665689184,-0.00604175912499105,-0.0073026413876215,-0.159426704323851,0.131586661753257,0.0264435769773656,-0.425312130396575,-0.0406182353472364,-0.00794501885837785,0.0191545270456862,-0.551274042533969,0.153994241452456,-0.112660231968629,0.0229768426391558,0.24052013378186,-0.055450590997889,-0.0376411236302489,-0.504247682156508,-0.340612728323211,-0.0283776367337282,0.416496029052676,0.194270044657376,0.536730401855248,-0.620492570334726,-0.000325854188622174,-0.578187915441667,0.0805248112214949,1.04000210853021,-0.138004118267461,0.236221657634428,-1.19715390986763,0.429916398750138,-0.439901760901009,0.49859042614596,-0.1062658676155,0.875198734038428,-0.072178554085869,-0.361736912115943,-0.0582967502388483,-0.0505696732427508,-0.105746895415174,0.713259321918697,0.505593045630904,0.142734164441759,0.283622051506826],
[-0.0292590126402674,0.506573345736357,0.481483997726486,0.104293938596069,0.269890197648763,-0.0236427730820053,1.30462649470198,-0.993043294397307,0.325254770393129,0.545981415910444,0.364514334830925,-0.78848557290364,-0.0168021189919762,0.427564957804998,0.093570899724839,1.14735772243519,0.292102559522465,0.478762745944623,0.10952172789343,-0.776529643959226,-0.376330208439774,0.123905233907735,0.24001710019421,-0.539480169447206,-0.931071459751547,0.986038019178785,-0.0456131200044424,1.27959353555935,-0.330428427962709,0.509318893637308,-0.0385287207139182,0.0658127474548718,-0.0728023006042154,0.624940252933717,0.475812828061464,0.0591143419253168,0.430171766019021,0.220246091060965,1.31372261279513,1.43066389023139,0.676394812911432,0.0889352560871071,0.479370243352005,-0.0594383153492672,0.422879774888084,-0.388641172965774,0.475738749955952,0.225895083538358,0.0958687718992732,0.0848543585517149,0.625553764796782,-0.29969500898039,0.286014837325299,0.032537629295855,-0.142223021291648,0.217524360062704,0.144845159311603,1.11967105498488,0.013949221959716,0.180617842644363,-0.287798458175048,0.773521586046967,0.630248118058239,-0.887637572176739,0.319894156332447,-0.143280862857422,0.795092352407847,0.908483779303602,1.04990734925416,-0.0337954094087106,0.565142060516489,0.0818435654270999,-0.0372824892048145,-0.879052356836438,0.458584165909691,0.709782646016665,0.651367286030887,-0.295821359723587,0.374072065260361,-0.109350648733779,0.730573849027295,-0.332752173529359,0.620852926492899,-0.0177833003087357,0.12242126195542,0.490043063834208,0.525419423638443,-1.02632854449391,-0.681987512993565,0.448787910117722,0.202748168895048,1.09767620358211,0.447079535811016,0.249218857396148,0.495325469543041,0.0710541375375574,0.28287808075828,-0.127904245197098,0.906358345003643,0.602704005041795,0.530973338669798,0.530837040971214,-0.467031620828316,-0.272537738705757,0.63630843521181,-0.312697763718233,0.696389743605077,-0.239654849187547,0.427871395229767,0.31851658284122,-0.0377007951546245,0.588342601905646,-0.131970093191963,0.52052883937421,-0.577166935998336,0.577017823712166,0.481660154683268,-0.590949708626503,-0.759044192609412,0.0792788809203966,0.00344059808207932,-0.635282919448876,0.195385962929711,0.964514410932763,0.349244134350794,0.398829032308133,-0.527258175104346,-0.015825020274634,0.394518658883557,-2.14557592448989,-0.415108628610371,-0.143425417849063,0.350713230807492,0.287640619786648,-0.629732252146592,1.18577793555012,-0.410857385643646,0.599225089621917,-1.11606385742715,0.645382865316974,-0.150019186615558,0.668684457879418,-0.103931458044122,0.0319932120341245,1.74935322136702,0.54152556926986,0.101705976033878,0.0712604447931444,0.224379771591764,-0.577550797244766,-0.0882564749816495,0.285469515949973,0.333966846131367,0.211102185459852,0.159138491597796,0.249668287119162,0.0827303928027698,0.214143753781559,-0.115980867000062,-0.81619178444359,-0.0246002765405811,-0.165242585188116,-0.143990191840179,-0.755537633676282,-0.276230307184442,-1.61704135516144,0.260856033668105,-0.0834493341217934,0.790543661950824,0.0453967989582442,-0.00927600346499072,1.03500026224086,0.133970321981156,0.188888363712461,0.950989273615568,0.0617731533150231,0.0429996593272176,0.778910174512787,0.0700648558419012,0.068266327187098,0.354022699626593,0.0251157600559011,0.00536963953867729,-0.508935987406341,0.00799344231372247,0.0436017850831137,1.32752355669424,-1.8293500665692,-0.0813911259596802,1.40467498098987,-2.2332477344702,1.05070372603153,-2.59839450828525,0.254024328964004,0.259960397537223,-1.01980523620641,-0.598781087870632,-0.299416848638298,-1.03946624495267,-0.237041027845005,0.180320083642896,0.236798987547146,0.0263501611522265,0.379918209619461,0.937327774998801,0.484677169794984,-0.36852928153146,-1.5680273513481,0.714473835627638,0.691749811119503,0.62874509013992,-0.672710666441107,0.184823198313345,0.502315675165615,0.612559330260567,0.0254850036738126,0.05333608105849,-3.25465488011904,0.951797459983316,-0.454180435120675,-0.0432182779369799,0.68783511936446,-0.254442893336091,0.594796415212931,-0.384109579511107,0.539229942981688,0.00591653068530785,-0.216647093618006,0.0758931733453383,0.354910665250226,0.417746361050546,0.403122821039846,0.570526076003722,-3.55058534691018,-3.52479952366045,-3.38201003608011,-0.583638927839101,0.773528243517914,-0.325146145958828,0.247831819057436,-0.997027724333937,-1.109332403672,0.767007239612611,0.256401264355001,0.367153131142527,0.393404032469523,0.437187828841775,-0.213449889364828,-0.41095868740508,-0.407409387183613,0.321663892260418,0.174531518387584,0.219133553912722,0.0498983628184893,-0.733746257236111,-0.516754738348024,0.135860735109447,-0.0231327502255276,-0.794295779963593,0.237510668590599,0.000821813417444072,-0.306819553591801,0.483404981924268,-0.0814232459488405,1.17455931705544,0.0490402923530534,0.554043631837219,-0.393406998422413,-0.43161082013645,0.847251181853769,0.373896952766029,0.323546015917386,-0.155696703311606,-0.601788723864223,-0.758440764080917,-0.603356787977358,-1.2733938809184,-0.0662489395281231,1.00656760135886,0.795080707525177,0.246724577145263,0.137974758442912,-0.0124050394337488,0.718936466367751,-0.334652583748581,0.589479421842138,0.288568433020639,-0.282338196150116,0.0337536810057987],
[0.0486132967993419,0.299988405681824,-0.00772496207083618,0.496362626338802,-0.0348545668323665,-0.176374971514167,0.244701368270588,0.199509700174799,0.490077201589829,-0.228698415188067,-0.0707351160412944,0.0928718019719411,-0.00761270840353054,-0.156755048865896,0.00168002082548651,0.100824003947522,0.402699293193657,-0.0776905940115244,-0.0213969477368474,0.00277425198915063,0.0121185668339024,0.446376049355708,0.0799926677943506,0.0670907207333093,-0.00529831867237713,-0.628382889863596,0.204538357248085,-0.167889806186291,-0.623058168744163,0.0456168858028058,-0.0038791654587476,-0.0217012998667023,-0.0243631524373354,0.0396923832985887,0.0706621090844037,0.158570321936971,0.164677334095787,0.285413580479002,0.28733490931335,-0.0252749878681553,-0.106529903227153,0.312019909010027,0.138671725506435,-0.0993983811166293,0.00660955151282761,-0.0513162132529009,-0.211118173094505,-0.00125486856355803,0.13173821160293,-0.0310582990259276,0.120712264698765,-0.0134585163058807,0.0449861813685765,0.157004953829414,-0.0188129610726058,0.057144685124296,-0.0221884990247885,0.605713098393036,-0.418542723636088,-0.103091447038061,-0.0172230083892733,0.204551457091716,0.235316052598387,-0.150007529872245,0.173856316432791,0.095849774239965,-0.24949463842412,-0.326061803616703,-0.37858677373855,0.544405000418218,-0.85383266649743,-0.085905438231987,-0.230193130004312,0.188930186122825,-0.195144494591845,0.0174860402380374,-0.0091752937449914,-0.0295872989530753,0.053131001231173,-0.16590590196583,0.0171871850277751,-0.170830630564338,-0.225921622784533,0.203998150108845,-0.51136834339287,-0.138661862344365,-0.501843964158986,0.00763353277877999,-0.451682033615982,-0.0647582916804103,-0.257119633435285,0.0572403482006316,-0.261219582591702,-0.0814727033858796,-0.136906887271648,-0.159236189526913,-0.347012184262854,-0.439899932493525,0.236573552021845,0.146789190124185,-0.240643606929841,-0.0350431046432151,-0.270388380319242,-0.254390435242425,-0.0863426173336239,0.0763527859813842,-0.120945534339244,0.0875608047654831,0.14337644691826,-0.158414129531961,-0.64328342220543,-0.0324767748507343,0.0183804225223936,0.10448666406851,-1.21912138376561,-0.0153099431759008,-0.0932984262414562,-0.0733414057217465,-0.174414700598116,0.388939548514394,-0.0890403393297436,0.0327632808158348,-0.549865842572306,-0.115916144682492,-0.350334106758073,-0.136239738165273,0.0876477254821573,-0.0424178379054326,-0.392418763026693,0.101367565973176,-0.0640734220568469,0.49203254722937,-0.130598557231891,-0.0506376382050563,0.100176950845237,0.0706800210490092,-0.0801363057291342,0.237472631204953,-0.195616062059095,-0.0577862287897583,0.244439345029184,0.135970029250916,0.152515663856655,0.0945522963779804,0.00751873594266081,0.0686526241358407,0.0132321960662385,0.0353829929665252,-0.070808878608791,-0.036541981992888,-0.114442783301454,0.0165518860956221,0.0462909921026262,-0.00584269501257911,-0.29513416683074,-0.207130696439383,-0.0869519944126202,0.0418443200032082,-0.229620092070379,-0.0214506496429351,-0.175144362657059,0.211350113191512,-0.10894560054151,-0.0789741882855537,-0.0209590475707882,-0.0675640384815155,-0.0873247725196313,0.235978293363354,-0.211853614034832,-0.0486371737969439,-0.242750257460289,0.060228236506038,-0.0128690405095751,-0.166102988091289,-0.0548665818923728,0.0384444829428912,-0.120156410521786,0.0542116440281878,-0.0640653338931632,-0.0116665994934124,0.188166697352144,-0.0717550639996056,-0.127725198903914,-0.105316639612559,0.0470112098458531,0.0206318859307306,0.0633709843499891,0.108346863318026,-0.158231939582294,0.000211963540785908,0.116522485777974,0.0596002183994007,0.108717950832726,-0.0566082853980979,0.0109029573752143,0.135823648721685,-0.0413419143516553,-1.59680162106281,-1.08347471826817,-0.678610138252294,-0.389173631026424,0.358047127556034,-0.57314594294326,0.375493223364186,0.281701884858655,0.205046713619912,-1.03020759709729,0.213167862630324,0.324758689906153,-0.183884349622194,0.44414671047072,-0.219780691659028,0.208500113866958,0.368912406813363,-0.389557429634476,-0.0117156938699637,0.0201458096448619,-5.47643082080134,0.0458292677677846,0.924911607173299,-0.313367025617452,0.167503130315546,-0.170230569045325,-0.0924720104851356,-0.367029806910537,0.0920849715275145,-0.420797033595911,-0.0612398423914578,0.0488764721518792,-0.0118127588879323,-0.042585587227696,-0.126386774266565,-0.223009281480079,-0.0967883023505098,0.161696849536751,0.0355185184768616,-1.45080315855336,-0.159044684964602,0.0161180852263547,-0.0801814588155229,-0.259940888775479,-0.136779497592011,-0.399957020462964,0.0686755538774504,0.162801030541236,-0.158282137357126,0.071236626455369,-0.379942518099331,-0.00103048007802414,0.194854044099962,-0.269402664011882,-0.0456924481932692,0.0578005136895117,-0.00540769629588686,-0.0365863827377695,0.0271457418864615,0.241860631728739,0.0843025090835625,-0.0759224195961808,0.0821337904421786,0.0702515876514848,0.109548810568376,-0.305504478790675,-0.060790126595408,0.188041461077436,-0.180080499438903,0.263743326925125,-0.191303922661,0.100455981995246,-0.192103700321224,-0.964244115674321,-0.637724852532438,-1.35509755525482,-0.529950543920511,-0.869182772238506,-0.942466058571328,-0.739195419062988,0.798815069831784,0.455108213971868,-0.200412956422589,0.0126679558287875,0.55847954945258,0.394110440030814,0.19875551539303,-0.102885240546817,-0.224947141594863,-0.104662837677782,0.210629051363792,0.254690180459578],
[0.0658845589899294,-0.24344743688211,0.0924162625947558,-0.217559688933858,-0.0518288206034741,-0.0526770148481765,-0.135635547091273,-0.0672977925063998,-0.0642550321964394,-0.0620321650826537,-0.023023737921725,-0.414621627911019,-0.117215530156447,-0.116125383757559,0.0091452906158689,-0.0218831061858319,-0.176921993455019,-0.186290056184609,0.149850044871562,0.0876704701513367,0.00653795530179436,0.586299255753089,-0.0905047477744904,-0.180235490192063,-0.172718563129831,-0.444641843990603,-0.0588221181189393,-0.157420657863152,-0.225458704366756,0.0144030286191889,-0.0238848760224649,-0.0943853820169636,-0.0708695167783935,0.104624209774054,-0.0965587690716397,-0.0610653253080238,-0.121561074722191,0.0061449306528463,-0.24949621398975,0.0605992439922984,0.368653135835848,0.0159751990396663,0.09359106115858,-0.140709367100129,0.318877207990042,0.452320958342802,-0.0531297196342564,0.75279974739805,0.308292603222924,0.0202466324935064,0.322363546991038,-0.124712002196238,0.00297887588377901,0.00534337108902551,0.0676896931438289,0.0876722191825992,-0.190007725877026,0.15484473222093,0.0357920942610764,-0.136713047264902,-0.0966786452070711,-0.129569681085897,0.428438759339976,-0.171507761034415,0.133753404346391,0.170666040424968,-0.655376185174147,0.281614731362165,0.170214074084889,0.0739167396092808,-0.166653676483969,-0.173511506013627,-0.103222235646277,-0.3282128007963,-0.227310221763793,-0.0431064622028426,-0.0651013253584786,-0.144940092380528,0.0250309897473709,0.00357611508205513,0.0108333407710015,-0.291340732975975,-0.0262087917766249,0.0321704638933742,-0.816487776477676,0.183616069042077,-0.75907157874687,0.0550233073908582,-0.731028275227708,-0.0438120637786068,-0.530406883139716,0.036481524967825,-0.156976793931501,-0.27347503335643,-0.0480452074044781,-0.481228309758092,-0.102366104522915,-0.206075855155373,-0.111524500277416,-0.312482100617744,-0.0212862539209742,0.0343310623600035,-0.336221410983401,-0.07102873207201,-0.0482125730349589,-0.0563590059051756,-0.0803806014531701,0.0769829805663285,-0.190961899068059,-0.330299136065183,-0.402359901439426,-0.0506028444948236,-0.212403069458143,-0.156347474724213,-0.969042756284185,-0.0485808228439814,-0.0468559357480595,-0.237354088319359,-0.594861831412307,0.23689910559841,-0.220952663068466,-0.0281223515846199,-0.188937429060168,-0.203345134110059,-0.00383665711242912,-0.0386459166765467,-0.300333330109402,-0.189588082412268,-0.638888055537944,0.0140359971139774,0.0168373450031161,0.316240943248628,0.114501427633827,-0.0732691051694695,0.098548122286157,0.151044171580471,0.0252004806980631,-0.22419472401147,-0.0308896992043893,-0.135132489747863,-0.0699666365305858,-0.151626172526848,-0.0448238106552332,-0.0616508995777663,-0.0814584119802141,-0.11968497797717,-0.040115556096258,0.0636160917337902,-0.127992919869789,0.132933916850903,-0.172428212466266,0.0993707469942667,0.0158767612776263,0.100869108937821,-0.142314186102384,0.307593585898599,0.0445558038702735,0.169021402441487,-0.0646928155782385,0.125323382827593,-0.0645301695019007,0.0739038816243936,-0.0125532015693382,-0.0736359999508886,0.0685020717522686,-0.0571302694430276,-0.0875090480782948,0.0390472815843209,-0.0849782885689664,-0.0780718719279817,-0.1508667530567,-0.00418618175773094,-0.00364884186368055,-0.0634676389321712,-0.127100309119573,-0.00308414711559061,-0.0142803911152976,0.0437293012660328,0.023346205456673,0.0406172814928851,0.189046413687363,-0.0166172953192374,-0.161622827159356,-0.227826158030901,0.00847176413732341,-0.0245990321654917,-0.19839546219739,0.0988739528641056,0.12010551333756,-0.0854188371959412,0.224937528010485,-0.161864611234851,0.120673798719303,0.0296253576926858,0.0957823056101945,0.0348172795773161,-0.0913290636471192,-1.69448959479995,-1.39035156109684,-0.983162017407537,-0.31102523321427,0.641348829028154,-0.62702295543181,0.619957984771078,0.409128318916995,0.537388085391066,-1.06109353599436,0.321777878232182,0.471592933019577,0.19303302941287,0.77315629255741,0.240876636619304,0.721277368051792,-0.156050704842706,-0.241789034221585,-0.146033066655909,0.0663333226212228,-4.36571666293572,0.0298997019840551,-1.24747396788693,-0.295937540129838,0.348966443929913,-0.411024809545037,0.0071725187410666,-0.345999733695626,0.296082701938048,0.0880849689086623,0.0209177930987392,0.0838696769675145,0.0328904487538617,0.197554841705375,0.0125834901270491,0.109273294963939,-0.274416809880337,-0.231529843115144,-0.41175425542852,0.688492160160579,-0.0187330626609733,0.11107836666325,-0.0755441272915004,0.0484064602269789,-0.221607138258508,-0.0329523731620942,-0.0608718842792065,0.176742425092314,-0.285824211839954,-0.0107186931954343,0.400589527241754,0.00850209362523835,-0.0598450587805254,0.0150931807394332,-0.143326761516786,0.143928309685641,0.0238490269649408,-0.0963117752073634,0.0741233061533974,-0.136242428792107,-0.0045835975259752,-0.0372103147084255,0.14326961674446,0.0131946768704956,-0.0881410893197444,-0.144988978702629,-0.0957003741041998,0.0197098969336389,-0.416583946685334,-0.0727929208957103,-0.622193079232757,-0.172673832231642,-0.76381124182098,0.506673204304508,0.524555966886545,0.203985749620296,0.00693386534748044,-0.246858345628928,0.248163501170304,0.460388447778783,0.345873430771796,-0.0915898892945161,0.333279937495833,0.00536240417952241,-0.234664006610011,-0.046456997101717,0.121908336594591,0.350993631923907,0.00721943123571631,0.109766815695159,0.018398677872635,0.192337420883013],
[0.0325975912713548,-0.0576334802838891,-0.0866595987181921,0.0393936573748629,-0.183755714467193,0.0225063756499606,-0.08678034315139,0.0135488252806855,-0.0112350651922208,0.348470002085179,-0.0210419727777468,-0.232808225604885,-0.0940047062122063,0.0926974925667454,0.0961618386392376,0.077970397687374,-0.154929485020697,-0.127945811598828,0.185334109023358,-0.0986436422906669,-0.138826866351549,0.453660333466756,0.131496106536425,-0.127299295938754,-0.0609532631918146,-0.289737228784002,0.0530899612300309,-0.027296013009251,-0.482026539643604,0.458617979280149,0.00511142997888045,-0.0270036068157393,-0.0781371153139747,-0.0521355109860784,0.176470982047541,-0.0416854692501011,-0.0742490832129653,-0.0122167910323104,0.283097686052893,0.146170219182559,-0.0323766194283938,0.0453809053793261,-0.0510696044038291,-0.195983813423614,0.257392631008408,-0.150015061866317,-0.164788884211646,0.201217814027449,0.302743565257921,-0.117134247855071,-0.0724830977684411,0.0863066938874664,-0.13182168702108,-0.0303324998294717,-0.049308972773517,-0.00290064223923555,0.0742208482885955,-0.156381733541741,0.0756382840458239,0.130731879672765,-0.0483710934593557,-0.100252711708915,-0.178443594364951,0.180463983607923,-0.0601173970544919,0.0171526286340927,0.111648090336404,-0.0284235539339176,0.281854991534326,0.209715343368395,-0.29446962894127,-0.176455138703676,-0.0926868493919619,0.0823821763767799,0.112030235820241,0.178170191135674,0.0428677643605256,-0.00772458478555227,0.395999916666025,0.0900218060433764,0.0807104740923667,-0.00336307240162906,-0.0448717615554483,0.131969146455855,-0.293293708205342,0.00879187663705404,-0.383260849008744,0.0847690127791139,-0.365096162664949,-0.0322568630476893,-0.288511956485058,0.0764753870872888,-0.012648530100149,0.0157026935143189,-0.0127524951996499,-0.15962022443495,0.028221877595314,0.103662807297363,0.213152749512887,-0.0422268349854065,0.256566776081171,0.137309879901848,-0.108928545785999,-0.157276662303189,0.0542836681156156,0.179678231794256,0.00708450169240033,0.229606116580464,0.280880456352117,0.0546801655595684,-0.166144703815895,0.0235124658220029,0.0379310392464777,-0.0985874247150268,-0.79975785314267,0.105678039254326,-0.0784209834155714,-0.0426726686406697,-0.262655470228222,-0.0685459352200458,-0.171790361974172,-0.00641603257865502,-0.144078196528933,-0.196586222700885,-0.0757741298280188,-0.151638223963499,-0.211982448459944,0.00298610808008426,0.185549008108746,0.0708669372500271,0.0221125242425618,0.0459303403678829,0.003646816683345,-0.00531146150923059,-0.0700460186850662,0.156712507292753,0.0753155583741123,0.059376913391626,-0.172442363642021,-0.0658473131988208,-0.0505314258915918,-0.0155502431112156,-0.0401986905566986,0.045062010643557,-0.0458335165179894,0.0223420201163239,0.0705627000934584,0.0306821604702684,-0.0425683023461484,-0.148648879369442,0.297007270315612,-0.00644134275586778,0.0574941072991545,-0.184828646816661,-0.0128248801112457,-0.0874161561993432,0.0855136830247222,-0.0796208899939208,-0.0534363931812993,0.0450902027286153,-0.0108069941622303,0.0181022241243956,-0.00273355144673268,-0.0626479748307865,0.0600034124975787,0.0478916548277475,0.0105909447371859,0.0105040303882717,0.00560454359062464,0.00769971447873892,-0.0571289778773813,-0.0436142631200648,0.0248117172593837,-0.069773069668974,0.225428076026586,0.0188433303954228,-0.0232763248849964,-0.225183584387381,-0.0212745170187425,-0.00232001198591759,-0.0864379386494374,0.0135446644541047,-0.0685045877854448,0.190979455081608,0.00950833156390761,0.0396702657792098,0.205751501904316,-0.0289345593271416,0.00893614730901493,0.197584782442564,0.139657397614599,0.269276307718344,0.0954333221810318,-0.0586027211067498,-0.141750991531485,-0.208548289026793,-0.331936242756174,-0.725753759913991,-0.228930746364426,-0.113282643751064,-0.161503512494374,0.233412932476646,-0.44990522138656,0.219172124575668,0.213662425873514,0.227797331702009,-0.524003844227822,-0.0252487248621285,0.350036499570621,-0.314478945577226,-0.0408907101419356,0.0140858213037295,0.326402847823294,-0.0200111940038806,-0.053532845027432,-0.0265296512882178,0.0122190404857299,-0.583851947481695,-0.609241896523346,-0.0497896019077626,0.0804462870151063,0.153347089546715,-0.0743961875583915,0.307652812191232,-0.1460965975505,0.426929577176983,0.0363375235496869,-0.110180917898916,-0.198082694105559,-0.0791621536233465,-0.100379594106408,-0.187696495145247,-0.272638675143928,-0.205571131061169,-0.2136058205229,0.119480407426553,-1.28507645208809,-0.106868589516798,-0.296071474052335,-0.144292436704137,-0.146812705181684,-0.0221787762465075,-0.0970577945852155,-0.250732943255335,0.0896371200337414,-0.305170378462998,0.00806490024668743,0.178999019236862,0.147983137569517,-0.0220896804508477,-0.198054270025406,-0.108085514471219,-0.118055219735636,-0.0288980890095247,-0.261180001565355,0.0134894079975411,-0.0103940760340962,-0.0105938648480617,-0.0396774593066025,0.00517714132055748,-0.003472071674528,-0.151680026150077,-0.536434531808419,-0.0767185336474033,-0.0111109254430697,0.167798789121092,0.111619627848941,-0.105311620159242,0.134764102524372,-0.161751926234575,-0.467874953364828,0.00113792332477346,-0.535829819199155,-0.0547256473843014,-0.520120721385557,-0.233620516232213,-0.330435954728978,-0.0480425864099845,-0.0395576042195232,-0.122594233926786,-0.560436706326087,-0.181425138603315,-0.10234159775319,0.0639786908020245,0.0959358402733819,0.0163692304799258,0.0806268509009231,0.0894204452087134,0.155663771655085],
[0.0353766216374101,0.0437461392625969,-0.184126952615318,-0.0371394002781222,0.0194354990741748,1.05265778680265,-0.00879974739160564,-0.133944737235566,0.280030573020209,-0.29622816903609,0.202560268457703,0.0856290701688088,-0.0213910830669405,-0.193339136424821,0.173895723491451,0.0540439001028399,0.406652931301895,-0.0529626930825269,0.519195388228544,0.161269006841417,0.275238430619203,0.789426239254816,0.00276961039218173,-0.0346949841070839,-0.0487111051388268,-0.323909787914754,0.0322160328280922,-0.232016639283067,-0.251257918080962,0.180323728705586,0.0564609163487754,-0.0942892594210424,-0.0700695890593015,-0.0531779232059467,-0.148165949774088,0.181223360695401,-0.121989391658905,-0.117229162297305,-0.354803685187535,-0.404359703096418,0.0250556727740016,0.0551398513593253,0.0775011294145449,0.203004594036372,1.19391872034853,0.397994640524034,0.0164688089482348,-0.138621981560101,-0.694856191660664,-0.0691186456935016,0.218786973187022,-0.257871202670702,0.000516525816105786,0.135618044047279,0.00657803689558527,0.283346382814335,-0.299673168608004,-0.229012102907925,0.20910153216633,-0.281589747335592,-0.0298640315177519,-0.110608201917716,-0.00112221005760011,-0.0780485455637595,0.026135567440029,0.11632407842587,-0.0542433749779441,0.50209118776507,0.489602824807061,0.445023903781364,0.154328137277534,-0.14692506000305,0.0654687216750638,-0.357103051267146,-0.36643843343312,0.0141411331469601,0.236021089092991,-0.204314257717775,0.178508735140016,0.361543138869179,0.335594374787534,-0.259687289515588,0.0205326561848656,0.175652263758282,-0.16971379123936,-0.0640364657339106,0.0832671002068452,-0.0454918584360039,-0.549507311887542,0.167731401251405,0.355184858440203,0.0929779800216463,-0.294630245781959,-0.260155931210333,-0.109032281612945,-0.426498581569688,-0.53203291866749,-0.124132467719762,-0.0711992217280767,-0.524737653778035,0.192415219228118,-0.277636996966837,-0.234732546552494,0.0226211600367153,0.221791183777647,0.0684141601155294,0.214743861807094,0.134466146563293,-0.342281136305239,-0.176818093211858,-0.854871863015763,0.3858251636276,-0.243777469925539,-0.351655143882858,-1.33827739828755,0.378611637507571,0.260404879302496,-0.36846131389162,-0.645420204467156,-0.151243174150327,0.048319491267384,0.0734933427611895,-0.24396565407211,0.25834276948281,0.138100013779982,-0.0537580121811502,-0.352367966114724,-0.399597521910668,-0.101918049879159,0.0440918584154529,-0.246170122610874,-0.172604598005589,-0.417541995296296,-0.160449310393262,0.36028792614951,0.248622415128622,-0.071883710249954,0.0365091737848449,0.230528312196864,-0.204235370305791,-0.209517880425952,-0.337241139441065,-0.0966052248118211,0.135267319301754,-0.42315451665248,-0.704884680377561,-0.0151684880882636,0.0775874841206678,-0.031163782967878,-0.0791318337738376,0.163215045006215,-0.212289858470784,-0.109271999871964,-0.301172316213519,-0.114951940516541,0.263093898499713,0.0819353309112217,0.451887214730176,-0.0565893123377527,0.26608036754799,-0.0647631797561397,0.0233622711454079,-0.00442936777184393,-0.162787074003756,0.061924820143567,0.0746452064066273,-0.0750441107342477,0.0324585567769453,-0.166112754873317,-0.0486626534202036,-0.149719282915722,-0.147449473192486,0.0582416833397256,-0.165890876049177,-0.0529810456084887,0.0292651681194851,-0.0981627406613742,-0.0462428793551276,-0.0204339883692797,0.0562436222129785,0.121659088467728,-0.0178248820434057,-0.195138239329634,0.196112076688708,0.0249554076201089,-0.0689302419410083,-0.0980199653509107,0.0526877042310019,0.390726992582096,-0.15192120185612,0.00335652540730592,-0.217108354005131,-0.0108900121059449,0.102082426163144,0.161864919453979,-0.0219238630006088,-0.0725762808149656,-2.11491109390217,-1.14793857677932,-0.844446807285774,-0.242144106742154,1.12130591407893,-0.879219008119946,0.911726701651715,0.49006300617096,0.881504156754521,-1.41750316451946,-0.804034604234121,0.480400577382551,0.539840177282455,1.62122229622265,0.884503158112988,1.58517425039605,1.00551861580582,-0.629404338339318,-0.253431586692682,0.136658281921191,-8.3496608490553,-1.65008118148412,0.3337261924276,0.170867527181757,0.0221616492722705,0.226185783168822,-0.237479024086005,0.16537705621395,-0.0797275837852848,0.0496048183868199,-0.0689576545108259,0.159082543400336,0.0927700219167371,0.556286833425786,0.222136807232917,0.628473514012369,-0.0675420645463192,0.0214562460787897,-0.542897428738102,2.14906467231597,0.255612786920994,-0.107422968148049,0.113917535480281,-0.036167212682264,0.0211983896399443,0.299424262378658,-0.142552457761709,0.0183138428580361,0.143363435207498,-0.0527256017923163,0.494737887645239,-0.129129354103304,0.0397760558081727,-0.356179280036763,-0.0933596307148414,0.0388608724660464,0.0942515814122559,-0.177912175490042,0.0298183181342885,0.0590705160739483,-0.0863922739671128,0.00768062226860734,-0.0791883599297025,0.0919051394206276,0.229152620313018,0.172270783719426,-0.370472411097477,0.187585663959122,-0.154803564087605,0.0868477798408604,-0.938643694630224,-0.968298651601459,-1.66460628761145,1.32663046729845,1.67775161719375,0.341655750648529,-0.307488681575269,-0.145691614177943,0.735566521327521,0.751440867420559,-0.167074598965149,-0.500057245975385,0.0985927143149413,-0.520723336991749,-0.120229726939278,0.0900311981802515,-0.222711179066543,0.373053679646443,0.055033078149233,0.133787846144667,-0.0173317592466561,0.36226058439891],
[0.0208174586418275,0.281152522900954,0.106493611896898,0.570971802063988,-0.25771847280738,0.0542810280217708,0.0801553521955836,0.0460083798001978,0.592597507976991,0.28343638328998,0.16841391663758,0.186003091156301,-0.0557031936745366,-0.0657222943887437,-0.223898520663428,0.112922191836395,0.0338059347328053,-0.298891163072887,0.194108759327688,-0.0869522214287838,0.141057827938719,0.815043463182711,-0.290279230638887,-0.01015673826945,-0.0432982643353352,-0.376069290707765,0.0701264523697223,-0.905804075965835,-0.254230229356778,0.181947689468833,-0.0408777974460077,-0.0719819157622965,-0.269828884480125,0.44522813725676,0.198300501391482,-0.173915629447346,0.147345139561345,-0.155080072281436,0.253991140111239,-0.531681765900926,-0.501055860589661,0.170034197424381,-0.122716134138418,-0.428706585257027,0.627254674187539,-0.0934986371204519,0.0746215164073749,-0.429299619461413,0.239094116128032,-0.0911865548186777,0.302278196690177,0.237223445616702,0.0603268351145906,-0.231765194629377,0.0819080945163049,0.176432179053196,-0.400269167164397,0.104093354548453,-0.0111785908719855,-0.279315890256979,-0.0177984305210213,0.244123835468625,0.348054398302496,0.0342713915264324,0.351259617861198,0.239273385267054,0.0369778268330484,-0.273097579513636,-0.635350901884749,0.565421518709886,-0.96296534058461,-0.135925247106813,-0.765576530885109,-0.130028616899046,-0.0247678776862925,0.393960529770612,-0.0307236995453014,-0.242323968534701,-0.0822192385243355,-0.008766980796453,0.028500367310385,-0.365694605137401,-0.281216792823699,-0.188291405971568,-0.67503345637944,0.113026345071285,-0.616263749810928,0.000122966074984855,-0.662539459416013,0.0708222711237508,-0.598817280834883,0.164626445829665,-0.225376253580149,-0.250128388972135,-0.0670113929646872,-0.389403577469512,-0.438387919228735,-0.251575232042633,0.430558385983567,-0.0733835779818943,-0.177612296894969,-0.0515446783656002,-0.384251557103747,-0.0946099669536614,-0.199968647558672,-0.0585398820437067,-0.226626213876428,0.0279747095591053,0.00723569146403434,0.0101606334227827,-0.914430954176787,0.119074120476343,0.429301762363361,-0.335312004087233,-2.48309858796619,0.193540081973625,-0.115379626757739,-0.361473718803599,0.00500916846723786,0.0507883074555191,-0.270658988392089,-0.178006121493922,-1.04381708970876,-0.201120595929285,-0.32037864580089,-0.0084341690687791,0.671112123530899,-0.26906108385062,0.373961267877416,0.019970124702433,-0.0256870603920758,-0.209968160755682,-0.00203909410311243,0.0619790854649464,0.235262112545697,0.110708276564444,-0.180648281300997,-0.0880086152021193,-0.142881885836673,-0.19287952774379,-0.0436836781298629,-0.0796412981989848,0.0128642140595319,0.105526009498435,0.203446864458095,0.0322383714723765,-0.0412027056878376,0.126134320571746,0.183420335458022,0.0216223443158725,-0.13744653084426,0.0244560000382165,-0.00104389990401996,0.142903297702754,-0.142380701467667,-0.161417729208825,0.0333439606095242,0.14789903064446,-0.170622021292487,-0.0719762257388003,0.0492691642160023,0.260703530003475,-0.0179387584320326,-0.0585735246393976,0.104389354660597,0.325625665366194,-0.0149524717226057,-0.126706295939464,-0.544049934063327,-0.0671936189286439,-0.242841326231412,0.147596600088161,0.121171247218257,-0.211394744782274,-0.292798310213465,-0.0437707482720157,-0.0888314799686397,0.146941232826217,0.0323243186896366,-0.00631695979237555,0.195396519119859,0.0165975935497991,-0.0689553464855917,0.128637960068763,0.0694188285639358,0.0222683256046853,-0.211061032856754,-0.116913725867086,0.228158497942483,-0.0363863862327784,-0.0788685368109135,-0.196391702180442,-0.182529421612686,0.411310721328263,0.120521505487641,0.634229482345913,0.208140963410425,-2.27717622347905,-1.32074517944982,-1.5924945384899,-0.396666004944508,0.585550916966037,-0.740399137961464,0.855031926175212,0.1794355940772,0.44787572915063,-1.5852839072653,-0.0890615690361109,0.811501759820636,-1.078480237358,-0.741500092837681,0.297119102183846,0.806781736167131,-1.12434277628498,-0.591982796352768,-0.13238991516459,0.0415712006304328,-7.21836924059012,1.47469468833329,-0.708028100926974,-0.435210811905862,0.334411651470788,-0.340761883813327,0.73376701315849,-0.842861652722787,0.79445419787417,-0.278322954155747,0.255607988137226,0.254042539903093,0.504916973292924,0.557347491374155,0.256488562501079,0.612025853747459,-0.716525399270668,-0.816781892446324,-0.557762317120237,-2.231915231579,0.590092012121128,0.495375767524747,0.370919492362505,0.55597808443914,0.830836967643255,0.252134727442787,0.608952751995062,-0.395021985240505,0.489233260190631,0.153883703832219,-0.0814186763516354,-0.1046131092793,-0.334691124729169,0.487615622240824,0.571269836671643,0.383166265430893,-0.00548884982349803,0.406705686597551,0.151051610538337,-0.296114035188928,0.18521846990074,0.050261678451589,0.0511355512875041,0.00913547229607962,0.0235347791897392,0.438942105790819,-0.543221847227673,0.521410198469454,0.142121870073104,0.125628144550943,-0.355557276471547,0.768613558647763,1.09589761509158,-0.53894186114618,-1.00395719433861,-0.141500975723882,1.01053497677022,-0.536668022338976,-0.336046651212861,0.302989982813183,0.728358823374707,0.416489631628318,0.777548656221658,1.08795381540076,0.141430217422258,0.249579565923374,0.514932556086684,0.427475453964804,0.194557377692254,-0.395230569848341,0.214762450610609,0.478229779196169],
[0.0366399734901772,-0.326478139219729,-0.409400538086694,0.310450732202,-0.4060751088493,0.122064250791785,0.334485492440448,-0.846316535340774,0.242493649559365,-0.0798951180657027,0.0133783061065264,-0.808979865354554,-0.0513644039194988,-0.229869476094009,-0.0722886814365861,0.329855128535248,0.44906899026734,0.134649806937285,-0.0488648785542275,0.232591042850485,0.338378584703043,0.606070272639377,-0.38214113006655,-0.22837257744516,-0.375873915091677,0.301144834662808,-0.208051764914228,-0.163505232264924,-0.5160284991515,0.193601092732458,-0.00611956345086794,-0.236523696324957,-0.0570629683632611,0.0739994121709842,-0.211959726151512,0.0247045265883259,-0.307651775561185,0.279136625310543,-0.833572280962041,-0.458353640997445,-0.34235193595134,-0.018809374928677,0.0871811385474229,-0.168247374561673,0.345606286010999,0.307585229484417,0.0513637859091436,0.567798206290584,-0.0959493578156547,0.00222337516801372,0.259849855901751,-0.616123641737256,0.333021592064321,-0.162230113413905,-0.430840490333399,0.137188904820661,-0.385850648785822,0.433476175978043,-0.216708049815372,-0.359873661152805,-0.270282006468251,-0.123261556125512,0.0122451844313314,-0.270377775495236,-0.0112414272108491,-0.171134872659154,-0.848368614262663,-0.178230900931171,-0.0950146085892575,-0.242174721194688,-0.0168478874392647,-0.204753604682613,-0.136065196389264,-0.589141061089833,-0.539631786596987,-0.415305842933222,-0.164519347421658,-0.13719037190524,-0.308939936184657,0.244520810277963,-0.226421124216874,-0.177290121495323,-0.245781295702219,0.072701146815487,0.012648547551332,-0.429607079464968,0.154132878708974,-0.804846569117773,-0.962647924037557,-0.0348877351046449,0.128194556530243,0.270994621070896,-0.364124538983864,-0.328809779097121,-0.24524804467195,-0.23886149363407,-0.414521868745835,-0.896280723225352,-0.082086922011012,-0.487151463193292,-0.246534458991289,-0.00402283514611028,-0.664130694087005,-0.663395667529099,-0.297301700494444,-0.0949513106015697,-0.281980709611253,0.0172114015944415,-0.237716479742904,0.206003824292797,-0.984575305938995,0.0224378795858178,-0.60393902039281,-0.271864677545702,-1.07006058699756,0.0770971704750568,-0.169036084432736,-0.0625283147966862,-0.834307594912008,-0.274072061784098,-0.090424337819708,0.0551025019715385,-0.403859130559097,0.180066188563073,-0.089601746998041,-0.199284047665289,-0.381768240979889,-0.285328907795732,-0.4092575238034,0.406284078754507,-0.211145211935994,-0.302903991172724,-0.0147365153602727,-0.576142273262899,0.193301278979083,-0.25772857076403,-0.473913352601693,0.763662304437744,-1.19084263392979,-0.178529764608819,-0.106369204235746,-0.259003499582775,-0.134850157088634,0.208155368198913,0.381498780717502,-0.704823570202693,-0.0411677417378049,-0.0719578171529861,-1.01435402849224,-0.632579877775649,0.0610632514591345,0.0751435722979786,-0.248401784879169,-0.00621456184525336,-0.0836927942440024,-0.0754872615580076,0.00164365286926181,0.258324842379352,-0.120102997020007,-0.249794519549866,-0.0405817026298541,-0.137488942766155,-0.0560599614522505,-0.294345760556424,-0.0546809546934399,0.0175106973132165,-0.0246868524252525,0.12674273938029,-0.0457704817889736,-0.00141118682841852,-0.0846191770082583,0.215891041564618,0.075475500729292,-0.0575435583450756,0.748786794576847,0.04180757866679,-0.0622909843304847,-0.117794040850534,0.00363361537965589,0.0434007310732945,-0.266101531939282,-0.0274128620165377,-0.0993833584870724,0.865430805511971,0.017206573876317,-0.00667737832605242,-0.0554576650007374,2.42023122170728,-0.380704593015232,-0.290532285884376,1.4510862629836,-0.588979841537651,1.14305758207672,-0.573803239936601,-0.599278626264126,-0.712585360339978,-0.392125698129815,-0.829447487364281,-1.08640813224681,-0.562154807528754,-0.13243839629119,0.39123652953097,-0.251901970864499,0.37070849777157,0.55141128847947,0.311104141862154,-0.550106128894953,-1.44038705221677,0.299620474347154,0.657584468217724,0.959674345678527,-0.108085373378528,0.622925217010864,0.370995517143195,-1.34726716851343,-0.0966026357011888,-0.0052947813512097,-5.06119263023421,-0.378161849412798,0.184128572881602,-0.408433935590825,0.00924998449443016,0.146806540388192,-0.0835178763288978,-0.134054170248969,0.185162024024896,-0.404518690451501,-0.532993559051378,-0.230212860424909,-0.122210929130101,-0.595223646493117,-0.506089493675495,-0.352219058200363,1.93397545223621,2.01811780215358,2.50349695181913,1.44760646163507,-0.342347072524232,0.386220128943744,-0.230240649515783,-0.15941529439713,0.431905910766478,-0.159288127212664,-0.420904595765225,-0.0304009465325589,-0.657189715311761,0.280928215020289,-0.172414055484618,0.350727840002459,-0.527931543826261,-0.135248828076093,-1.1593531475586,0.417175726735157,0.0168486886180247,0.552848855045923,-0.338482277199924,-0.939480340364672,0.0783691452658174,-0.254569305187289,0.210984112134736,-0.0274065803516036,0.207563746637468,-0.991528484966616,0.206530206058373,0.596044603528439,-0.814237784853907,0.21764376214632,-0.340473879660959,-0.0820133494920125,-0.790465284677999,0.695886744366057,0.195461227596872,0.190205525815421,-0.763031081616935,-0.420365127196883,-0.390682571514429,-0.358242269847431,-0.0229320344459356,-0.190986822362874,-0.0751484226594934,-0.0367275414931442,-0.240679680923761,-0.123264358270836,-0.0922013327320584,0.15397646708902,-0.246327848747271,-0.146354743189487,-0.608578551663989,-0.40956696681256],
[-0.0103571651608148,0.798610341732903,0.176832456544236,-0.361865180256992,0.0561182876442424,-0.239461195170769,0.119768176000796,-0.382828854401877,-0.0304416455611877,0.298207543994848,0.0648258310881131,-0.94191189010525,-0.018321726930582,-0.185975393511634,0.120618707325156,0.261801776867293,-0.784304548619382,-0.0180030166891053,-0.231623223833657,-0.835063326777397,-0.625449300105732,0.336350056058327,-0.137322163474926,-0.634812875972285,-0.84343191891199,0.0939763794179174,-0.608342414017717,-0.274592042043507,-0.466008427741441,-0.0250878004626618,0.00163163646574499,-0.214923816927124,-0.047934180778586,-0.145385676606579,-0.0264502221108392,-0.048337551404116,-0.219097316178514,-0.0267110027574836,-0.735194095306006,-0.6698054524388,0.164332506645186,0.186608997088171,-0.51964770344499,-0.0346028429463106,0.239891754367078,-0.597632657713128,-0.268843219854005,-0.432869427405094,-0.601336071257422,-0.0417292058278846,0.0576785671806378,-0.303509448431338,-0.526839057109414,-0.223329320140337,-0.527174721545127,0.084789204558204,0.110845950404749,0.632817777277581,-0.293458965569999,-0.0674004058793928,-0.0799015834211114,-0.626018574815763,-0.558056171953645,-0.486766234685672,0.213608293320817,-0.836889769132627,-0.468033659796009,-0.505082387137485,-0.196317050012801,-0.565452777710703,-0.605784665402777,-0.0781182991895736,-0.69069359493508,-0.936977521895281,-0.4437676903189,-0.622062140559165,0.111266672502546,-0.264952961442237,0.0707101597799107,0.0685080059901174,0.14617860488822,-0.303092856788107,0.137298317530786,0.112754441509015,-0.60828878813192,0.132651260017572,-0.382563185587467,-0.412191417417324,-0.921126223722347,0.0296817477838783,-0.74511402767931,0.137294112058425,0.218507944935142,-0.237151586380329,0.329563692740894,-0.30909118243649,0.0402871744470531,-0.783416457710782,0.373764949266954,-0.688774634675296,-0.0758728380179322,0.0182963924329011,-0.250075700211896,0.0738529231777973,0.112920548710819,-0.30407339974674,0.107841373211028,-0.11216146067203,0.0883345762344702,-0.424946460706852,-0.61764725601709,0.05128484068615,0.018563238283042,-0.47354500776231,-1.20780566574243,0.177974805811763,0.0801082479370723,-0.238581271824406,-0.329046085068792,-0.495430235611577,-0.102974483616935,-0.130836219479829,-0.417019054708687,0.083343589988094,-0.581219879333198,0.0954094265392449,-0.156702380039131,-0.482446530768052,-0.188728198533294,1.16589722469917,-0.124113551268091,-0.208437984280836,0.159877847803396,-0.381309850848746,0.193487598740121,-0.225528395113603,-0.571947814365597,-0.660498692522687,-1.04842120193969,-0.183399372761197,0.0167323600098631,0.744229945083963,-0.0604759689268297,0.212549855296088,-0.0129921547713368,-0.342182426446467,0.00108808081592685,-0.189314593987509,0.371735645923784,0.146460989992555,-0.714301860367569,0.536298396520062,-0.193707180034639,-0.257552523325286,-0.286621799869378,-0.89719388697342,-0.124001529097899,0.0727271258928337,-0.163444261026095,0.0907633365377176,-0.0913044994315145,0.0285208790198457,-0.123672942320561,0.29789555527966,-0.146010476300221,-0.66945398346981,0.0318302542236812,1.12160585513847,0.24734123742859,0.0224050504712168,-0.0619567249870897,-0.082468923838036,0.000236901802358512,-0.012925981704692,0.524721945338581,0.0235196138106691,-0.0480788206978231,-0.776171007952412,-0.0469889039026556,-0.0238517587710093,0.488255013681441,0.017094613400388,-0.0563677517500107,-0.745458579813259,0.0192101609473324,-0.0378301746370894,-0.406718192719301,2.85190486985749,-0.21562731997865,-0.823142419219019,1.31152433711999,-0.906604452696362,1.23241617823946,-0.634765881268147,-0.722863582319979,-0.154139238099348,-0.0522450414716008,-0.669848630681494,-1.29658966517946,-0.919311739721761,-0.136110524553657,0.145059570879445,-0.169998195151986,0.205758061480028,-0.149897347063193,0.0826955754207975,-0.418401550458029,-1.16881186404981,0.237878778295588,-1.27302670797987,-0.973895908693016,-0.400788681163896,-0.112727014868994,-0.737229316070684,1.10096564922471,-0.0778412210639817,-0.0265038899776191,-2.54184808488256,0.819519143811733,0.344818863129302,-0.910873804909635,0.658212069406824,-0.82978753703253,0.350167839237111,-0.424302942802543,0.421726783605993,-0.844780176313989,-0.360047973332472,-0.303374434289862,-0.28937358240184,-0.434208769151937,-0.469935253346772,-0.159737588400531,1.27307829736516,1.12429536965163,3.12245053507623,-0.0641033197840193,-0.274300760717786,0.589544092386866,-0.515026429443445,-0.467386066228757,-0.647569264853475,-0.102556118744338,0.750516782022511,-0.0618869104626074,0.361138421038337,0.0205567744950799,0.665785987813829,-0.723845241146902,-0.145159759766662,-0.0733159190058006,-0.155996957918349,0.729299663596225,0.00992788232604043,-0.239385619166487,-0.0463378132385442,-0.18784139360995,-0.0535573939572316,-0.488550909368712,0.0543669947303772,-0.00821176653942862,-0.242745595744239,1.12891764534394,0.0334617288902022,0.15595818855099,0.537730766492138,0.202079872378973,-0.316278007079296,0.148611455662029,-0.198772681507403,0.394253529538362,-0.341044990450103,0.267759160559228,-0.539696975645136,-0.0622741343714984,-0.345707117661257,0.242867340360227,-0.115376065783023,-0.388751681922036,-0.231258959555165,0.105950147524827,-1.06898350353183,-0.971761263662014,0.477409683202612,0.613092970838411,-0.0388275743241275,0.492313039722311,-0.666011767372372,-0.606605855063286],
[0.0263543938316998,0.0933200949191528,0.170630419932292,0.943156722240806,-0.516004587360666,0.623835074109411,0.216037373952715,-0.191400048782613,0.712663896342801,-0.664014115258997,0.705785900157483,-0.912202637890806,-0.2932996787423,0.159299158347081,-0.269460072065112,0.599614475838728,-0.402882749877011,-0.784278196589694,0.464576373520098,0.145148956241089,-0.0449234671175843,1.35698422940462,0.0332772541170778,-0.22140491475339,-0.169392678469233,0.0730956692442361,0.348325533438813,0.393640636524175,-0.0647169195213302,-0.610647055702374,-0.0470539889606102,0.254774575521598,5.63543533362263E-05,0.423682305291518,0.148513614288872,-0.0186704778340156,-0.225567713054402,0.157343412963636,-0.956973210236569,-0.742779381395771,-0.921241616152937,0.342080835300786,-0.183409653257218,-1.20605568908246,0.499801243122258,-0.276061278100994,-0.241241648185107,-0.339312182050824,-0.439487918798257,0.173558293620742,0.375734381611974,0.183647016483237,0.137228766688581,0.0177937919607726,-0.0728387556039737,0.867421510359086,-0.345500282163587,0.194134822692769,0.642607350469169,-0.130670186997301,0.0560362370516239,-0.00895013212931681,-0.861205619445999,-0.415286895015141,1.00602132354925,-0.865799261926811,0.107768073849924,-0.0848887738245388,0.90759554257374,0.396036181878576,-0.428400285220318,-0.217296495177816,0.028557312148568,0.207920273134858,0.0846738942729768,0.200837862606275,0.385494648729479,-0.420792063910703,0.689246181508778,-0.272381622673218,0.420908155961424,-0.50571726839226,0.244743429202009,-0.174652321681858,0.558440215422141,0.108279362164898,0.448369310339334,-0.385939305971426,-1.50051634829976,0.461425992860561,-0.465746469293738,0.310458925030601,-0.00111238265163355,-0.270986980247041,0.123129227068771,-0.516278728180683,-0.0243608189465507,-0.468849391421657,0.364412588441366,-0.654094760464995,0.140475887528775,-0.124633971114065,-0.431196874821946,0.0566732299853246,0.289296632792611,-0.133461462984681,0.240114258650458,-0.0700215864446799,0.54529564177151,-0.316766940345447,-1.63415554685042,1.02807713129853,0.335447076934453,-0.599155045075322,-5.07543224256263,1.08881883929695,0.2390649293492,-0.184792592081332,-0.151246299794176,-0.10693787070798,-0.0360165259065758,0.239248572622338,-1.40420563942036,-0.108402897766903,-0.441325295495542,0.367661984560303,0.142426765333178,-0.129812906537966,-0.847927607949089,0.129483748922899,0.05672718117807,-0.286876537862942,0.0614128826475848,2.74115445897633E-05,-0.249233915068729,-0.480198868944378,0.0619303632248602,0.212947202166228,0.13057931510266,0.0243930019197998,-0.237051616450283,0.229167327087063,-0.160283237541727,-0.295617878912151,-0.0901174691218578,-0.183374402743802,0.0216608327294958,0.0945062376940027,-0.185496028664237,-0.0270252205007754,-0.0554890099706187,-0.0449556118771939,-0.0261060669108828,-0.0585555229073379,0.170383070460361,-0.0425153030749395,0.226372888582939,-0.00632406566033223,0.0781376068153745,0.0397765759014569,0.0863603758799522,-0.220460820993694,0.0534563090933023,0.0302945301710163,0.180250520599729,0.329537445677027,0.340950507599271,0.0930995242651785,0.339034988568927,-0.0445052895406393,-0.0987473927997611,-0.0184767352271121,0.099767622883712,-0.123515590162509,-0.157859904461601,0.0465042425815494,-0.024033297393308,0.0649608081093866,0.0406834865998863,0.0373799759520455,0.0539825501708106,0.0127385402111422,0.030697136177393,-0.00888916674559351,0.0721030215784642,-0.0331054941893991,0.0159648131850311,-0.0881531908137328,0.10847926987539,0.107349152641617,0.0736277085151944,-0.0462013813749801,-0.0389831770883016,0.286975312594096,0.00158925232016444,0.102199045707711,-0.318360332779833,-1.74334018875152,-0.930970188465615,-1.13616606775866,-0.278306260659596,0.64910008190916,-1.18609320547601,0.981286355806699,0.397715252116918,0.790801740971551,-1.44484608426275,-0.338946710737596,1.43347218943082,-1.15976666842831,-0.846479559612062,0.374516143086722,0.720942046357429,-1.26292325758335,-0.21201090165072,-0.185162500950409,0.110425602094448,-3.24517087497203,4.07489526863791,-2.0228557219927,-0.248775121535639,-0.0398557188207363,-0.71293464517289,1.28720954955062,-0.545096157187844,1.43781482980589,-0.411573217019291,0.0984436999824627,-0.0546947103056164,0.290210110140687,0.162843021161021,-0.286337154791434,0.105544734974045,0.199602810718606,-0.316206014778408,-0.0560059608126457,-0.84672841695712,0.0753305942510243,0.126703331638976,-0.0180050572528902,0.0897580073425469,-0.0406636420272878,0.0869579666751853,0.136949100626739,0.0975998782201094,0.0952427502248153,0.0202018484704812,0.190786291051507,-0.0750895209600645,0.0712081147176407,0.162895867565169,-0.0251997795162361,-0.105848676034203,0.00768153010606249,-0.0551519157974236,-0.023574754534736,0.141092240540753,-0.226637432477784,0.18916951146927,-0.00691036184719686,0.0358202450626328,0.391423592548732,-0.00132863653011787,0.0143037284771806,0.129000327916269,-0.358562216505606,0.32872085390632,-0.537886240025894,1.44309738498872,0.584136665712342,0.148667704385104,-0.0730191677994651,0.655892416999648,-0.14554436200572,-0.965870584358761,0.0163878107272423,-0.201773678775606,0.620395884933369,0.113329539837574,0.749524362021665,0.584520550008424,-0.862060155029859,-0.384991542489108,0.496489010333669,0.750884462790144,0.597373882692052,0.547088372647847,-0.78126470124055,-0.233045395979897],
[0.0480987289123321,0.0540733001792856,0.0550701803484246,0.402444832504045,0.0806648677571512,-0.0745726936553628,-0.052698149380361,0.240272068604206,0.129644874885172,0.045276894900193,-0.160883710397593,0.421158652301471,-0.0050452268546964,-0.463130240543487,0.0554287361322175,0.0998177734673711,0.163629994504118,-0.158794301268251,-0.19515195502863,0.150049863960643,-0.0236906931320841,0.0446522572624525,-0.287646624784622,0.0953156837419834,-0.0185688405843197,0.209534101858139,0.26285751273669,0.198856581057501,-0.662248054175277,-0.355939344501179,-0.0670547578140945,-0.0011216784424822,-0.093825239935285,0.117434374833444,-0.338023443956514,-0.0626287073432771,0.14018001929589,-0.0352126558578918,1.21363682761312,0.36789628364443,0.102296016535047,0.635517307755021,0.318887167837628,-0.0525985141857422,-0.120225578444557,0.138984864579261,0.14874053026711,-0.0140482527636405,0.181866810510195,0.0823099866562279,0.168332628420035,-0.156525904161834,0.0381967799926091,0.332828836521783,-0.00395577724440588,0.295793503479543,-0.429103398869641,0.259195858795429,-0.221098038161775,-0.634275926807226,0.0180263542631545,0.164447055162566,0.0464043023289771,-0.0186076920982748,0.0699592327453573,-0.391569261207068,0.187690102378281,0.110228715914159,-0.329544235655961,0.452195539812951,-0.316539724940298,-0.150764054816223,0.644729882545604,0.109424064495763,-0.0149683887710445,-0.00908150144347086,-0.541692669674278,-0.343659999242465,-0.809465331730937,0.169096575426257,-0.59238095803952,-0.558612255914281,-1.22913755896193,-0.107057748327969,-0.631671294727471,-0.231323946025594,-0.506844649845216,-0.194147610430958,-0.306238385323682,-0.245009710004569,-0.415904661127806,0.0363333101797202,-0.835756507668429,-0.0351262187492001,-0.756475574368844,0.0904204159856975,-1.12985640165615,-0.398423954454575,-0.176925179477425,-0.132845693714046,-1.01094132102928,0.175777226598082,-0.538968326014914,0.118041996093435,-0.733407566385169,-0.395230983502705,-0.65848484239415,-0.209612028176846,-1.31611471268916,0.274283108239326,-0.642166892197039,-0.140743799847829,-0.260348947820398,-0.153656009179751,-1.62735377614164,-0.195858215765776,-0.754993210754262,0.0377883878041953,-0.363710353341117,-0.133087017999821,-0.766061464966763,0.226638532989498,-0.285905216078052,0.0290603428758291,-0.159562009223839,-0.525021642884306,0.406659849236809,-0.159525392580734,0.231250304113473,0.00440820221322384,-0.19061773836647,0.153829485088655,0.0700195499342808,-0.14730959614298,0.0146310214881622,0.0281568555388434,0.0209992033829315,-0.0444421229006872,-0.135208927021122,0.315383675003958,0.130562452993221,-0.14433571474734,-0.267122181648811,-0.243138556829103,-0.127879289195322,0.213420929195664,-0.21962696209491,-0.267342239942887,0.14551552950739,-0.0141265079514878,0.010902574482333,0.0839005073332577,0.138230081896913,0.0590984176371901,-0.325961315655428,-0.270849182874978,-0.0497165824148873,-0.00306546267366788,-0.293714493446315,-0.214228187388388,-0.231035568109402,-0.382438020815107,-0.0269973169569123,0.027764306039952,-0.010036295491944,-0.0938303668020931,-0.091612634669858,0.245717950470233,-0.0893319807943123,-0.0973908096847508,-0.293391056551205,0.0113347676142823,0.0418612174971587,-0.261954426504227,0.0937280650228901,-0.100091324207415,-0.138310557316036,-0.0963787348182204,-0.0549640327441525,0.0580554145035483,0.0597460484862919,-0.0914757323808146,-0.230739014514127,0.160249373761774,-0.0606480828384497,-0.00680354729067809,0.205565639383395,0.067436335607044,-0.0989751986507572,-0.0505353439587223,-0.195088134609731,0.186224759409217,0.208057840632476,-0.135220153318098,0.258189731687286,0.636521332833609,0.0417048492667092,-2.18845849831114,-0.923778632338311,-0.360959901714205,-0.567440206112693,0.214143420227731,-0.717731828164206,0.386356171225032,-0.0715995603435229,-0.00376001835907856,-1.29056618263141,0.114543868342286,0.0678983147419057,-0.267688028412709,0.187328203518474,-0.239507698711715,-0.0823701055193771,0.381309511920567,0.0574733684728388,-0.018013737151359,0.0379161414490646,-16.8200922254591,1.00451514932602,0.0233891712318821,0.00661682188852041,0.121534332853543,0.216666363763805,-0.377844115603698,0.110730529516659,-0.0251435452045799,-0.0746788162951415,-0.618360036114278,-0.678678738722253,-0.931417680811789,-0.335760929390859,-0.690911249645843,-0.942001719377377,-2.97072918562417,-2.94883089158071,-2.66990054517703,-0.80372619015324,0.00518059127990252,0.00556451843899844,-0.144525374241082,-0.10646119374297,0.00288043818997685,-0.202237473705652,0.00487618005310557,-0.0831760233615117,-0.0479745005052157,-0.186818295089938,-0.277949517949618,0.0450048054402702,0.0126824165509402,-0.0757344113403302,0.0154761218604391,0.0149474367549651,0.0935782739019393,0.048104393552354,-0.0250977629611231,-0.0270999438070532,0.0935275742076846,0.069677585428811,0.187629501127384,0.0221775154266568,-0.0226127511004413,-0.0548707589168715,-0.0774717852404899,0.134854661168639,-0.0765605389663002,0.0909970325710557,-0.145782027809394,0.000445912664969093,0.32426415331059,0.0441558013032069,-0.18063923497927,0.0768348454386341,-0.332131745951044,-0.846530443348058,-0.102646036124573,-0.512827256699474,-0.0881531342279005,0.206947727118379,0.0940274587919339,-0.0592236723917541,0.0453414050728045,-0.0446944714220655,0.306087900087094,0.388158428727035,-0.1552974184349,-0.0561862585840432,-0.129760815878693,0.117271691830585],
[0.0329909851478035,-0.178612361684715,0.168922847266005,-0.297618798350615,0.397353608344129,0.335693414705561,-0.472429875846239,-0.155889555852862,-0.429332869809249,-0.170048399011785,0.096238917565182,0.490948715317641,-0.0269317796893964,-0.00873607489177145,0.107960367512186,-0.464239973024506,-0.629282762343111,-0.252019630639494,0.503980869806835,0.260082124444833,0.31470417295404,-0.222792363325495,-0.139513577946431,0.314454065969104,0.284050556441244,1.74352218974084,-0.216710042824382,-0.644782916024212,-0.314764998764139,-0.139505649265817,-0.087803991106403,0.0380170563659765,-0.187880177677491,0.072322310634732,0.137771666086506,0.0298837109796329,-0.163909455143243,0.106652121026665,-0.087996328263541,-1.24328269038714,0.193907076122814,0.263871153420792,0.108852902607746,0.74741261303074,1.26658744853321,0.134856661976678,0.696215800410017,0.1757846019583,-0.30038073602183,-0.0758378059248664,-0.254203831034032,0.407295517471203,0.301249958767725,0.281662032001982,0.389992650241272,-0.20995718837221,-0.0819569989870731,-0.389569165549281,-0.336044157601882,-0.00760685671485749,0.238267323807455,0.42231509667666,-0.222941581270117,-0.0407467155295513,-0.348894485500519,-0.621076116017074,0.255088893763747,-0.486656796001343,-1.09314546893016,-0.0439952063460699,-0.0967753291396786,0.262729759469334,-0.275392171925304,-0.232163088128779,-0.250606816676261,-0.350400918380555,-0.225352426113877,-0.239093382624967,-0.125508630021924,-0.153397036967755,-0.230605892781681,-0.232063136432776,-0.480009881163178,-0.185123785879633,-0.787180495060489,0.2293344860597,-0.61785400915719,-0.0975661121910762,-0.0540825385864733,0.121642289331921,-0.322343705851384,-0.320292174004784,-0.288191036099036,-0.106072523978664,-0.163634630605467,-0.0728033190085768,-0.355504254736753,0.146388438341947,0.574560545949148,0.449341520176957,-0.358792896554731,0.110067848450505,0.221303815539444,-0.169347201811071,-0.385588790867657,-0.0988715728266081,-0.339857195815233,-0.126266208843236,0.173169477937138,-0.396553266589152,-0.224979686654343,0.0980037758475579,0.747053863023379,-0.15943787673251,-0.784350928578422,0.052251681754179,-0.264944583021122,-0.158415109816525,0.551632286268269,0.152117668425149,-0.25288770039054,-0.0414066450436197,-0.101608651491847,-0.143612708718299,0.0134033966915326,-0.420259532734606,0.489831923595082,-0.344861249656639,0.72196251821015,-0.190094589710434,0.163580925910321,0.234617471300779,0.0928703472230112,0.12668245395903,0.300253739638535,0.330451665750717,-0.132199170997869,-0.247396052945284,0.398832977631474,0.0767867062027836,0.348265113618944,0.379359006802891,0.218052268547181,0.0904035108753555,0.230135894277542,0.0651844131033082,0.00334497463203054,-0.0454170017318239,-0.181575021825501,-0.0773565170129102,0.0750945959550788,0.274808704429418,0.284035549350588,0.216140230599059,0.0193956765230084,-0.463014173058071,0.152486544606915,-0.206200826355097,0.0318826431191821,-0.00488534015384123,0.0828105786283633,0.321660378401267,0.0889125150735767,0.0355284857115794,0.0568545617304093,0.094046326774537,0.0590956562295533,0.284232833468198,-0.538700230919304,0.023637302575701,-0.141415804816231,0.20830850912293,0.159359251827417,-0.0602452646235068,-0.134502306397928,0.0229524681083605,-0.0650722404420022,0.317037853450799,0.0780751495992365,-0.0587731070258358,0.237554109746801,0.0706093862219289,0.0771631621933371,-0.165113298418695,0.0115059066584379,0.0335511773155502,0.362917708705493,-0.341614148112923,-0.0428900877084145,0.135956685245288,-0.497447980792297,0.0759062297266522,-0.684000593167439,0.0766101926760428,-0.0404879484254704,0.608274042256868,0.0684688191783389,-1.6083607164097,-0.974409832082191,-1.16767535863343,-0.287518320071055,0.406647840760293,-0.363535286380495,0.580332483597579,-0.360811195144002,0.285841540339891,-0.980376228621844,-0.151715237704067,0.526491503896372,-0.935275564585908,-0.373147225177193,0.264983647735662,0.480686047477753,-2.7646770963722,-0.343014726751277,-0.0316833332913599,-0.018107249229974,-6.29135669453663,0.305607904285167,-0.383410212965386,0.0379516207801835,0.26961435124647,0.102736044541947,-0.00894883093265379,-0.0935218559701268,0.13593282950504,0.284913449342652,0.14955335995657,0.0675817284781065,-0.149578814013295,0.506819002494666,-0.0538235200642095,-0.152784216419991,-0.491521003354587,-0.602753854980477,-1.13897222194857,-0.49909252761149,0.0920677960324332,0.0112678945931023,-0.100687067984923,-0.277887318783594,-0.26618279621402,-0.451743829939802,0.0942322333703419,0.0727041161279572,0.0122533036293954,-0.0563654324051824,0.0854498340373559,-0.0787447234760108,-0.0209994969747228,-0.129443465947074,-0.0710672944245315,0.00029787895126012,0.048055069627615,0.0900589349497065,0.211772486551719,0.08605612277699,-0.0535278127659549,0.0735720040537678,0.0833284787459474,0.03566367034455,0.135784122175467,-0.025124628243992,-0.098637672030622,0.568529761090338,0.300717765071634,0.0211484240779851,-0.046097322245231,1.17768975937613,0.523208961178872,0.0716796527254767,-0.29047265380292,-0.264433743677063,0.281231839375928,-0.0680100679443297,-0.545663860599906,0.677931708814223,-0.013239070685627,0.646402283624772,-0.958457499735872,-0.144209866623732,0.297058709433167,0.306189654576874,0.104375404615525,0.135497360700393,-0.290654404795498,0.100459347568007,0.201761459892414,0.217841685543293],
[0.0705420465903903,-0.904126288989053,0.483985726051899,-1.36952070678004,-1.59008346120288,-0.057041621868217,-0.668592016589547,-0.244841139369804,-2.20040583417935,-1.41851940845171,0.878518522141769,-2.76554202109207,-0.741749353897709,0.196687205599884,-1.3904551793504,0.461579898802608,-2.23698015868938,-1.65803829967983,-0.030388197405934,-0.136441961559506,-0.135737823036783,0.746639854853862,-0.111937956073658,-0.244588413790268,-0.326510940285807,0.103208046781176,0.199905343222989,0.886684588486394,0.208320777219556,-0.310504111080594,0.029041295853984,-0.179693413093625,0.0130830979318688,0.237896383079308,0.0993616078792949,-0.237629580772831,0.00487275946065047,0.0429725369078201,-0.41614547137578,0.208867976313929,1.11492401751881,-1.21646676301083,0.248451209608459,-1.86010008950315,0.605207763816941,-0.215740602985548,0.0840912057174896,0.935118303575197,0.608538376633732,0.130565733592624,-0.0346887131033023,0.0153431000052902,0.0148341045505573,-0.0781046145513526,0.156468252692418,0.397126212923128,-0.136434202382366,-0.110963411059729,0.503340789581585,0.201495457292409,-0.00701305186627949,-0.0169464088735722,-1.73011736413979,-0.14164399240171,-0.100826060659734,-0.399642917916716,0.613093770398777,0.511925611224625,1.14415210721453,0.262427114895954,0.0144598080120802,-1.57548355343537,0.0427237450562463,0.389404206117295,-0.047774024136334,-1.0616870851639,-0.0271541389951918,-0.0533257648556258,0.566729457923532,0.236143396962354,0.140752719801439,-0.177774047729687,0.0114657422053701,-0.00553389117489365,-1.56206327387952,0.573172986449713,-2.2479663247809,-0.0660770025196365,-2.74752396704616,0.355085802644108,-2.29080434492895,0.0970127277389682,0.0865681130573622,-0.262330589772365,0.123042548272234,-0.422530119844144,0.499586008825059,-0.0582385416161348,0.197615042896172,-0.313919769819013,0.137954072365058,-0.106180181016616,-0.855186901969685,0.169976889305776,0.170928404116317,0.112049979085046,0.106914950538576,0.165384747460264,0.187815293763467,-0.057630259735907,-0.825925327705963,-0.0803859762870936,0.269217439544167,-0.481073481429243,-4.00427621327577,0.254592413926624,-0.190380340757263,-0.0299568975753804,-0.464844090266086,-0.20797780749954,-0.749420756708717,0.234785160852404,-0.464691375019327,-0.765553123782563,-0.382510352861931,-0.619842521701477,-0.145057697482259,-0.489265332825273,-0.26213386207978,0.394755436216383,0.433771352972099,-0.459706420386627,-0.0720618767476414,-0.162655427648324,-0.0819547805174101,0.090232494475462,0.380956963362399,-0.0911684412535485,0.155062044321873,0.0731693770081421,-0.00820063764858858,0.321049667187791,0.00096262461897187,-0.371684465462858,-0.149934441847073,-0.0262878733342161,0.0104102430387628,-0.0037183514560429,-0.310581566004436,0.104451202333661,-0.193917042968572,0.200500948325069,-0.199728266879428,-0.088446798774886,0.041505445177939,-0.145302689536789,-0.0400074171709869,-0.0620944471455599,-0.0871337209920334,0.0368399937167154,-0.0225891229112144,0.102086701194777,0.0650715238748761,0.308584367005184,0.0537851553114507,0.0275401441227499,0.170145296646715,-0.116196891010728,0.181684368204999,-0.0510808733544781,-0.0596894200061113,0.0798378991465875,-0.0403566381393738,-0.140525630058312,0.144042504615634,-0.00845156471080657,0.0381593067360588,0.147649140743614,-0.0495625230732169,0.0439366789495819,0.297046376087255,-0.049251237309224,-0.0563508838741992,-0.486716928666392,-0.0325476382993799,-0.0182588313658997,0.234606620965111,0.719300578106446,0.183313768222808,0.0337162347736917,0.634531219108965,0.0478236163951739,0.630568562180279,0.0764666499047302,-0.0258334044475864,0.0808844232729282,-0.295643996362828,-1.19198888005695,-1.43003069395929,-1.09188728316714,-0.111642694135772,0.248074370978621,-0.606224454617838,0.450553274762065,0.376133578914161,0.454042780384532,-0.868232056925372,-0.234662479744782,0.848454206799783,-1.70115354366438,-1.44025975071429,-0.139332905503578,-0.00178369939773795,0.0576618000221446,0.0906380588652467,-0.121193085394181,0.0862306743003742,0.433009986014951,3.60244596346314,-0.751365529585022,-1.98335337763124,1.35025435080114,-1.04060491296914,2.90212864445938,-1.2920188069307,2.15637429601016,-0.650260981174501,-0.210158997258039,-0.348988805845601,-0.000755970753018238,-0.124401525362647,-0.420466873201357,-0.324866010667696,0.326315686406211,0.0195646557235065,0.372386494455663,-0.451498098053206,-0.0768324013160282,-0.310867377471261,0.0626001812306564,-0.225183825725909,0.0660993116464352,-0.286718979104035,-0.126553820459544,0.430594466946936,0.0690457318654157,-0.0837883219843134,0.428139667862639,0.0521076744123378,0.167808518410312,-0.288664309763032,0.0593498916224766,0.173484316473789,-0.0269436852005996,-0.342567252165236,0.0145153644690347,0.204544569014284,0.189351125113383,-0.0679239933847603,0.0663407223643493,-0.0141077266187093,-0.162360581031847,-0.0332198777245452,0.19013585630108,-0.266553351340162,-0.114299263988415,-0.127758593325089,-0.848966584759989,0.654456023123372,0.352909675642521,0.1503345697525,-0.00480104881241746,0.749260969143967,-0.191441774404278,-1.02381912821414,-0.0189713091110345,-0.366306246792963,-0.100277143454802,-0.189125383207499,0.56423776529255,0.0460936678721224,-0.93071769299158,-0.544952255975529,0.317264100889836,0.346519851593886,0.506089396502493,0.01010517205764,-0.471326495605825,-0.29517902460308],
[0.0352943033496677,0.110442382296145,0.593211720706849,-0.0966272917328134,-0.603949451732759,0.770382403949028,-0.155935769373869,0.0335171729899879,-0.0198049586753176,-0.0982954958308818,1.05630150278829,-1.03697768316669,-0.327501373814773,0.62424356722587,-0.0891519750695193,0.649799366731951,-0.694580733503744,-0.980937201288613,0.76056990497032,0.0953437424914538,0.186458480801068,1.28630042486029,-0.173489514820464,0.0794820977728603,-0.0950646951778498,0.0782217409364771,0.129849494137341,-0.628532152334954,0.0579064952776792,0.419679314686385,0.0252073751076079,0.0849523739563051,-0.124572201326526,0.503695363207774,-0.0232207575519391,0.0775302954054635,-0.141435489120618,0.251487998884122,0.709460855965023,0.540114039400169,-0.375717866982422,0.451436049006249,-0.594657757681856,-0.651073936248476,1.31295605505618,-0.14715510409695,-0.181830581419488,-0.111386244039086,0.468714061276782,0.0181137619815618,0.196223741542912,-0.0110731555623898,0.132023641546697,-0.0272605215060249,0.133496295424752,0.339609568758988,-0.339034643145281,-0.0414358047038342,0.288429744179858,-0.136001867912909,-0.145358880206274,-0.155640122065039,-0.171824535517793,-0.0373343779468881,1.10746964601317,-0.664462187568862,1.55164057061027,0.330100910389873,0.35088279872091,0.467314686206889,-0.835629546070283,0.142605513452483,-0.695996748210637,-0.446403884313901,0.674132765637953,0.518078542022566,0.169981192125562,-0.18636801502908,0.470080275669672,-0.126395223658749,0.26840874927365,-0.267669467243705,0.297820385681399,-0.108963200131199,-0.613094429621295,0.644688541116937,-0.629940110168151,0.0749036290704823,-1.50525624996594,0.882283404279753,-0.827891233735275,0.353193085974266,0.101625173516817,-0.460313150568336,0.0117013027273519,-0.662345432946702,-0.189662043935657,-0.0900945534544814,0.546197202154972,-0.13080345696481,0.239818335875015,0.300266686345372,-0.651416554992307,0.109584401574001,0.183271416758819,0.11987540713866,0.00819193152017924,0.168518788116131,0.524783064384151,-0.424730728180166,-0.547573208224578,1.33967447937996,0.579346660357148,-0.242278842595811,-3.43478174096154,1.09548138914101,0.0658774748728853,-0.307273824950121,-0.317378859379667,0.303284222718577,-0.309322871720485,-0.0641597687648683,-0.787940323727304,0.0892628307149771,-0.223276131780214,0.635460247637142,0.594277675798771,0.410211284229804,0.27520555125948,0.0695442019471616,0.374115547903424,0.0130522618418133,-0.197290144574444,0.0606866891537194,0.032142686040504,0.438367312023243,0.0889692050475125,-0.0964862621714565,-0.235176962149543,-0.135821577869013,0.106484104488021,-0.253532130632322,0.180301682451208,-0.0237680639004444,-0.0593768777100911,0.390590884337056,0.114979899660091,0.304185219529256,0.00133524761333674,0.00708548002055763,-0.00399213853767205,-0.131999475949965,0.021406869865706,0.11571584001845,-0.225901590683104,-0.153851061357798,-0.0672044420320902,0.242499846332122,-0.172386890883141,0.0129812702743385,-0.033633675888716,0.103119077251854,-0.103683008547172,-0.229800118672434,0.0279054364401699,0.0814083026012187,0.169460101463026,0.143162784929031,-0.251029611739028,-0.0216256830043969,-0.118204363670441,-0.0659011780020348,0.147798519301963,-0.198248494185357,0.0616684651648539,0.0185205124471877,-0.0784769729673872,-0.225564919025673,0.0526332177022794,0.0862216594002741,0.114068734069647,0.0790690570443554,0.00757345470655362,0.0743182857956799,0.0182613957929694,-0.0219749601282838,0.166179367535917,0.0239081952686153,-0.0186638193443045,0.309769042159768,0.402605737973421,0.178898776475816,0.141793809577182,0.065998343027424,-0.269863416476622,0.0372215577658746,-0.387627057131026,-2.07443975999514,-1.56114874804715,-1.80428147297593,0.081761460358733,0.985964706069603,-0.841752325995912,1.16366437356918,1.0559041941477,1.30338709279351,-1.6497409886932,-0.0715315554684644,1.61420708122384,0.469941296065273,0.915167123598786,0.1339426038243,1.01339033871195,0.952435097745159,-0.55869094866408,-0.180052150342833,0.0631122591174599,-6.05303686882762,0.970411666131054,-1.32306801596661,-0.773230018295609,0.482753212861983,-1.07886693660669,1.79561849957837,-1.00111647724721,1.873465863049,-0.288347229091829,-0.0202301275989664,-0.130225196753776,0.170478526054114,0.0306091327750639,-0.355817610702611,0.0887117322372192,-1.40466485567093,-1.32598599709152,-0.722200432824842,-1.76982933575683,-0.200655586532146,-0.304793116722072,0.239339887042147,0.0366658304077081,0.147751391928184,0.217133387021189,0.0919656776648494,0.189755603664955,0.0667168575726748,-0.146485315955492,0.339370074367368,-0.0320637882847749,-0.0513976377198254,0.0869117468779778,0.156422144828247,0.027089933466496,0.0192005837655194,-0.301735578849395,0.0178653814331957,-0.00399141544053182,0.0402328856380688,0.122753874305753,0.362979478419329,-0.0278739047591301,-0.27863134391731,-0.0874299021585111,-0.143284816522075,0.788129593482405,0.0548473273238974,0.25338183923212,-1.40151778992903,-1.26995592875247,0.16594171456541,-0.407284755678341,-0.0976078721082691,-0.197227983105217,0.471034651832734,-1.20834525289809,-0.159943560787926,0.112511584057859,0.849117314347237,0.488705170756475,1.0116660379758,0.0341948202126034,-0.632713249334921,-0.264188463042995,0.655303879888719,-0.0811096748715111,0.607431252091784,-0.0112417028666181,-0.219021422055218,0.622296880886532]]

NNBias = [5.32237321834919,0.539784275025835,0.0562315861830115,0.74311425603347,0.574523745726224,0.145451818229369,0.310991412989352,-0.244924717572837,-3.79712552647442,-0.846598622052557,0.405284120670962,-0.845174634555845,0.353251112268907,0.714594360823962,0.492989228104464,0.612418428157902,2.56482259657262,-0.16539133853796,-0.0566830217587984,0.500272279542515,-0.420847863241571,0.913022625412839,-0.95996626359513,-0.701757521465954,0.959930202004573,-1.64185984132579,0.271170083192466,0.596143782060747,3.27421898643386]

HiddenWeights = [-1.18302161410255,-0.689415791688668,-2.38119703353245,-0.351382747645439,-0.643231215489457,-2.08654446481066,-0.965655201183896,-0.906965804478668,1.5857770528258,-5.70167498808861,-1.06599920148791,-2.41979489615427,-0.52449606708839,-0.317143639008225,-0.352451963735985,-0.480700966049291,-6.13071122195773,-0.918509456322938,-0.577703580124827,-0.67074421984548,-0.829212323309725,-0.38306176613389,-4.95885299445902,-4.80629879701294,-1.02864918265252,-2.5335732678056,-0.48000324511684,-1.99817158694432,-0.947332247497617]

HiddenBias = [0.420610813234142]

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
  count_input_size = 289  
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
    return final_logits, sub_train_vars, model

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

      (logits, sub_train_vars, model) = create_model(
          bert_config, is_training, input_ids, input_mask, segment_ids,
          num_labels, pos_ids, percs, weight, pos_bias, extra_vec, use_one_hot_embeddings)
      (classify_loss, per_example_loss) = create_loss(logits, percs, label_ids, num_labels, weight, dtype) 
      loss = classify_loss
      hook_map = {"classify_loss": classify_loss, "learning_rate": tfvar_learning_rate}

      if FLAGS.multi_task:
        (lm_logits, lm_sub_train_vars, lm_model) = create_model(
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

      (logits, sub_train_vars, model) = create_model(
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
      
      (logits, sub_train_vars, model) = create_model(
          bert_config, is_training, input_ids, input_mask, segment_ids,
          num_labels, pos_ids, percs, weight, pos_bias, extra_vec, use_one_hot_embeddings)

      preds = create_pred(logits)

      restore_vars(init_checkpoint)
      
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={
            "pred":preds,
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
          tf.TensorShape([289]),
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
        writer.write("%s\t%s\n" % (str(result1["guid"]), str(result1["pred"])))
        if i % 50000 == 0:
            tf.logging.info("Predict [%s]->[%s]", str(result1["guid"]), str(result1["pred"]))

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
