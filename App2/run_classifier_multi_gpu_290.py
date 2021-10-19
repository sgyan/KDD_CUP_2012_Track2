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

# Joint Training pAdjust Abacus Counting Feature 202007
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
    "extra_embedding_size", 290,
    "")

flags.DEFINE_integer(
    "ori_layer_num", 3,
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
        for _ in range(int(FLAGS.num_train_epochs+1)):
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
      pos_bias = float(0)
      extra_vec = tokenization.convert_to_list(line[8],float) #Counting Features 290
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


NNWeights = [[-0.0241975982914548,-0.0142563876662226,0.0303752327527659,0.003683524815132,0.00849580023797987,0.0101312251482174,-0.125546746902753,0.0333587177304159,-0.0419090448149897,-0.053333953917916,-0.0465370138284538,-0.0761874037759993,-0.00174359036904181,0.0241132247406917,-0.0700711301570146,0.0108605936866005,-0.109236346750041,-0.051749998902673,0.0709390650241466,-0.0055853013739042,0.00854020038416643,0.0507554968413989,-0.165048863077912,-0.0168062640100523,0.0541383425980019,-0.152965796748428,0.147027026534232,-0.128338280742323,-0.0621197854583919,-0.11838871034529,-0.0165957106104882,-0.0152532107825873,-0.0226031883734695,-0.0842509341305823,0.00155787581120312,0.00811858380885958,0.0126340887221026,-0.0754808735856101,-0.0803217330092751,-0.166500075190018,-0.1152770802468,-0.00774652524742211,0.141399947404478,-0.0729640899192722,-0.00436485766736631,0.0894275247313785,-0.0339838135418123,0.0319777707018921,-0.0646339303072199,-0.107570435883815,-0.0224103666186041,-0.0931794924537249,-0.0160093802430703,0.0218490670291113,0.0660964526700228,-0.0260433399441553,0.0836287878672851,-0.106972227683973,0.0312066489393757,0.0270463489402688,-0.0905317545312869,-0.0448198670035086,-0.0335966003095932,-0.135340937224408,-0.0879887457859189,0.0163219091525207,-0.140730409554999,-0.115865699814548,-0.109276705065021,0.0927067262687327,0.0396537570489121,-0.207676414873658,-0.0454002611530271,0.0257143277241321,-0.0498203371196135,-0.08635746045676,-0.107275158032406,0.027747003944844,0.0242326411415473,0.00783581883251172,-0.0206999268110652,-0.0273754467955471,-0.0716219251715632,-0.0528211688273267,-0.0727547043892679,-0.180258599845288,0.00616500363796197,-0.220196772373165,0.0117490271960467,-0.0403144380079317,0.0542722147772486,-0.0602575785800725,-0.0748205140208065,-0.0860950444764523,-0.000142634412472519,-0.0315404854369262,0.0216938066120005,-0.0480003413588344,-0.0426575623874478,0.0487864739081698,0.0196062171759732,-0.0508390219822827,0.069542125067278,0.045721369771526,-0.0190417594759129,-0.031754282018973,-0.0670600677022947,-0.0364410779872791,-0.0346710266628379,0.0812162824890425,-0.013990249346243,-0.10212213817147,-0.018864534080527,0.0929913346691422,0.0363181583641307,-0.0866558450989252,0.058631411529163,0.0104075637651598,0.0156199537347788,0.0522783403116587,0.119284301044772,-0.00890759427904314,0.0177756203589588,-0.117208830207273,-0.0688745445797883,-0.173039923135715,-0.0632959747510713,-0.0222817180913279,-0.0138922975521912,0.154862963681731,-0.0357812633755255,0.0112623173948289,0.00896653431212093,0.0771550761697589,0.118827376078675,-0.046624675642557,0.0304140763842518,0.0573106369675291,0.0922871258557141,-0.106959872376067,-0.101174511157371,-0.0281649488399445,0.0249818751629392,0.0239403852255945,-0.0157720946937537,-0.0080903989749925,-0.00126733891447717,-0.0598653440294377,-0.0853583738608985,0.0219389075787174,-0.1396889587016,-0.0839213764079508,0.0201066379016938,0.0466622223053466,-0.0281028986899217,0.00191988868616841,0.042437333908898,-0.097550683390665,-0.0186060632738003,-0.0431769115868054,-0.00850358666021122,-0.0679126522161724,-0.0990876820615748,-0.0686385225624999,0.0241155561017589,-0.0163680557080354,0.0055487565725996,-0.0230521860872077,0.0811186686674009,0.0161698016033715,0.0210779717467545,-0.12356561638858,-0.035007673397473,-0.0373412938310029,0.011927365297532,0.0143032938975295,0.0273388364706029,-0.00652600246531695,-0.0693961203325746,-0.0240605198106772,-0.0376520747584966,0.0128605257373063,-0.0478039229953709,0.0179435814168221,0.0289152401231221,0.020551863709478,-0.0442551139383057,0.0288955119483443,0.0115142191192053,0.00149678576481935,-0.00976360570811962,0.00559451271563982,0.00505443478724488,-0.02648841436048,0.0482003012224752,-0.0316378896570504,-0.00260080688753051,0.00282571167966622,0.0787465249507006,-0.0185290529353289,-0.236742685843861,-0.533927537890542,-0.305941431299247,-0.138765461103977,-0.0206559447695786,-0.0753907702164314,0.0072673887518055,0.119773700186042,0.00534506122681785,-0.315177061452649,0.0188072182019802,-0.046257252159994,-0.106152864286848,0.172956710459522,-0.124440222635642,-0.101328665519065,0.197419384578175,-0.120904237869779,-0.0510035323475921,0.00665196649206073,-1.5445471017676,0.0295197590176298,-0.0319708046839123,-0.217590439564411,0.114126608008171,-0.035172927595439,0.168214174619419,-0.051323349459436,0.171095796228094,-0.0986910635755914,0.0108690792947024,0.00259884520151774,-0.108689286350335,-0.0246577791604236,-0.0561447225712647,-0.233540988700729,-0.265434009462437,-0.198580538069145,-0.321344518416642,-0.266862914054281,-0.0876193741727923,-0.00431504711662234,0.0389836469289284,-0.0589330817810412,-0.0231349902385563,-0.0208744283258358,-0.0509697818569425,-0.00338282044752502,-0.0839164905959923,0.0267682462673735,-0.058655413793175,-0.024886957059434,0.00771524597174739,-0.0360757987712926,-0.0248495701396048,-0.0439548405754076,0.0391688765135449,-0.00771753642980393,0.0378071970584621,0.0430225097194297,-0.0289432181997836,-0.0381571124155305,0.0853810959560824,-0.0289407770846573,-0.0469341039509722,0.0248465805868257,0.000944865020232343,0.0576772607554319,-0.0650622788900049,-0.0466491272651371,0.0443071219389295,0.00736247591635183,-0.00580776072580899,-0.0363923269612071,-0.104776880158424,-0.0629085942178422,-0.0140792629416969,0.0257646228021263,0.0302495314384806,-0.00579686303922637,-0.0494538670901016,0.038830409587442,0.0425487309100793,-0.0352051743858151,-0.00997578153322554,-0.0289627402286906,0.0486969560618757,-0.100532364983156,-0.0216084511591784,-0.07245567015614],
[-0.00813113335291614,-0.000747235196942902,0.0408679717952969,-0.0137273851363119,0.000281842276605338,-0.0195511746977268,-0.0573155182510236,-0.00457874636100832,-0.0366246121085046,0.00584679995317578,0.0291456728423907,-0.0165435282970286,0.0368821094363474,-0.0607211906002009,-0.046464273719816,-0.0400779991554823,-0.0907530590225077,-0.0687908722513538,0.0307259638384667,0.00329708367728213,0.0324187848747128,0.0535442992532329,-0.176468922539706,0.0297815455834287,0.0688828124473426,-0.057267475629759,0.13159412426182,-0.217450283754128,-0.13844162864382,-0.165817801625711,-0.0566956793454007,-0.0734403647450465,-0.0460347988458059,-0.0721378132239213,-0.0328792461166645,-0.00964475342255981,0.084605956798053,-0.0816854189544983,-0.130161170237748,-0.174211744106306,0.041541877930098,-0.0305933269833194,0.108066911871014,-0.0379211172829244,0.0262298087526687,0.0973498262471683,0.107677812211482,0.118540105334208,0.00479860578842273,-0.0362379195334811,-0.00250531280762037,-0.116979577442727,-0.0217553626298687,0.024815373417592,0.07414481001093,-0.0385143750007633,0.0301649996266813,-0.0583448494599411,0.0386393468682358,0.0301651254391062,-0.141168550059363,-0.00626072366059799,-0.0769441511835186,-0.160980182422078,0.0142493992843708,0.0273858894917457,-0.186522507445177,-0.140269505621169,-0.24785050779889,-0.0137283667348886,-0.0470948494658725,-0.231580042465929,-0.0121653325673507,0.0117899957661636,-0.129881657899824,-0.197114921273944,-0.117694648634749,-0.0410406821959906,-0.0771038051490795,0.0910148292383783,-0.0146667116016268,0.0396065641159457,-0.0947993797725978,-0.0982179356284586,-0.113444266494938,-0.29195125799853,0.087991334374162,-0.189379136389453,-0.0327743207386637,-0.0878564510783191,0.0438039134440397,-0.151893442078951,-0.0945749154755515,-0.0433290964043959,-0.0500230785939914,-0.00397079863986244,-0.0154463435679052,3.39051292035283E-05,-0.0573374049879369,0.157675190320139,0.0536053760755779,-0.0364631920118703,0.080488174225761,0.115147305878598,-0.00261144880924264,0.0238707163391211,-0.0360859174571054,0.0171921741969621,-0.0706893898940422,0.200764517316914,-0.0883331171319431,-0.0957859441208794,0.0169168111310827,0.0614198522050692,-0.0108816248330808,-0.0934598172234441,0.0527370160925703,0.029291105422844,-0.0797804050380399,0.0829052792115617,0.132603561664017,-0.0116467025000039,-0.0630512513184962,-0.0555432302259145,0.0122614211183282,-0.207323829953848,-0.0362422196380786,0.0214294081543373,-0.00384592707556493,0.3088660790961,0.0363303105935472,0.0412040623346904,-0.0654152274490937,0.0214203671469431,0.0699525579845686,0.0459017114945704,0.030244267406207,-0.0445254208709525,0.0457860712650999,-0.131729749695905,-0.110740279496996,0.0297374149923254,0.0300319977366986,0.0396559839389956,-0.132895832375903,-0.0403425506381436,-0.0505372804230258,0.0287483141567691,-0.0810897307558016,0.00426501651602882,-0.140541090569509,-0.0544982264044655,-0.0158727087745425,0.117962901610189,-0.042455182516797,0.0268326786613453,0.0397301342341646,-0.0838850668353985,-0.00624285327373469,-0.0198032217426805,0.00672247765266902,-0.106075950974393,-0.203388551277543,-0.0319301596603195,0.0339782678187743,-0.0717628301877293,-0.0816998677837664,-0.0371416286904977,0.0736942059949234,-0.0406363006160672,0.0538563023331977,-0.120750563128611,-0.0465669625018064,-0.0354305070296447,-0.022466554500933,-0.0113386235739544,0.0312074613145946,-0.0106463309426983,0.00542454909592935,-0.0178390195588598,-0.0754667403689341,-0.0184435734960948,0.00588130509513081,-0.0123913098807583,-0.0436122683581392,0.0180957684518997,-0.0570872395914435,0.0170377055097137,-0.0156637267408747,0.0685814938573269,-0.0523629298273751,0.038963546639601,0.00829548970633041,-0.0392506496869218,0.0494363809340097,0.00333905809597613,-0.0189397823391273,-0.0326807727165876,0.12578553964989,0.062032995820304,-0.379067880987085,-0.646751367669247,-0.410964921033476,-0.0866868152112566,0.0275609244289361,-0.118721084439792,-0.0278969934744011,0.112299015417958,0.0240749108787126,-0.381938282299852,0.0269244679338462,0.034208909187158,-0.113913812351988,0.294558369264,-0.133979656241452,-0.0858082979860648,0.277485035220216,-0.0895943985252847,-0.0508596523668706,0.0124677744170342,-1.96350991629729,-0.0438569516415607,-0.0179021856329069,-0.280973013046905,0.119328881691436,-0.0366290653988934,0.176431026026462,-0.0691854516489892,0.115607051322667,-0.0734459264076908,0.0390097975719411,0.0839029649701265,-0.0887327403509591,-0.0849801392556857,0.0235984199451977,-0.201272188188396,-0.324439146217058,-0.273622450097597,-0.3762138241284,-0.130110188682458,-0.0108671638194945,0.0480688074189082,0.0351720142207938,-0.00931928788481118,-0.0556426077914309,-0.0207542346815442,-0.00998703896057983,0.0442756704711746,0.0361327240641884,0.0741530927247586,-0.0458333385506397,0.0202433232249224,-0.0124268008627771,-0.0562184056719467,-0.0432840463422728,-0.0404500790709177,0.0860063775013267,0.00167465702322092,0.041466470306948,0.0794957117626699,0.00600904302994319,0.0441226385882477,0.0851316780344794,0.0522109975711763,-0.0155065486357569,0.0415139001610013,-0.013481168192292,0.0871245565624853,-0.0965480281922736,0.0241556449605334,-0.0415867129472085,0.0664387326325251,-0.083468444211287,-0.0299614405983982,-0.0343073968738348,-0.0152764150377853,0.0610287296249678,0.000801095120483132,-0.0201843602541609,-0.0877022429332016,-0.0525481471093798,0.0323625085133645,0.061451259621519,-0.00950021124176979,-0.0511870792317927,0.0805403929047302,-0.00697865997308609,-0.0638385196433351,-0.0130077371665648,0.029054818585237],
[0.0196897379761313,0.0650774783711344,0.111289452794933,-0.288186083568779,-0.136997063389372,-0.0287494381440128,-0.136289465836439,0.0623974152399452,-0.198898974702564,0.0939838063985671,-0.00441545947761145,0.00166241815205521,-0.0155467948617607,0.0206768317391312,-0.14191096092231,0.0279973787960371,-0.198483360891857,-0.105968851315466,0.0766773720869227,-0.251419073580267,-0.16118163801212,0.353252401112214,-0.230625411711242,-0.11301085184341,-0.0438534107713216,-0.570324685439604,0.0794951472380429,-0.650016966149597,-0.042174321251564,-0.076953423299591,-0.0761640861448561,-0.107466743085625,-0.18002914866151,-0.0215649191816164,-0.0825152730610612,-0.0335853587985648,-0.0215878964700491,-0.0858816698317943,-0.0552376106323417,-0.0202119359556846,-1.54368733199255,0.12155324387047,0.0802887879904869,-0.181350790017269,-0.285531912830537,0.00448625191117221,-0.215261189600933,-0.276423923615937,-0.0762735741615537,0.1886663572661,-0.126843617723886,-0.00201595092112993,0.0278826117397863,-0.0578878816854839,0.0214487916135501,-0.0527569369440562,0.0464387522661217,-0.213468412790655,-0.0724180721805487,0.0520051245072917,-0.202809228420267,-0.174339375712319,-0.139712106466132,-0.173200044497157,-0.129404017650135,0.246790824332697,0.123182197885551,-0.221169467287477,-0.130067918607406,-0.171898543452268,0.10844942165616,-0.40431562176499,-0.167795873875434,-0.214734326910785,-0.19506592940788,-0.156604868130305,-0.059859953540841,0.228753038080487,0.0450728583067371,0.254832143331628,0.0241786838392449,0.185182469539365,-0.00825684002694535,0.0341818164888791,-0.0812598812273601,-0.491033976415404,0.0213203742155504,-0.449471970311325,0.0749099080923965,-0.0759633545042828,0.0484877005122734,-0.291920629654131,0.0193239532844731,-0.0895402961824857,-0.169330333092889,0.0350068960767345,-0.105171987784294,-0.081663739512832,-0.0485828794487378,0.311463849037386,0.0014865145293791,0.244671029743326,0.0723309397351585,0.0566421932361298,0.0153763157570734,0.190497091733189,0.0292772716329073,0.197705509606677,0.0224916327011327,0.148757889742356,0.0511667578043452,-0.0882861804012423,0.0263766906904193,0.177314249588789,-0.0226703788168505,-0.12879160344688,-0.0264461945774897,0.254512263047808,0.0166994065525429,0.0550270326586028,0.0413514911629073,0.179410737908687,0.016564460976646,-0.0621906044022073,-0.0531755404489835,-0.0153772723479333,0.0294841174114499,-0.0385832172914279,0.266374631254918,-0.0723428968827429,-0.196904842074416,-0.140911644621504,0.153563653600149,0.31539455477672,0.10423094561392,-0.110596438190296,0.0732382516899107,-0.0232669812429003,0.0897688328577227,0.150334496324738,0.0663482257324728,-0.324637740501414,0.159009758069863,0.232612964984385,-0.0573716164528712,0.0458032505433426,-0.203951387379515,0.0249278283961892,0.420569916527187,0.00886295080913915,-0.0527382912840244,0.230275545433688,0.112242298310234,-0.0958724768595221,0.0526852543140879,0.135770406942152,-0.0347320737928564,-0.183498876463512,0.300969222759562,-0.0917161648132655,0.0740034252278074,-0.295014927625236,-0.0562158234583292,-0.0765906650213923,0.223487946183798,-0.101798801471487,-0.00954791947695967,-0.000104532067145942,0.160679374383808,-0.00982307566146807,0.0906414011109283,0.0530181679388972,-0.122890731886281,-0.122723593354454,0.0758723491008227,0.0141044771181062,-0.0371766544928624,0.0929306683851615,-0.0697441060560336,-0.0062671817826549,0.0202056755418448,-0.0785964170476332,0.0227370182899923,-0.0651671706183986,-0.0460748019088214,-0.0197215680775513,-0.0736904417853854,0.0337360166580562,0.0488319024786331,0.27220619554459,0.208525643533744,0.229664525739618,0.190733554659478,0.0847365475776562,0.185756541254321,0.137336947349622,0.236398415866007,0.287426287114498,-0.16895760278858,-0.083214823519009,-0.470093910878793,-1.04235072735595,-0.472036073861196,-0.227267208755618,0.073600476774713,-0.244392677511832,0.00621361901991452,0.385360997269986,0.0611359234257217,-0.615706060875417,0.275308858615241,-0.0277149089671765,-0.154029029185197,0.279623654351827,-0.200890285649417,-0.150673593226618,0.216992695851719,-0.100896252165516,0.013093381071482,-0.0203950895632462,-2.59478642658482,0.291606776502944,0.0536363326469831,-0.398080394786933,0.171751375647269,-0.461428849940661,0.188858733736094,-0.187197671701967,0.272429951714594,0.049613475084869,-0.120744230580038,-0.0549832631274311,-0.356968682118614,-0.207413869852099,-0.124789543767442,-0.424008963506129,-0.46046312836349,-0.37093474123967,-0.266659030999287,-0.571841225423695,-0.0760751749990457,-0.0587048427030675,-0.063503839607163,-0.0474052856178434,-0.0290936532317604,0.101366167137024,-0.0136822711396898,-0.0769570212801382,-0.207829096896937,0.0126787790823098,-0.076892424653799,0.114159633838663,0.195427784997221,-0.000167548631459037,-0.215726217834998,-0.00377458602165764,0.00951711677284869,-0.34961436054036,0.0733944740278028,0.0151682456924804,0.0974471554313381,-0.108815792843193,0.0694890343312198,0.0186629438079337,0.0161452330423953,-0.515189880273401,-0.353313787525256,0.146927393052827,-0.223599840977706,-0.113722054984557,0.30472548382181,0.375587358755763,0.319657580775566,0.105662373914152,-0.029043012656676,0.166531020380863,0.27718937623665,0.227864490733848,0.278208911267918,0.237202939838976,-0.0100027005675011,0.372559308474483,0.000425087973006483,0.0180423776455974,-0.0574789371296964,-0.0475003337613869,-0.0256912076235944,-0.252460299863068,-0.0629562068817904,-0.172058932671729],
[-0.026766495112653,-0.035533605062127,0.104954943116891,0.0804589302133874,0.0307591115549457,-0.226197764296726,-0.126565224326623,0.066416058073004,0.18956440909518,-0.0374011597196109,0.00459084491500539,-0.0986254441970318,0.00608541713050268,0.00361906599748353,-0.0598094692360427,0.0570214369562205,-0.150218724100151,0.034161570488625,-0.0690462073438371,0.0130764019994498,0.0252037832735891,0.115999769222156,-0.219921172247955,0.0395064272452283,0.0971077636923718,0.0306993457996998,0.162311847281984,-0.173103141086717,-0.191956988476195,-0.382187845998117,0.0184764829535573,0.0209166429628231,-0.0147138864796084,-0.0993318308442663,-0.184029028789141,-0.0108637840098398,0.0587060718642081,0.00704646445214526,0.0121491254665375,-0.193004180522803,-0.399397046226292,0.0636602731136103,0.152788790083423,-0.0669549650268948,-0.0220179490346586,-0.057082828792369,0.0704274926449177,0.112607971335342,-0.023121900602014,0.000728426721659632,0.0322987265581124,-0.101153354754693,-0.0862622094626946,-0.0180382875294277,0.0477756714264413,0.0234891028697729,0.0518581928196478,-0.0974182211548266,0.0582846194251166,-0.00265540996485229,-0.180665048549054,0.0887111342353497,0.0140716884233537,0.0361396115914437,0.161979988492552,-0.0116023238290987,-0.320330720355059,-0.225258343129623,-0.076882423441984,0.0113544793006662,-0.0373576878992359,-0.293810716094615,-0.0283135634357687,0.168630948097302,-0.031543217244923,-0.12782874138201,-0.018062186755366,-0.0939417708421193,-0.0984548083135562,-0.110288169652586,-0.0166008464266257,-0.0747504666869737,-0.11146966756743,-0.138418747718981,-0.078219756356478,-0.300954009384644,0.0894640536466775,-0.288617130044029,0.0507108414261117,-0.0927904353383012,0.0635197176609278,-0.164526011031578,0.0935948196525031,-0.143565291483943,-0.0656418617854482,-0.0131695758891719,-0.0701335675995748,-0.244650649907767,-0.086054737039676,0.0476303246747651,0.0406370902510857,-0.267920939740766,0.0839721524044444,-0.0162311651071671,-0.0808020359531585,-0.153148519077057,-0.0306915426876386,-0.123068758511388,-0.00551552974259763,-0.136159605560926,0.0207632200443782,-0.122507184262736,0.0172523391878849,0.0418017719196737,-0.0387551663968789,-0.0241274861143542,0.0239191388207266,-0.151412871216457,-0.100342391041592,-0.0545624564753748,0.0571008035876437,-0.128132526082377,-0.0217851753833469,-0.207639124181172,-0.00520176557746233,-0.245813972256844,-0.111569941395225,-0.0588065014306276,-0.0772626151911646,0.458117301902817,0.106787930658866,-0.0583906827566231,-0.0705132619254188,-0.0296219347729717,0.156942073312617,0.0775153495120628,-0.018398589080421,-0.134027542769466,0.0704312160776285,-0.0678585090392875,-0.140478084600611,0.0988406807045371,0.0135167552869737,0.0756075161894313,0.117572893056855,-0.0179709222771096,-0.0636710910235392,0.0856159451180526,-0.0700543958779767,0.0379268189541437,-0.28313107195123,-0.20048401180884,-0.0738866299535211,0.0910336312061122,0.0527557697955595,0.00778191213562766,0.0156315784122044,-0.127316129689607,-0.0858235600986665,-0.131608348136294,-0.00200866648068227,-0.224739776002751,-0.207895226817438,-0.00771791779240414,0.0620479301758269,-0.0851009558500323,-0.0903121218146985,-0.00361071073870592,0.116574552918994,-0.0759715746359586,0.043902954716339,-0.165380706122863,-0.145845538786287,-0.144550902859395,0.0543346980912957,-0.0306372282673306,-0.0667810308973746,-0.0696843887390979,-0.078760913212181,0.0110721397454228,-0.0331499309487816,-0.0496302479912002,-0.010866262749862,0.0673864746134289,-0.0774342134273157,-0.0280001006712956,0.0200407019411784,0.00086213305206701,-0.0371691715290293,-0.00981000521042077,-0.0430615501315618,-0.145426950920605,0.0332903673944211,-0.013207038573932,-0.0194967993647084,-0.0855641721614419,-0.0212593494813639,-0.111008402756856,0.108593580781312,-0.026356251587749,-0.380486819023,-0.649078967669713,-0.417446425542476,-0.100425184965065,0.0321293315381649,-0.16742403447225,-0.0637647162016043,0.0541227867918068,0.0428311803332674,-0.448580686232661,-0.038036058188168,-0.0686055902811416,-0.435515569192145,-0.0371049276287737,-0.180114424585385,-0.215559190874264,-0.12614243922094,-0.0833431213960301,0.0126799648612619,-0.0374323754712435,-2.81883520521042,0.0966571828249126,0.0989383126354056,-0.332171287886721,0.25702297274321,-0.162441646612767,0.113559462181406,-0.154333210191402,0.171688183379321,-0.089711534247286,0.00513931223913792,-0.0278969918991986,-0.318538738080503,-0.130329499825095,-0.203951783690996,-0.406550548747426,-0.605852854762825,-0.578709744342222,-0.7528151040206,-0.33248890158193,0.0153599895844436,-0.00402390096164607,0.0984066418513263,0.0317754157905749,-0.0566638889674776,0.0199626061017236,0.00402563261618229,0.126563161061196,0.000155632954171897,0.0113452985453126,-0.0972669973029469,-0.0478995234269809,-0.0629630732984677,0.0428930797456769,0.0317542812137357,-0.0646359075223977,0.0462867814863697,-0.0985771343781248,-0.0394429292665297,0.0642412008255823,-0.00531926405893776,0.11399235433794,0.101005874945584,-0.0129494510198645,0.00475205852827991,0.248772006520772,0.231180593235813,0.133852104263354,-0.116436420306269,0.0246111648611494,0.0284574097270173,0.0646709400988921,0.117416818556786,-0.0137267130059707,-0.0433933042608907,-0.0451235259977508,0.0267820137171053,-0.0512366407316089,-0.0172902951782851,-0.0293357217489992,-0.134241887886151,-0.0193968207388935,0.13276126363239,-0.0607843486064126,0.0402733165312968,-0.0473152843785169,-0.01250982864788,-0.142992904120025,0.0196783478003401,-0.0488946079974718],
[-0.0009703185790221,0.0879006821404612,0.0921613238597086,-0.103042935188513,-0.00327216930662051,0.0566754797857222,0.0209884921407089,-0.0124987964528094,-0.0972795919024164,-0.0500366716070963,-0.0187458743716654,-0.0750207254841416,0.0105799350982258,-0.0362760897636054,-0.0459342343364164,-0.0260428749474348,-0.173628218392936,-0.129176620391221,0.06863272049753,0.00268915917834616,0.0820392555350893,0.139891160268132,-0.265110251765614,0.113402762828403,0.122096180456747,-0.120974514915079,0.0951822523251337,-0.501488624340594,-0.200417634407621,-0.219923695029073,-0.00225035201610028,-0.0202413947211705,-0.115525700705194,0.00311561274961947,-0.120397972753181,-0.0200979482245187,-0.134480476418883,0.0118522637282143,-0.190870534093866,-0.18615786611553,0.172742300584322,-0.0102971703437414,0.203817598008738,-0.0729969454044638,0.0235726868645189,0.170036952756483,0.107137307579903,0.0590277363073762,-0.0189648957877981,-0.0936429628148133,-0.0882059755668429,-0.0411483194244652,-0.00998695910262338,0.11954855053009,0.143022616645184,0.0133573084669364,0.0778061345476856,-0.127790343118086,0.0230279095449499,0.1012601977246,-0.201878125527397,0.0259581283008783,0.00997344631146377,-0.234087643598791,-0.00471847301856954,0.100894566569961,-0.209603838966237,-0.225694461454592,-0.251292196824262,-0.0203768882155315,-0.0632198558128675,-0.381058783302727,-0.0303516815554852,0.0159418983850596,-0.171714683445256,-0.214381534699747,-0.2082226442457,0.0229020764566263,-0.0916376309946816,0.153289473226788,-0.0143114300268998,0.0514988153303166,-0.0808680432684588,-0.0885895451774261,-0.0950672843692257,-0.412686095012551,0.0666578456669325,-0.320899562766621,-0.0781253048078598,-0.107628841151682,0.060107335859352,-0.223865050099758,-0.0761221021228222,-0.0594762811724546,-0.0275931883102353,0.0413644730563335,-5.50091270700275E-05,-0.0310637533898287,-0.0265422129260986,0.230434523371803,0.205530234173013,-0.0184189168454788,0.21658685791081,0.182588424073302,0.0105791058402055,-0.037383309816335,-0.0752480598661812,-0.00576188369537315,-0.0249624250366567,0.305465660063605,-0.0475643008453277,-0.111935835990749,0.0843572192747221,0.203134052998285,0.0944705410366372,-0.115709180519248,-0.00561545805691493,-0.011020894768103,-0.0693884548113825,0.133604677773039,0.28348402014083,0.0203489820636358,-0.0722575912587614,-0.116929325450307,0.00826162402985445,-0.17336753260403,-0.0429493360793051,-0.152463110199379,-0.0949453044615206,0.26470582787352,-0.0210389396719255,-0.0456144109788224,-0.107355833591802,0.118914436485511,0.164049054850338,-0.00117265211058458,0.161568310704019,-0.00158614881521294,0.0926151121994834,-0.0652064087436694,-0.112266956716579,0.0843351528681768,0.126418298879153,0.146479491851277,-0.242886909459059,-0.029658489848888,-0.188624578005569,-0.0470320244853175,0.0632190820038458,0.0221166179338759,-0.225032688753366,0.195648321518163,0.0476274305061938,-0.0436146839537777,0.118347145205137,0.0962348222343023,-0.00647010919121415,-0.179425022386913,0.0382411750857243,-0.110514285647597,0.0315465435193194,-0.235791190054126,-0.112798154913978,-0.0392086631109392,0.104362760548304,-0.0740740902879317,-0.0297877115909766,-0.0517196264503134,0.124536075279673,-0.0607849307321293,0.0497305490831701,-0.0305184579565817,-0.0718937808901479,-0.085253621603489,0.0843125962244422,-0.0725130024957765,-0.013769408780826,0.121837625769867,-0.0398302112279318,-0.0252205017553194,-0.0013134832924913,-0.0704477856083598,-0.0114340884524958,0.0438196882834647,-0.0172129575456634,-0.0192772809938022,-0.0758962105793353,-0.058826920825442,-0.0438465194358005,0.0118814812024022,0.011500590516409,-0.0747041636429103,0.0375187362495502,-0.0305267410181134,-0.0651693546854101,-0.0441123241369564,-0.0853915496951235,-0.116688692441163,0.128103354585551,-0.0338602059465742,-0.55258248432124,-0.971376306053937,-0.583469786061976,-0.224509396810198,0.00748900681604326,-0.190635278889018,0.0482497146557262,0.163276549368885,0.044139384315951,-0.573765092701159,0.0990203730481704,-0.00538937471366916,-0.0830558938398917,0.509935497504998,-0.201813836908756,-0.204770385713003,0.569824594869154,-0.087474997121847,0.0212244261396285,0.044374055665136,-2.71708529754692,0.0492865992038653,-0.0731329558401409,-0.457673578505167,0.263385405607868,-0.114879259652129,0.248171267324693,-0.106505634699993,0.257710905488001,-0.106232262175653,0.146007007982899,0.109989971407598,-0.153929083544968,-0.0562394599266371,-0.0937692096791462,-0.341451278079807,-0.61639042475032,-0.562395457526776,-0.642880553524634,0.0121344628028072,-0.0951765949736105,0.120824477977869,0.1215677380279,-0.00017794253494121,-0.0125629478777033,-0.0439253011533552,-0.0299149132648151,-0.0245104386223415,-0.015361589172726,0.0929637848108144,-0.0415132472156061,0.0620504297226571,0.0648380100535361,-0.0587169569679013,-0.0781841712155715,-0.010877776292608,0.0867615522203612,0.0847125277344334,0.0155426838422256,0.00328047826992232,0.00543553171212011,-0.0112024773701941,0.114114713918116,-0.0381095954576617,0.00289210493355927,-0.0691512628080921,0.0730760436848396,0.0518638024980257,-0.184658241849684,-0.0670368274183009,-0.109867480588751,0.0952885336434717,-0.332584085975587,-0.103644995443716,-0.154262455472665,-0.0909152057114118,-0.0060343169342013,0.0891169854397383,-0.0810262112088868,-0.0394795103214296,-0.167464842280163,-0.0566707420599936,0.0817503056892956,0.0664348868417701,-0.0348228626562308,0.125604675922332,0.0809499287398878,-0.0013876574545307,0.0563301951699046,0.0410334135998359],
[0.00344121404891152,0.0649421303809774,0.0634851442509844,-0.054600236869869,0.0169520210212235,0.00126885543370704,-0.0361993145892963,0.0310085292892308,-0.0632592284364344,-0.0395563219601012,-0.0223666222272456,-0.0637626219994256,-0.0196111855176595,-0.0658485316538697,-0.0271854313252364,-0.0271131339153523,-0.133898874117819,-0.0666576148430795,0.00818232527982721,0.014485026390446,-0.0404763026995121,0.0767213595068204,-0.0808856557381046,0.00544921622315762,-0.0147322610614446,-0.189801072035668,0.0876588390029821,-0.14750007242782,-0.0590792039075143,-0.104587213646473,-0.0206248887747158,0.0533026035060546,-0.0413541431155182,-0.0234780600192312,-0.0461409284889853,-0.0301964134456588,0.038179143159976,-0.00459834860965647,-0.132225352346129,-0.152846891230439,-0.117735303862905,-0.0492958793477726,0.0529859003078646,-0.117675865698321,-0.0235690421155248,0.092018600364802,-0.0190634260483033,-0.0414614692739247,0.0146655240507906,-0.0953745273423143,-0.029696021739509,-0.0828223485161007,-0.00392653852186295,0.0261467223939412,0.00135347602966835,-0.0811218419731008,0.0768621303447442,-0.100151873701457,-0.0361276484619565,0.0422864754939508,-0.0685855660056036,-0.0121671446739165,-0.0552137000797402,-0.11205372496637,-0.00156908960529168,0.0055389299318317,-0.0728357104841191,-0.101272284320453,-0.178042703412364,-0.0239569378789585,0.038676700293045,-0.180730012705541,-0.0543429250679336,0.0343990878244856,-0.0777941066472793,-0.150052616733137,-0.133646669880101,0.0163303989268521,-0.0463284192306221,0.0277378578205556,-0.0763308596203307,-0.0194270760419212,0.000469357032902662,-0.0624034900103908,-0.0761496010471912,-0.222623932447438,-0.00679316002539516,-0.167060476073318,0.00753665679792068,-0.0671058520873948,0.00233198234658966,-0.114390084871747,-0.0511151123816443,-0.0542652773741826,-0.0178254689174617,-0.0299197500974494,0.0248752113807206,-0.00860138054716461,0.0209574243673607,0.0456055593407307,0.0744052922339909,-0.019318939332072,0.0540457005967585,0.0504614083399116,-0.0557822324706971,0.0308951604012915,-0.0647692107403447,0.0425694794808096,0.0222756458973164,0.10514581299721,-0.00859660186999227,-0.0877497393951765,-0.0322984686788912,0.0958023168882688,0.0199187036974664,-0.062115398773158,0.0186666004525243,0.0285685131980446,-0.0604658806280062,-0.00876226063317074,0.101544252278163,-0.0418133907377361,0.0273356094872056,-0.0301145879935212,0.00271026723560647,-0.10865179243625,-0.0578339768645448,0.0822627282866022,0.0598511510566362,0.16362894546467,-0.0260626934057306,-0.0643061429921178,0.0409585003456258,0.107073266037372,0.0303961142174928,-0.0188334899598107,0.0272743428158371,0.0432962759585915,-0.0127428095155539,-0.0743261596683291,-0.117905805722311,-0.0327952112925017,0.0324724056575073,0.0982482240075219,0.0368253460735497,-0.0592661249967625,-0.00567506124891716,-0.0396107958460101,-0.009557998030407,-0.0214952249275714,-0.0721765345680116,-0.0831975849378399,0.0471547834348925,0.0481930486797854,-0.0112507024519014,0.021304596122986,0.0481742987588805,-0.0523716926227582,0.0404758531331932,-0.00187603969154341,0.0422605816920599,-0.136678351307724,-0.0535746950361061,-0.0749851518117144,0.0280933496488348,-0.0418713738734156,-0.0044285437951903,-0.0476871461779572,0.0429670799352708,-0.0519559302511112,0.0310975037888267,-0.0901931911890766,-0.0311815229206581,-0.0471627776277704,-0.0358599946697882,-0.0123179092963374,-0.00497980308494348,0.0323849779432206,-0.0464516147203809,0.015005990749645,-0.0917156595520229,-0.00172910179714402,-0.0456793156434702,0.0123028880548949,0.0304521453965803,-0.0270325212897053,-0.0500590627220453,0.0379053935671495,0.00108714728255851,0.0302590783248595,0.00509943166359193,0.0377755395736477,0.0043407135666621,-0.0521698092108508,0.0252306978647164,0.0321504469127907,0.00653495218374852,0.0216850725306777,0.0122015006728676,-0.0559665068519691,-0.254931971666953,-0.50896167737405,-0.225210621618841,-0.0929420023982815,0.00901156590293682,-0.124775188156978,-0.00900759523590104,0.0510447900255944,0.0558363105915948,-0.245756852314362,-0.0385844857662433,-0.0551187184379259,-0.0434458121812034,0.199235796249647,-0.0533794489463099,-0.0299833389707324,0.147211231594927,-0.0714709944405508,-0.0567563759604107,0.0561881746801944,-1.40910851154145,0.0213395540088148,-0.037994948211279,-0.202379185504869,0.11376389890282,-0.06207428480388,0.0607924219604305,-0.0424774091647157,0.163114880135493,-0.0595399215818489,0.0568542731274281,0.0456761128335313,-0.137507795413495,-0.0839874936816663,-0.0760531708368263,-0.145121558008034,-0.216725346619455,-0.221170253695964,-0.305259861357903,-0.199449329344418,-0.107588180644834,-0.0403112235415221,0.00588873942412721,-0.00613559889888679,-0.0150780417924506,0.00710252921972267,-0.0241212572091464,0.0107687894610255,-0.053608481231274,0.0396267841055291,-0.0244787297998124,-0.0343885891540964,0.0467383341273159,-0.0257031855900407,-0.0118667211182636,-0.0432976854510968,0.0337566539403097,-0.104448229360692,0.0543057059351868,0.0683754088141268,0.00127638194071938,0.0245658136018901,0.00767556074768592,0.0185281063548126,0.0432078300289453,-0.0561514028286601,0.00291604711237001,0.0229927041806324,-0.0408235849309219,-0.0548224943492171,-0.032292975387557,0.0587271853444145,-0.00749181795896379,-0.0196551499761162,-0.0427931995058899,-0.0121584181906537,-0.0202338634205992,0.0685283368645491,-0.0557523586948699,-0.0245411624616773,-0.114526717758221,0.0333582993488172,0.028585603439369,-0.0394088902382658,-0.0642978945738251,-0.00116766804543964,0.0661787286448771,-0.13160550098136,-0.0124735561697199,-0.0837312387569332],
[-0.0310534281789131,0.0269785338955802,0.0602671061048831,-0.0285121036129562,-0.00731852818186188,-0.019958411215244,-0.0816321233780267,0.0158714290670125,-0.0258780891733015,-0.0232414746075896,0.0483808723999655,-0.0575321821491396,0.00420233013221341,-0.0482067125721598,-0.0241899136614409,-0.035970588391446,-0.0815886487462995,-0.0614960872551611,0.0345449040633611,-0.0313274056075538,-4.62351842794878E-05,0.0749508058018324,-0.0738727264815059,0.00591130554868219,0.0206811740656961,-0.107032008308177,0.0557417455431468,-0.123567131858226,-0.0854618839516344,-0.152635964480525,-0.00608945711187967,-0.002028801630724,0.010826049287007,-0.0227427720921919,-0.00470243953408414,-0.0125083919246315,0.0793438491781827,0.0201256496535902,-0.112843764457087,-0.125262104290966,-0.145970982128128,0.00838574963391941,0.0340911293651236,-0.112591525491913,-0.0409724444659502,0.0807274399272864,-0.0738062226513031,-0.00820016266133829,-0.0122122346404724,-0.088127055324518,-0.0621166509685896,-0.0385961717865252,-0.00895752884999839,-0.0391183727329825,0.0355662804012175,-0.0225807746538094,0.0492435947697542,-0.103156952926171,-0.0180004380821681,0.0293030450468633,-0.13394991663194,-0.0359389707218452,-0.0575971704771639,-0.0836395454275845,-0.00387991334459545,0.00854184995910871,-0.120602798107472,-0.0563591200357228,-0.16648774108649,-0.0531696160197377,-0.0217895197175011,-0.150530073595031,-0.0626257605579257,-0.055867016684267,-0.0531288905071023,-0.178972382342469,-0.11058232064851,-0.0146503327138832,-0.0281732037791082,0.0257390848415907,-0.0525159948135987,-0.0577718794672333,-0.0559732122612423,-0.0862450829370056,-0.0922323311431194,-0.239807339190336,-0.0300736841127206,-0.143593399430563,-0.0139737752477168,-0.0659474446645086,0.0293503653659593,-0.0564673114080632,-0.0317155442303287,-0.0505506843342583,-0.0405923521285963,-0.00966759710409306,-0.0257556288887558,-0.0166841595424443,0.0273810377842754,0.088035864429188,0.0218793745248563,-0.0305936896850458,0.0690699947322634,0.026946098522268,-0.0171017662021896,-0.0415179661026061,-0.0399457636457384,-0.0224869627881689,-0.000787718355952656,0.0454658784312022,-0.0134820041197288,-0.0794629489926949,0.014144568366402,0.0695773526629574,-0.0153043007050468,-0.0212699127644798,0.0392815643343446,-0.0696841121733198,-0.0524239139741276,0.00241951287265103,0.0596819799191133,-0.00656134694806212,-0.0188979409967926,-0.0238699845559459,-0.0222565747711251,-0.114219189667229,-0.0601044527429373,0.0397755712773951,0.00475286437312937,0.18254565016869,0.008063859108692,-0.013914242675559,0.0252182146179768,0.0760437130134991,-0.0118750707519219,0.0250168677318364,-0.0324013056891592,0.0745874525507326,0.00104482309348199,-0.0279372854055081,-0.0600566935304067,0.0195868206458002,-0.0561531381080456,-0.00634210915238631,-0.00580577313001954,-0.0277652181765374,-0.0265693329803987,-0.00461452173687038,-0.0721727929984808,0.0199702315729358,-0.0659737898665494,-0.0548387580764259,-0.0019838848778155,0.0667752312328455,-0.0196328506139326,0.0173697767490698,-0.013133295483395,-0.11040021836882,-0.0416892560028547,-0.0389354571601828,-0.0177021787488076,-0.0675300812306102,-0.0456426643949722,0.00290652745929719,0.0388369587965079,-0.0745823620703194,0.00432122926460778,0.0190716007785428,0.0623903894170632,-0.0269891025246117,0.0148606678654085,-0.123224296784696,-0.0335082272679543,-0.055712325160741,0.0591686163301821,0.0357975055291991,0.0179592808445462,-0.026894596158033,-0.0379349955519164,-0.0500195991361705,-0.0964471456596585,-0.00974520987226605,-0.00581117261888175,0.0207102360920856,-0.044801661237903,-0.0372656391322549,-0.0487556501611246,-0.0437418903484647,0.00271234537058953,0.00200395457850215,0.0116252171862962,-0.0388814611831617,0.0376598441457837,0.0317132756928802,0.0230691237473124,0.0310774344307033,0.0190131439678782,-0.0433800154306655,0.0092703093769053,-0.00916981612414788,-0.193113926541012,-0.404128045419984,-0.193856124248946,-0.0551175663448308,-0.0311711723974156,-0.0748583196255394,0.0377824883678969,0.0153910047417294,0.0485013073844167,-0.257800115557275,0.0389933094087915,0.0242837161252259,-0.067382486379972,0.200311913961058,-0.0906018576272247,-0.0712490288056474,0.113904579834302,0.0177449080469274,-0.00160416056264389,-0.00730365961502525,-1.12485757972216,-0.0230719218571146,-0.0850166770224365,-0.235659776368255,0.10071825360876,-0.0850808336402328,0.0608845157496393,-0.0243608477353302,0.110024290333691,0.0244081861447873,-0.0171952487884117,0.0207224694191808,-0.059448279896398,-0.00169383218624994,-0.0558235551437139,-0.158208932555977,-0.185683146293548,-0.168398402534217,-0.214544411255524,-0.16839619790624,-0.0416392609333191,-0.033233271336109,0.00468703898896696,-0.0553231984127331,-0.0167488887957193,0.0447540645419014,-0.0366200825667611,0.0644994194029517,-0.0220809695576216,-0.0022571819157046,-0.0279318703796279,0.00325783900398718,0.0437878997432884,-0.0669284090262309,-0.0201006995347804,-0.0478136324074431,0.0131792219795403,-0.0509100976940394,0.0595433215070122,0.047944717654351,0.0157796932600833,-0.0295766755118759,0.0194329407242986,0.0415248532669757,-0.0375425307653243,0.00121151838959921,0.00286233944763505,0.0679207008504912,-0.0451981691813271,0.0447751152564723,0.00662780102091821,0.0370441260924382,-0.103683153709145,-0.0637878885073992,-0.0242575975433829,-0.0180910937026291,-0.0498117320346924,-0.0380584987805279,-0.02542310715704,0.0400730974388936,-0.0547655721707593,-0.0146135174775823,0.00739059374132766,-0.0174773299760726,-0.0821238095556484,-0.0121184693755817,0.0457496033820543,-0.0242444890738842,-0.00797497647572718,0.0117004838974549],
[-0.0441310762867023,0.0754202433911484,0.10971933349702,0.281131729475279,-0.110850161874986,-0.168150393355632,-0.114412949469207,0.0323796586367097,0.239961325605069,-0.00867884448541132,-0.0538885651386345,-0.0592861060075996,-0.0069080238772638,-0.0265095790111266,-0.132358811646013,0.103177675520385,-0.00573721758883757,-0.021803127200301,-0.0736098274511307,0.0407903656343455,-0.036816799160397,0.143029904009149,-0.381254654956537,0.114913979116637,0.0288964070069876,-0.105146761288698,0.363047684736151,0.0135528637516232,-0.147145512975644,-0.478966144265186,-0.0102733899915169,0.0914295774897115,-0.0712557702340018,0.023439210304213,-0.1546980013832,-0.00162896329852817,-0.0405808413164937,0.0862362068980025,0.036887971077648,-0.107863976276002,-0.514281104453814,-0.143400833746423,0.365429008179209,0.125675092970721,-0.0202851239445314,0.0155415466098462,-0.130983575283908,0.127664254727384,-0.0357759245141634,0.107453313181035,0.0327782280686229,-0.118509463751421,-0.0296665416377095,0.0291295288288895,0.104201082184393,0.0442541093551559,0.121338154132875,-0.133208402001325,-0.0107640928924593,0.10014890380225,-0.288040346275847,0.0574354902002704,0.10479391572725,0.164445148449168,-0.0065154450439195,0.0310772455180672,-0.239940517853827,-0.202640598896158,0.0177591418696351,0.277739941569614,0.0288795176641422,-0.233429767728509,0.0341004402576216,0.291461729568281,-0.0241788907619811,-0.120414989455242,-0.000269851629553681,-0.134370905609191,-0.0706796961789636,-0.168069917755462,0.0454888423732883,-0.151649045346135,-0.0266223852290376,-0.186733432053112,-0.0809292959845109,-0.172887773477979,0.0554574386208548,-0.175072404655821,0.0573693369973499,-0.144259181391368,0.0615061593165997,-0.226377113565543,0.0725952754746633,-0.156864084695347,-0.0874147771498469,-0.00917186586222206,0.00847691663130863,-0.214671339480371,-0.0668909274324596,0.0212976584178758,0.0468159976361458,-0.281535946193784,0.115660318820535,0.0160420783370205,-0.03495045091238,-0.123183412084697,-0.059710874993861,-0.142540650637944,-0.0284633951929231,-0.257889375882681,-0.0172638782724575,-0.0957553896881367,0.0283589217461219,0.0728856114599031,-0.0723123991120101,-0.0442887753780478,-0.0399859053135197,-0.0881179192878692,-0.0538911356309285,-0.0180739033535391,0.0902851879168066,-0.137687522318441,-0.00452363308900894,-0.161830559882588,0.0314531212968917,-0.269542916531106,-0.131922629695187,-0.0394392465735374,-0.0151643497737444,0.409273235638762,0.0940778254269408,-0.0564500198896911,-0.0809452800800152,-0.0211770367786043,0.130898171590062,-0.0243557349618923,-0.0398999989148531,-9.9408701153866E-05,-0.00460854730863094,-0.0654671088987549,-0.179204837677952,-0.092309290035346,0.0752017140906292,0.143310606738059,0.430737067345064,-0.0974503000757289,0.0377199404144414,0.157283839058355,-0.0622291980354289,-0.00283281665596754,-0.25778548510345,-0.0047438712693353,-0.0154320007346153,-0.172786166258825,0.000830078781856306,0.11935767161836,0.0417244708411469,-0.199034471387369,-0.0301918179595662,-0.194435783270218,0.0690132436882241,-0.327254590928783,-0.248237307744576,-0.140881065700016,0.0451710454058669,-0.136827524317996,-0.0153562872072598,-0.0487995209808691,0.026866669955332,-0.0288820799972499,0.037774148111341,-0.102197034035757,-0.12108202522079,-0.169117944639477,0.0887475443264328,-0.0255496987026169,-0.0693281944017781,0.0573273558984038,-0.101851699009029,-0.0188516781617073,0.0590549764884628,-0.0708643981762764,-0.05021069961012,0.138778347309797,-0.0353965770550104,-0.114986683174009,0.0293687982874022,-0.0149276897117801,-0.00397328166794164,-0.105452565700238,-0.00228925324740315,-0.25604418131312,-0.0113196555303695,-0.084200606760996,-0.069255360176716,-0.104516847065978,-0.0943444091452638,-0.161105252955335,0.10246583178137,-0.011975979723554,-0.478480874148217,-0.698812511177759,-0.416728987454209,-0.188551061496504,0.00635752420687471,-0.191461248441108,-0.0779101700877381,0.121884659075021,0.0624478767102264,-0.487868140264563,0.0141779029271993,-0.0367979748787197,-0.358068323217332,0.0918731918098266,-0.131369145890673,-0.174286495780095,-0.261040195970328,-0.197737801960702,-0.0339936948872512,-0.00927451853645163,-3.5100491304434,0.0667800195275155,0.198422375011782,-0.269671745132614,0.365131020027484,-0.171189990068461,0.175225118136678,-0.128517961423508,0.120815231423372,-0.307482227570542,-0.0977171271785774,-0.0934885612666838,-0.434207249136676,-0.215429450701434,-0.241107728903173,-0.576690035920974,-0.873273273294941,-0.775270883616223,-1.02224600512938,-0.552741331823132,-0.0297787881516057,-0.115766230068709,0.0862956336926228,0.0639471655850372,0.0135931017921613,0.0633968618707546,0.0434646318362189,-0.0231725455249159,-0.0252463853681132,0.00351563683443571,-0.123554552737605,0.0827286838768177,-0.111596985770853,0.0730956302214038,0.0395045838368197,-0.00157148617210995,-0.00619812087960195,0.0321855447384635,0.0246055377580475,-0.0653908669865694,0.0524208971722029,0.0721416144176842,0.0453840603246866,0.0292404752868656,-0.00337255580604678,0.0489191174083767,0.192823935817619,0.190409084414176,-0.171398194826346,-0.0113921422815763,0.0969313064758967,0.021327088656176,0.162234743127485,0.0291950963307534,-0.138011292407502,-0.0757288484246962,-0.0498581480035262,-0.0677070265094295,-0.0198025961301439,-0.0147394589911437,-0.191475459044332,0.0539122944088097,0.192542861333767,-0.132305097330703,-0.0345170343586229,-0.118126466028835,0.0756428596090939,-0.231438927417268,0.0719420146707399,-0.160355628000103],
[-0.0205068987071991,0.0623443060747729,0.0583336838450058,0.179356144323728,-0.0376560399532653,-0.139507636515114,-0.1093270133703,-0.000588753050417852,0.152094429865563,-0.096194074935856,0.0460046820748403,-0.115426570605379,0.0662599435600483,-0.0288361170138833,-0.172663089390314,0.0650896514359733,-0.0589285448854462,0.00399328945602122,-0.0210388192944963,-0.0391841618175968,-0.0452678217245321,0.104655917641169,-0.346502464736458,-0.0502113442606744,0.0871994789887202,-0.0551936366871209,0.156530025266168,-0.309745076257561,-0.112395314909814,-0.420623022186825,-0.013155632366811,-0.0305849251792468,-0.000400458454981748,-0.0455545847023861,-0.194715073419106,-0.0394437422481441,-0.0255723580531309,-0.0280153815104181,0.0546216361631418,-0.0908151211352954,-0.616036144005655,-0.0567709071587136,0.217849971286634,0.0285224048197267,-0.046601697287282,-0.0445900971912085,-0.0611090906601355,0.0848373646644426,-0.0052101964834167,0.051324758888966,0.00526020925926108,-0.0347774038903138,-0.100598798002056,-0.00977333475690394,0.0865941465540764,0.0282504445745607,0.106564360244099,-0.183345199314499,0.0721109419147889,0.0292735016574918,-0.258316749202303,0.0793768783199985,-0.0271101734238078,0.0878969541116193,0.0516546547217092,0.019999842808013,-0.225694943774585,-0.17750468281734,-0.0903653076797241,0.0485358873755249,-0.0206829984242949,-0.262685164291947,0.00355476676912894,0.177206388942006,-0.025092278605065,-0.129006717399363,0.0269155793016493,-0.0883887267867197,-0.0529843177871761,-0.129086872148469,-0.0270791409076614,-0.142143001187918,-0.0997445861186302,-0.161290034403976,-0.10903058888371,-0.189021954544014,0.0213332448504414,-0.192402250400079,0.0585856702960072,-0.122197156810393,0.0620093683671113,-0.170828221952341,0.0207848789686926,-0.0764374622415404,-0.0679895776649235,-0.0529734158057831,-0.0154968687922106,-0.239516193007687,-0.0398427757311967,-0.0109122853284017,0.02409674523146,-0.313602854900597,0.162041714560735,0.0110691141095421,-0.0420571079046313,-0.0903478765128576,-0.0239631236788358,-0.103781941123108,-0.0142434630127908,-0.172406437838337,-0.0321892584538154,-0.101910340749439,0.0334240186465248,-0.010415845852144,-0.101185292429414,-0.0867388886570623,0.0296108514749463,-0.0733859524858144,-0.0455118473134827,-0.0161933383422379,0.162725902634879,-0.147715890226449,0.030091279291261,-0.0835027646836875,-0.0182706896273503,-0.16328380377258,-0.0987071187974587,-0.152071677943271,0.010024811354642,0.308917197132829,0.106797900062213,-0.0391560215143267,-0.109376302709492,0.0779189427887722,0.062658895388125,0.01185345932543,0.0190597445312594,-0.105617731937526,-0.0501179309505564,0.0138655184145174,-0.181334331061078,0.0278130153739924,0.0113630303175519,0.131556046382745,0.0273492761737367,-0.0323470648394326,-0.0582252485618344,0.0235278543506487,0.0630827634545839,-0.0209262108991271,-0.215907040897532,-0.002506376406748,0.0268021919298973,-0.114819430383963,0.0371237050835847,0.0800649304702819,-0.0546626951075063,-0.157411571008764,-0.0310573140028076,-0.118024381219728,0.00047870719311755,-0.213786661802479,-0.251076078231958,-0.0822263775354486,0.00451466969048689,-0.120848750814983,-0.0432604350867233,-0.0074245608201267,0.0715926225689256,-0.0309081370830656,-0.0513555967659311,-0.0767268105283805,-0.0916887857437308,-0.0896397028950846,0.0737977218866552,0.00648360834870031,-0.0579808429700295,0.0419984112933791,-0.0327672490360038,-0.0629845819027995,0.047738744397535,-0.0795563137876549,-0.0301400867307417,-0.0549143283375364,-0.0217287135711126,-0.0867819743164722,-0.0289822186781495,-0.0047846621377732,-0.0352807075708571,-0.0137669762056275,-0.0420486489066969,-0.153937176659844,0.0107713157363001,-0.0358026375721968,-0.0170587947352042,-0.0660084778655502,0.00989225359242899,-0.0708716611786426,0.0818545819386355,-0.0136033434922174,-0.446855002795981,-0.698912385678493,-0.404225622010905,-0.101569845857796,-0.0373846164851958,-0.130995914350784,-0.062805964417256,0.0878806211871455,0.0287876238028207,-0.420792579845293,0.0545805619097955,-0.0375261436076094,-0.303851109689199,0.0853210952092818,-0.154425112720934,-0.162026536509039,-0.160584400734592,-0.168973532172835,0.0141731222950837,-0.00406926816367478,-2.79873203562392,0.185111389009502,0.0925779218446622,-0.265876456480962,0.289240138452868,-0.127642309167822,0.133641522759384,-0.182872739721292,0.141065917943067,-0.148081347087246,-0.147049427787942,-0.111308178379729,-0.352742897262818,-0.174579577686828,-0.242612798674814,-0.441063178832772,-0.705171511030164,-0.685943521174318,-0.851915486902725,-0.297415222186666,-0.0468850043512316,-0.0738003177005066,0.0281848265700178,0.034033949193911,0.0153398102629657,0.025718390007319,0.0455409568539612,0.0376797856893025,0.0385162526017646,0.0803919867412724,-0.0475505875108345,0.0783207565422139,0.0359598284971882,0.00869490443774878,0.0892402633178938,0.0293786588754125,0.0545879604300879,0.0150124023635024,-0.0473645880138171,-0.0405924760688706,-0.0121450202555703,0.089580113952359,0.0728749996954011,-0.0344290872940871,0.00487366185417013,0.0585626360994375,0.0850675440197944,0.0895543448023561,-0.141288107372974,0.0138749766699378,0.0793130768026271,0.0479692776516123,0.141848393852278,0.0637045043940738,-0.126201516351367,-0.0214799271666015,0.0425035301727747,-0.04452551314758,0.0134755973412241,-0.0398588878261566,-0.146348635820493,0.0779021179891142,0.182274759515058,-0.0487564411269594,0.0385042659425609,-0.132397313791923,0.0981617262883394,-0.213491972554571,0.00457650035748643,-0.17149152433269],
[-0.0129348333637913,-0.0559502620519144,-0.0450728469745996,0.528875045767641,-0.182362236063393,-0.436064121830728,0.0790798522415315,0.00727029816929731,0.531517777247674,0.098656017373682,0.0305798936386864,0.0361107817070514,0.0295302820315713,0.0327361104495242,0.0937407108983399,0.170588975589234,0.137133435879805,0.0737259198802023,-0.278663342782143,-0.0391815359868707,0.118222782185752,0.35183063317013,-0.153247350377999,-0.0237191782231714,-0.0217755186723562,-0.136441195394536,0.0920204937832131,0.594902192118854,0.210222216577626,-0.704149012867657,-0.0370611972879528,-0.118962600680925,-0.115448557984518,-0.124670609050481,0.197543513383075,0.163063310689083,-0.0233549023084727,-0.0254165271990798,-0.0615857035727229,-0.230245339035678,-7.41478330971343,0.0149015499949437,0.0521990405748067,0.77533591664355,-0.348991465313624,-0.23094842134011,0.0299197363754103,-0.124235212953966,-0.181735652771831,0.142181829500653,-0.0983745492424773,0.0702810901175287,0.302793170529012,-0.0291070090647589,0.0482744889677634,0.0340943651146607,0.014225736003023,-0.0203032537850601,0.0583822407663163,-0.0086634372414464,0.0704260432246934,0.026223811871683,0.130112343129266,-0.184153656049323,0.154723695600519,0.101072342191293,0.287209287789195,-0.202964753991744,0.0917434772015592,0.152709307647308,0.0329085084451075,-0.0673518104458307,-0.30745493948517,-0.214969866694631,0.336365763697207,-0.125687658118058,0.410349645353432,0.161056247863482,0.0143162194881172,0.0212191099149343,-0.0691164215062204,0.142653588911688,-0.0755710848806809,-0.0372288693502237,-0.203188631882863,-0.270838189613477,-0.138490405765878,-0.304361999001835,-0.00825993921773098,-0.0611885324851179,0.102014878330421,-0.297618617072474,0.000340824729026744,-0.121983188757839,-0.143139067040887,-0.118492757960817,-0.0577561372623528,-0.20691256771637,0.00976769919845129,0.28632231905387,-0.122653474395465,0.1119790387763,0.0297817801348169,-0.135787607046622,0.0356379844626103,0.136457684908823,-0.0407033819519691,0.140693207586236,0.0895205667738733,0.140515444162503,-0.0871876609982305,-0.13270297382153,0.0563387799042716,0.0856191519626349,-0.00245920526941426,-0.224319433408808,-0.00701175680997292,0.0524207623927143,0.0715107789592509,-0.126215864100459,0.106031218622482,0.0346716179521072,0.0279769341973532,-0.0239449087562079,0.0120019882560724,-0.0463275418788633,-0.168909236164344,-0.189600122721138,0.350523130485613,0.305098119930792,0.156438378872823,-0.125150727483723,0.0111581736368881,0.555783767684094,-0.0386940169745064,-0.153040240893302,-0.0443669705454011,-0.162147591734031,-0.322680091319075,-0.0746508431413776,0.0517227371660771,-0.54157607565918,0.170340896599885,0.111092002429383,0.369794558963313,-0.0620437909717605,0.013915010940723,0.00396837375835117,-0.0726579410702871,0.0310844725852641,-0.143630178121466,-0.159036378636045,0.0525273926306385,0.100283964956621,-0.0230349819879526,0.217680306786378,-0.00167679768956112,0.0908489327636887,0.0459071267511721,0.0802838806168594,-0.02110851205944,-0.0675667190372162,-0.15285153710088,0.147917658634859,0.071521251521826,0.0130229031180045,-0.0777014702119381,0.124129000021791,-0.0404218280691342,0.0761413458295794,0.0911018257268384,-0.00447445401707169,-0.0264373775220686,-0.135954424323453,-0.076423624183774,0.0190868324517248,0.0244626781498167,0.0744796777846573,-0.00355217353139127,0.0205562742265841,-0.0948437099316635,-0.0859976619523724,-0.0412244152407761,-0.0871401274154218,0.0262970746163787,0.0219380352700806,-0.0204886813907003,0.111998039748074,0.0919490545678186,-0.0164368633770536,0.0945844739280437,0.130805712753845,-0.047667134839694,-0.0143664776948525,-0.0898472731906404,-0.0570092906519024,0.122825116954584,-0.00499048577864668,0.0253724339483778,0.161492190836527,-0.474647108027985,-0.856753420286365,-0.473390677103056,-0.220267243000257,-0.0413999818016813,-0.245266412773747,0.0580054012365516,0.269327832072849,-0.0103828879576853,-0.571905311806734,-0.0547727235670813,0.0788013679129918,-0.40480561332217,-0.213177448471314,-0.137971378038224,-0.00951153468274085,-0.683245977273528,-0.195774735989014,-0.0414551916235715,0.00898200174640481,-2.51246302504107,0.233361988634161,-0.018541390686375,-0.159526317466429,0.11441654145582,-0.254544205082165,0.389648975476452,-0.288636588693407,0.503300204575,-0.121341083907797,0.162216031666527,0.295796044906207,-0.196915677389459,0.0402477052808818,0.0348544719214264,-0.154557525266682,0.422086975434006,0.543104664211512,0.516368117313382,-0.570531074358856,0.083110602193133,0.074603104653598,-0.240785684039012,-0.0487717509261898,0.0990653372588044,0.0667958229573117,-0.0272150298986601,-0.107030799649451,0.122528668406703,-0.0452619662795809,-0.107506188438107,-0.0898805634419967,-0.0380922515023668,0.106701927733419,0.00686409845589338,0.0716237905315077,-0.0655106914634844,0.114291548268691,0.018127146979578,0.0307013517997485,0.0957803779470952,-0.0175063316240818,0.0981056379695033,0.0123338995821254,-0.00654928774504234,0.0183243483191806,0.296040263806912,0.190404175387368,-0.177643806451743,-0.0237160965746032,0.0470390946968508,-0.0578278949664696,0.162952305810601,-0.18568132259569,-0.174978968845888,0.185748027202406,-0.0514290143433396,0.0171938698265782,0.132837903873894,-0.1798593788987,0.0475632640434106,0.389675700906897,0.0504776225286581,-0.00961046787734952,-0.253764229478972,0.298517881917529,-0.158417346045817,-0.341492949745406,-0.290427648046491,0.00651022799236575],
[-0.0340988626616317,0.0843902078918344,0.12388753862098,0.231908286447787,-0.0433241769797188,-0.224300187675298,0.0714257401256893,-0.00696234141360721,0.198187155040857,0.0277989920049021,0.00326948102136202,-0.130041877060613,-0.0101427350144515,-0.0127610213797979,-0.1418850547503,0.0872292876026559,-0.100436761572857,-0.042162470792404,-0.0216188066444476,0.0444783264259364,0.0172717997090007,0.1748735679899,-0.28319585862503,0.087267689334602,0.116842083164323,-0.0607760160317597,0.296447497357017,-0.0672375027855407,-0.196365660924195,-0.513358511047469,-0.0337788722274723,-0.0863538257209673,-0.0282082848545159,-0.064932206981345,-0.164388828447524,-0.0723280502623385,0.0429235623878592,0.0785149424822069,0.0405053134601788,-0.160986220404683,-0.904298964764686,-0.00684823767287584,0.261981345618477,-0.000348187683821277,-0.120805301534974,-0.0166761005407053,-0.15607448939232,0.0963913001174546,-0.0293353023551402,0.122670080615964,-0.0618704376808963,-0.135408275478069,-0.0107373393442546,-0.0451596641990601,0.00471881706458855,0.00696672008076687,0.0564559534726865,-0.139105424080554,-0.00200106221994426,0.0457231653031601,-0.196334238918948,-0.0226161405699883,0.0360817178321897,0.0706173055811053,0.130725581507186,0.01347437114733,-0.287110168567777,-0.151635942856118,-0.0282488335835057,0.108083502849634,0.0277003151229064,-0.302644410361279,-0.0537680243787171,0.17068805465155,-0.0150733252349318,-0.118364117040927,0.0545830552079912,-0.0800797823866906,-0.029988021022452,-0.052057319125166,-0.0174483240609128,-0.047904154450438,-0.0740451602672103,-0.201670567339845,-0.174746295103857,-0.291501427130972,0.0153481495335816,-0.247787214011317,-0.0412339194828486,-0.101864585262142,0.0732254931723155,-0.204903125718148,0.0224866479579001,-0.147203803044935,-0.0168320983003087,-0.0546981047071191,-0.0443297065205635,-0.168755837130347,-0.0115546361582669,0.0908315286886101,-0.000681457303326907,-0.238414039881652,0.07437818227177,0.0623312759539629,-0.0923698245959371,-0.0901780235416463,-0.00461684288584543,-0.10913109847248,-0.0494223540644449,-0.0863984964583751,0.0140985506833026,-0.0553228836432453,-0.0383944807558033,0.117863796142211,-0.0521324698887377,-0.0641207912761755,0.0344390142039668,-0.0719293963748982,-0.0541798250667212,0.0564741978933216,0.144580062344879,-0.0977451091791448,0.0305777636137346,-0.103185155829238,-0.0660788071183483,-0.307150248615741,-0.181037421262436,0.0220832506680923,0.0249291818435419,0.392509336148854,0.075019556896185,-0.125957709734658,-0.0123457729737104,0.0982167464694548,0.0690586287165361,0.00284813999393398,-0.0084681676137653,-0.0184491613341489,-0.0224874374160629,-0.0761394523765594,-0.0959858439433887,-0.0126842905465949,0.0660350231671439,0.111068567781438,0.354180107329259,-0.0822538486350573,0.0547356251285868,0.0952694880319268,-0.145739683076066,-0.0149925936404593,-0.258180136242799,-0.190557090913644,-0.254977390513316,0.0328000418834021,0.0346227746688363,0.0497281302478167,-0.165066982534261,-0.223588042649903,-0.0788394034141938,-0.163504466044439,-0.0636633354096487,-0.248105834641214,-0.212559828525819,-0.021569042683469,0.137618479655721,-0.146686723296329,-0.0963842906995396,-0.0205238470108669,0.127460023062801,-0.0663918615772632,0.0786235426201311,-0.0528851834286414,-0.0761706599682822,-0.112176761546325,-0.0495090592561622,-0.0392464359380945,0.0178939051931455,0.108404927390678,-0.0128897927775763,-0.00453458487656972,0.059833519648165,-0.0253774821551533,-0.04464762778473,0.153352429692528,-0.00187344361408531,-0.0633051369663711,0.065626354736244,0.018952956858571,0.00939004146922487,-0.0502235996105,-0.0290849656516736,-0.203561707488101,-0.0106858805665299,-0.0756763024969569,-0.0311506629208509,-0.119397016040358,-0.0491970052016639,-0.128834501447188,0.150557990247462,0.0520619575242751,-0.491857021313852,-0.784199795664354,-0.421671444831314,-0.140914059388936,-0.0306638567459977,-0.186164887934708,-0.0386378169467164,0.0501539319938987,0.055189768309609,-0.524745504836726,-0.0827844086010134,-0.0874682347276631,-0.366383077492389,0.0565035003048236,-0.159149857025268,-0.119763363008542,-0.240599279508551,-0.076698632989178,-0.0568309285545715,0.0151095406476196,-3.28344684630081,0.122135645986861,0.16338327356045,-0.307045935110894,0.35756082880844,-0.130378009325482,0.191526764306303,-0.138141004972391,0.163042932040581,-0.187817765694593,0.0119025791793106,-0.0544969544135441,-0.356403679292111,-0.21913133682567,-0.164532178809354,-0.492577201794396,-0.709673311569291,-0.692379422966612,-0.898716754297357,-0.472072747916392,0.0327683799270478,-0.13816997353524,-0.042774523346134,-0.000716382049264786,0.0248649617833031,0.0440341621092071,0.127419231129073,0.0295772789637023,0.00354360545570169,0.0631818024713337,-0.0641007872729368,-0.130190464678479,-0.0230298507935277,0.0509084378478205,0.122885530856894,0.079114143746921,0.0059047491012975,0.0109646304279606,0.00954603761560228,-0.017715191166385,-0.0306563060749851,0.164514438864258,0.115032559272087,0.0211158219524495,-0.0089131514536257,0.14073228927299,0.180733649335313,0.0893322799688391,-0.103360021125899,-0.028483063178633,0.190687754844323,0.0766449199822669,0.18353075767879,0.0544783996339522,-0.049219863287412,0.00725073850170013,0.0356457940493445,-0.0254776003857091,0.0561588317162631,-0.0122439624058799,-0.0742196280782121,0.0755148047290389,0.0663872584335329,-0.0437015812214114,-0.109170762701221,-0.00102015323123835,0.0631690122168564,-0.14906359559398,-0.0270185217911051,0.00395594283688698],
[-0.0282555414320905,-0.0541087695335413,0.145719022877827,-0.0181631252694577,0.0482125003040354,0.163866964087614,-0.18553245616723,0.00565847175384009,-0.00890409516319707,0.0135740119721865,0.0131071522751415,-0.0283878934004697,-0.0163368578125365,-0.0410067109131243,0.204823651591723,-0.113387358441208,-0.143344606316818,-0.183828707377256,0.197533822598266,0.0154971677900867,0.0371810078560844,0.127489186219939,-0.22914058584463,0.0503422455293698,0.117671917022543,-0.14987461045805,0.069680414874761,-0.378092030221953,-0.209690690534982,-0.139252139471903,0.00600971192433297,0.156582548702742,-0.10584227537972,-0.116461817687202,-0.0709130607161111,0.0307779464051311,0.113525055321917,-0.0652011569137251,-0.12688896582002,-0.247806367732219,-0.137965132713752,-0.114418753597457,0.15634449591305,-0.0768872979578635,0.088224018261274,0.284931308811655,0.0267159895220011,0.0730673734346528,0.107726338680187,-0.0577085073744607,-0.0885460275800491,-0.129706224460074,0.0446867613871897,0.0503235620907679,0.141293164139444,-0.0157929635114667,-0.021617693838969,-0.100180360646455,-0.174545587683323,0.0249064225228181,-0.130230637842733,-0.0753084019896836,0.0166331515691305,-0.165541354567878,-0.200468762818812,0.0347930330461115,-0.111943643107298,-0.0276168671580553,-0.36658913892284,-0.0863818295093989,-0.00826655502759843,-0.280861863316932,-0.0541030386341287,-0.0798937157737859,-0.166458209391188,-0.227900029878699,-0.147977996980588,0.037186687064686,-0.0970078868829236,0.212078014504339,-0.139126842274814,0.0453307797668372,-0.13507611705036,-0.0292464326496849,-0.177451434247444,-0.22343727275153,0.154188752333729,-0.323325685657722,-0.000436591188391336,-0.0437324280064302,0.025695765544383,-0.109533264952257,-0.163863570037057,0.0029756891139547,-0.100119429351218,-0.00906135290826256,-0.0578322919221349,-0.0190674458969143,-0.0172554715722597,0.295914596730222,0.135874051032866,0.0270478376147397,0.127980068080151,0.0526455121322537,-0.0653079399607,0.0579308978306227,-0.042699690189047,0.0298123533543925,-0.0286664157125056,0.355244063774738,-0.181258835254826,-0.0796286613444094,0.0161518116622563,0.183523708251809,0.0956501966942671,-0.15191935623296,0.0516244857936732,0.0485518170498313,-0.112262109553077,0.0456007755898631,0.184303370636809,0.00463313769768296,-0.0145338710822891,-0.116416778662595,-0.100278936379002,-0.0728437691434289,0.0193838474509223,0.287523205893911,0.203640799662205,0.35212773323111,0.062261240908456,-0.0234868685609532,0.0189537349884767,0.0464942479665917,0.121951419644629,0.0567215633320178,0.117791165096554,0.272849789520373,0.206700733731465,-0.12305058698414,-0.146925397891561,0.0697325883953973,0.0203350376284368,0.143869341938269,-0.281444305846361,-0.0141376388703835,0.0734376235206667,0.0252351187192461,-0.0807409288333545,0.0506930964403024,-0.172324267172848,-0.0394875365384519,-0.0718594385890289,0.0838609184759872,0.0493489995629483,0.130261670677122,0.0299781570357641,-0.0880552366642473,-0.0577654492167481,-0.109714483233598,-0.0126481093911912,-0.122540804009152,-0.190869403768475,-0.0292644297843988,0.0483424211355997,-0.0902773681993054,0.00481137382690124,0.0385295580315957,0.0999471761390393,-0.0294579306317521,0.0279578289596147,-0.085308977894121,-0.0487970132239075,-0.042108872195935,-0.0539674367885243,0.00994595403020198,-0.026730282137791,-0.0930442348345087,-0.0819164346911541,0.0388725981947555,-0.139373657860583,-0.0468905878869289,0.00547294161743253,0.00399462538810526,0.0203968272404431,-0.0174090416351893,-0.0924059780020555,-0.0151789753808822,0.0394549956167406,0.0369465784608712,-0.0049350429532046,0.0772282708594185,0.074663818298038,0.00907054597399684,0.0695611310387441,-0.0638312211790305,0.0099543681416858,0.0130521884648218,0.0917207355474665,0.0477388384951991,-0.564621840657312,-0.972187072860476,-0.573415602476381,-0.135553906254713,0.075931540558177,-0.170944297817404,0.129110144728697,0.225363924856373,0.114201063614716,-0.605710010133982,0.0423223027902351,0.049830824410824,0.317313981793637,1.03256828683669,-0.0953973422340506,-0.0727760974451537,1.02362239774195,-0.274304335794784,-0.0144972771835737,0.027127003154705,-2.22410273901335,-0.158613918104578,-0.280878032459838,-0.363050601730095,0.167977563010025,-0.0214447257424311,0.359460048709122,-0.0544207803950259,0.326869808160895,-0.0584312850962877,0.0951958830221266,0.172940395809578,-0.153808184520197,-0.035545837320095,0.0362337521033373,-0.252885611628752,-0.282661518783074,-0.213904096777688,-0.399612171925137,-0.344073101605854,-0.0597217254302844,0.0886373445713302,-0.0872254634205288,-0.0265616196007572,-0.0306180679832516,0.0320937868026537,-0.165453239031418,0.00253204429773783,-0.107083768925431,-0.00695447599491074,-0.00177128436266417,-0.0477303914335291,0.0788639918377715,-0.0480249880620436,-0.214349033623501,-0.0504570901794646,0.0217726374089691,-0.0431393613957577,0.0783524633745673,-0.0884512477068327,-0.0079826234037252,-0.0588695300815334,0.122259306311945,0.0111189016657184,0.0294807709239931,0.0950966574540468,-0.153258463721049,0.0754416408183999,-0.109520364713579,-0.0445973346360549,0.0348012251590291,0.217724645576488,-0.215251868684077,-0.0972512351721479,-0.0402190737487836,0.0645764032347593,0.115077848852513,0.185089647316644,-0.0483795638173033,-0.00430100924813482,-0.061487167966805,0.120083913243666,0.0646636185153229,0.0306384592093936,-0.0998125838406531,-0.0144784193956015,0.0442190274342933,-0.0294975095100532,-0.0569839273229913,-0.0580038731970335],
[-0.0429453968438045,-0.0154804435908898,0.144159385925246,0.161333558808565,-0.0573170756900439,-0.151912509510395,-0.146136005967208,0.0559957371473959,0.209959191937985,-0.0757341598697975,0.0134945880752335,-0.104886010957725,-0.000849628053659304,-0.0628477252504285,-0.0564130447256239,0.101542493711372,-0.0667514408846355,-0.059593109836545,-0.078675250371849,-0.0991332871731691,-0.114769832424064,0.147757987027714,-0.265687376905253,-0.0559888026212095,-0.00613885641941231,-0.0736174933036766,0.197210975265828,-0.340705204646103,-0.0851565108931849,-0.364203654757308,-0.0716511053210546,0.0252161882757129,-0.0376126556286929,0.0350161163512153,-0.225211437256053,0.0150320237910202,0.0720786537002173,0.0879535606921618,0.0313044294325065,-0.103862845334152,-0.53442763470904,0.0119951882431146,0.217997536083502,0.0832067103098478,-0.106407394921586,-0.0598106977948389,-0.146565706871634,0.0188863953972182,0.108747854100669,0.0495106421368047,-0.02912344169967,0.0144825640031741,-0.061166275877909,-0.0911826901482959,0.0667782588110881,0.0115986146009749,0.0571622099116504,-0.189741298725944,0.170901208204805,0.0600006708591804,-0.295040153429998,-0.0568552149061469,-0.0252246258987929,0.122914136739013,0.0592169750634216,0.0370218068006308,-0.160736974338117,-0.235623566384091,-0.0168364195716812,0.0854872261615279,0.0212064310171022,-0.220963163722546,-0.0254981293044687,0.130810784028759,0.0289808976868738,-0.0957648192888456,0.0857766747489608,-0.142155110239992,-0.0853417678615893,-0.155751108798827,-0.0392656411152324,-0.11516828619657,-0.110202671489635,-0.237630405251899,-0.132743869334264,-0.268391891982375,0.0397008464309917,-0.19903719324191,0.0214413683326704,-0.0673680588622444,0.0330780685159661,-0.156871389105056,0.000239712183787326,-0.108722838415911,-0.0963295401226788,-0.026616644277168,-0.0159748783343743,-0.194104857024696,-0.114593352375181,-0.0344409368933287,0.0845840477667477,-0.255536783481039,0.128672331478421,-0.00992906540082586,-0.0303131376984993,-0.120358540272483,-0.0993232332059211,-0.11020161127566,-0.0303033785170389,-0.191416369782013,-0.0393897656952312,-0.0937395337579961,-0.0342892290826598,-0.0130719685397885,-0.0771441352223241,-0.0484410990066788,0.0411695968799703,-0.0916967393182319,-0.041969218554684,0.00963909745253039,0.182959369795118,-0.136917002079307,0.0485838996270134,-0.0507596600604149,-0.0723864930535248,-0.125144550563712,-0.0567329092996071,-0.102469942273643,0.0947523561391641,0.254387930297422,0.0721188259197095,-0.149154922534571,-0.0838004080213728,0.0595770029227068,0.0417923165035781,0.0110320951916337,0.0702111009307513,-0.114235039752499,-0.023711664411941,0.0234108422131079,-0.185265665687904,-0.0183572289655495,0.0182094004953107,0.164107004495063,0.0139797478986346,-0.0696581157231827,-0.0126135027609766,0.0762840052975542,0.0692107954490077,0.0306022692336,-0.178651331749796,-0.0877575047111598,0.20414382417064,-0.0334656800893432,-0.0367472051137216,-0.00872844732844355,0.043381534465838,-0.167292630630721,0.0351114690898296,-0.176426969426462,0.0274782164356679,-0.310368939618041,-0.234010381478309,-0.109354502503572,-0.0280763701227714,-0.160239043019089,-0.0242685258875922,-0.0663073152947467,0.0635987270641246,-0.11443446200911,-0.0649445962426698,-0.100739031662527,-0.104038549718331,-0.0925731613339344,0.176032411498925,0.0068373119144714,-0.0362345895927048,-0.0488303909493175,-0.0565313266455074,-0.0638483183525682,-0.104151960632168,-0.100657047878814,0.0155415644740853,-0.114311680013181,-0.0677364944079306,-0.011524438453552,-0.0993700789155972,-0.000115836083374197,-0.051901136273018,-0.0577021837246071,0.0333866721169318,-0.107320814724699,0.0246929337240884,-0.0773072285759323,0.0408628627156342,-0.0752022804380452,-0.00626126956474433,-0.0216568409645918,-0.014760197331204,-0.0595243252680623,-0.401150545136301,-0.646270737248568,-0.345212507756206,-0.155533439704241,0.014053792103214,-0.0979294585075965,-0.0372721555465156,0.149357506534952,0.0448195688584419,-0.40228446390026,-0.0343805481630994,-0.0780389365994161,-0.335877695654182,0.0380699326176174,-0.162445643492617,-0.0971146523155512,-0.224608251383578,-0.111216041980865,-0.0568595285590925,0.0263450430531305,-2.70289472602957,0.109999053682796,0.0711703179062991,-0.176616327364772,0.244734067976631,-0.140907767670085,0.114805005885924,-0.12347634978307,0.0966726982791836,-0.0359065821273617,-0.155735271560797,-0.157010871262032,-0.427830212092755,-0.256834565247745,-0.173579073390038,-0.519556017419888,-0.752759816406641,-0.683656226913656,-0.780624921132806,-0.448498905148993,0.0150334647328777,-0.111069256608198,0.0434197548374371,0.0468629208251451,0.0672606238533146,0.0105491340575299,0.0411391770258384,0.130745939268741,-0.0877931995757179,0.0741739007528728,-0.00493616996965609,0.0335437414478795,0.0490383603610754,0.0288084052151842,0.0330667674661606,0.0373529344568502,0.0891558669323796,-0.098357300539238,0.0340991959074464,-0.0552797052015887,0.0225116796307054,0.0477933794975111,0.0774366419366191,0.0324088418889773,-0.0184059523676713,0.0488396740885324,0.113743461232487,0.14997113617037,-0.110778578932078,-0.0280117691566664,0.0864679268810486,-0.00514904630701555,0.11140012589851,0.0859146130326185,-0.159075585332072,-0.0368356667357351,-0.0362646916331567,-0.104554952872579,-0.0548974358042538,0.0327124679434983,-0.18551830533671,0.043989337331801,0.140942941239299,-0.0951875899070961,-0.0195633206268472,-0.229236835393935,0.0723451703771269,-0.173028237570655,0.0656328252599227,-0.132494762846746],
[-0.0230188617552337,-0.0364095291317711,0.166623970202308,-0.144867096870978,0.148478356816879,-0.111366051408555,0.0528690061901367,-0.0571339538354896,-0.047314696509501,0.0972038399236766,-0.00149863550306818,-0.0687493196753188,0.006319762087037,0.00972955097875519,0.0267108156700402,0.00104047037921108,-0.218884088333597,-0.0188607508229093,-0.00116862408590659,-0.142248542113194,-0.147125213406425,0.0324053518652934,-0.241287266822834,-0.0136468725516659,-0.0225172566947782,0.191745107615783,-0.0335592826803101,-0.87039905011862,-0.256444529054305,-0.424698013461829,-0.0548649962802431,-0.0415586753238034,-0.0989647626408044,-0.0621057964027584,-0.25495736372414,0.120289081407353,0.144840838677513,-0.0596790708213669,-0.163914214866791,-0.302239584134337,0.518474019265,0.0752618760637573,0.172419495922818,-0.200530944849673,0.038411222925179,0.0815111564905805,0.253359466601786,0.0406725447898861,-0.0512711109858645,0.0343583466296598,-0.00270587539907665,-0.108089862231456,-0.0982198494407065,0.0764812615643404,0.220460438237764,0.0657492830267707,-0.0258721287511893,-0.12641618419761,0.0479409366476627,-0.0204887418125965,-0.267265825554974,0.0959520401745251,0.0820087692511655,-0.0312538001271045,0.103605145194895,0.0270328511954518,-0.355667285654073,-0.265630528071309,-0.252112172074751,-0.326629298913986,-0.170105824069357,-0.506629354768822,0.00234064774780747,0.0201377327306289,-0.348422525791395,-0.315189864375667,-0.157272673118592,-0.0865442937282843,-0.157132106967957,-0.08655266794068,-0.0819351179578804,-0.119083919451932,-0.138975290458875,-0.100925547027384,-0.152999183484851,-0.561146525415662,0.0633888692759152,-0.454987805909081,-0.0930957830408411,-0.14765679627275,0.0716887851167955,-0.325854665703974,-0.0444243216770818,-0.0996981135332993,-0.0876693170575154,-0.0457387620307908,-0.138930141013651,-0.185760941903262,-0.137365295229876,0.0324058301775865,0.193685386144992,-0.34868651526943,0.0789502687788609,0.0569251514095981,-0.235365519876766,-0.162856186055965,-0.112679340757861,-0.105839934576682,-0.143318266253634,-0.00521671964674321,-0.0721913256678513,-0.0734915401568124,-0.0237839455244537,0.111476364899003,0.0351886738484606,-0.12182233451333,0.0542747022256716,-0.0542859551846365,-0.196883292631015,-0.00591680781977382,0.299447768100443,-0.0635022082548225,-0.103799131106206,-0.273888507757269,-0.0464280169394605,-0.33265902526844,-0.0848953555258522,0.115284559212419,-0.143289669667327,0.576690136915208,0.0630228936343844,0.00877455764440303,-0.0348897106884498,-0.0261238715448772,0.163628668624837,0.173564667252,0.197210740640151,-0.115604645972267,0.27926872196179,-0.154324449412159,-0.159726966168788,0.311527077871654,0.0242281088226083,0.1515293414247,-0.574072780278967,-0.132607180399819,-0.21311208242906,0.20766518328232,0.166434012543343,0.0789954999897849,-0.317766027780738,-0.0386449663187514,0.0311836876747886,0.204675481680536,0.00492511591962053,0.139702597908063,0.16381445867679,-0.182273761741216,-0.130200041488827,-0.0700121964802337,-0.0208255472707986,-0.223032573867598,-0.274956162832723,-0.0805056958331204,0.0262333156485653,-0.12573721343558,-0.115700281896337,-0.0346425163035652,0.068786650350372,-0.139135521074742,-0.015341980158153,-0.217884763372747,-0.102097705824587,-0.152641997038768,0.133571244334026,-0.0601971574254416,-0.0025710214963447,-0.0317820042862276,-0.0198380976325765,-0.0264055168576801,-0.114030365256017,-0.0691672541331047,0.00243705302161838,-0.12694842002516,-0.0176889206831699,-0.0739671794879576,-0.0201260448308839,0.0305587702600519,0.0511342693857873,0.00406426174854017,-0.0133283222179842,-0.117542397424821,0.141942413267391,-0.0650526575895507,0.103813624560207,-0.0981551794553741,-0.0484529064539185,-0.0565044187570682,0.120697927473334,-0.0426655503943625,-0.685098347389946,-1.12646059645423,-0.675328018259235,-0.176828538642958,0.00453243973390528,-0.200118495604984,0.00020691527903744,0.0507646104108415,0.0229465966900913,-0.709956466410437,0.0420563751770269,0.0127602418189343,-0.432650859534852,0.265305963152643,-0.311546466287333,-0.255528646160937,0.474165199796068,-0.124532263786907,-0.0219444333843445,0.0527194544979166,-3.67573377540368,0.10805905609454,0.180125209786571,-0.5368642672573,0.257623267488219,-0.199139609426137,0.259313468908423,-0.184521233244241,0.196057261504335,-0.0741532410906876,0.0636223579578772,0.102953812770194,-0.322453273626073,-0.137759482788845,-0.150975822043212,-0.547772337406236,-0.790532102226942,-0.731696562837859,-0.98886692414643,0.224210901867809,-0.0102045801350614,0.296410027637651,0.16589500884261,0.0779750437766946,-0.0409926200509657,0.0728549930738186,0.08612016184726,0.0981265849156271,0.118574737689742,0.142833746214954,-0.155569981686142,-0.0664920171235257,-0.0182520713646188,-0.0569394438495016,0.0391196496644209,-0.0703924603413218,0.137023729250308,0.0845479598633079,0.0310315024493754,0.0708493138001996,-0.0137272205139184,0.0986683521149538,0.0721198343291756,0.0238736838737569,-0.0117365860186699,0.208248896519575,-0.00563010722492716,0.0384472997785052,-0.241165994650922,-0.0110640094881184,-0.0480775429864901,0.160300834608043,-0.333957753755201,-0.170711786822929,-0.0969332878945273,-0.0327660881464696,0.0839856441189661,0.0794700090232656,0.0268271231870455,-0.174320620704479,-0.118539914247248,-0.0735707483052557,0.213842641782638,0.0878657205621775,-0.00806030577500007,0.0486869236535017,0.00152389001978381,-0.0816624612415225,0.0200963440011598,-0.00342580649835469],
[-0.0390541107789201,0.031206310271125,-0.00181795522641157,0.039493176768303,-0.185771731189534,-0.188548582146997,0.367346138103846,-0.003549906385048,0.0636908181607062,0.206045791305223,-0.0529351334257995,-0.00600704242322111,-0.0162438074113492,0.0203982350630446,-0.0813146286274634,0.151406460070851,-0.121861319179104,-0.0382194830223259,-0.0588247818439919,-0.0400530482350396,0.00145367099519598,0.451832981840102,-0.38191646146545,-0.0474712466730322,0.0841658478093755,-0.394486688784946,0.301464106970244,-0.418707886035087,-0.0455189091972202,-0.387794520686594,-0.0260384656761557,-0.0279949710279901,-0.122709644114367,0.0526914233459027,-0.0694132712754447,-0.0553067578810802,-0.00876110612939158,0.116534792741631,0.0347284433198433,0.047982329534243,-2.2525441621547,0.136922163342154,0.213344482861955,0.0728076076415284,-0.331669827121515,-0.0488676833453113,-0.508068012753683,-0.221546779295961,-0.0979968190548259,0.189748868126686,-0.0643157681886079,0.00741617549547089,0.0465596631209791,-0.011380292739247,-0.158788155797611,-0.167202155475424,0.177205508522542,-0.159067853344716,-0.0344722913468104,0.181842646612986,-0.240979817377938,-0.0480160159218107,-0.0112851192607457,0.109063000540835,0.102848013522966,0.235035616495682,0.145762402387519,-0.229086156809127,0.124521011830259,0.143973369559654,0.0644975018866173,-0.266783610983835,-0.0873203643974462,0.0337701160196601,0.0632901076976688,-0.153569913559774,0.0286211152748804,0.095576546983839,0.183172145151699,0.05117327703257,0.00229553982897649,0.0723480588224619,0.0448525266475473,-0.101225418346171,-0.0252408847782227,-0.369762944196896,-0.0722994180148855,-0.33392398505415,-0.0130059013827317,-0.0518401158705672,0.06668443829827,-0.297452749374968,0.0409009531711602,-0.0188940488069233,0.127799433160408,0.0656175574120169,0.118032060618383,-0.0019976344204434,0.00044661285183403,0.299776741117153,-0.0187596219815409,0.145932715501889,0.2391119925099,0.12448594879334,0.0876137580547218,0.0904661682239174,0.122727552751339,0.100885164835598,0.185956013684708,-0.0264236414464564,0.0477521782357848,-0.110945105377599,0.0257225254583167,0.0976993811573017,-0.100456781983968,-0.0467416198948218,0.0237541581306605,0.0952621630167,0.141839316826802,0.0590697598399725,0.151974079605954,0.0702801198710529,0.20258868255817,-0.071149967040302,-0.022994837253026,0.0253225971812663,0.0293769167437133,0.147516798043142,0.292868704733635,0.0779257570303834,0.0748070791407049,-0.292934682972664,0.157596947849346,0.353092963887274,-0.00208395676616388,-0.163036522810349,0.0650917458097359,0.0447231901441415,-0.188009657917806,0.183721896353475,-0.0676523586379279,-0.428095076142893,0.0737306180606567,0.19517466743622,0.306435599816718,0.0603810910126134,0.0642022700681943,-0.0743720243925702,0.191187408706346,0.0392984601103653,-0.155507145969081,-0.0760519150885846,-0.222337245462087,-0.132121177064068,-0.0720004368803655,-0.0499273400880768,-0.236971428927091,-0.346600745586519,0.19562504962554,-0.222596508556301,0.0129365606307783,-0.383330297619092,-0.167703188092517,-0.139890600997795,0.0263555093027306,-0.151377715338546,-0.0294166880132757,-0.0556799635591464,0.0809736094382322,-0.121738740005174,0.0090673572715717,0.0102199009671469,-0.188728113045909,-0.176821030032859,0.0850814473875211,-0.0521263364029097,-0.00177034039567285,0.0607111128318944,-0.0268579299356943,-0.106778344378413,0.0344383328767499,-0.0833316840035742,0.0117960616042866,0.138276690153143,-0.0872414625056532,-0.0694476127202616,-0.0390973856944603,-0.0740207126520475,-0.0422903198203874,-0.0130752575027589,0.112159850076751,-0.149141653445886,-0.0489282407043238,0.0243435606450147,-0.0326134071253996,-0.00791196110758567,-0.0235253239246395,-0.128122906194145,-0.0118644104924248,0.0236805048215641,-0.425602415150076,-0.708650167599997,-0.373162969808889,-0.188964639117935,0.0281413137924554,-0.123958076685825,0.00777847784597172,0.139584505639283,-0.00468280168395332,-0.478380726354064,-0.0110134686824468,-0.0569600654911884,-0.482802957968438,-0.14049249862189,-0.221965955949244,-0.157547371623152,-0.372692278895114,0.213216192360351,-0.0259024527998145,-0.0237886425015126,-3.03494357933777,0.294685462494463,0.167235778208885,-0.22123061731046,0.329490741213979,-0.114532230593421,0.207539763864461,-0.261592437471576,0.164175030545985,-0.0484390410341852,-0.0800730150357479,-0.111586744070978,-0.515381919149569,-0.249175296164568,-0.208048795924471,-0.538528856554916,-0.732105162237164,-0.642262887890008,-0.698725234673832,-0.590136667780754,0.0786602359111887,-0.189235620613535,-0.0233198595380485,0.0833782337517271,0.17614375784497,0.0153604412715514,-0.045258910926621,-0.0942964657966552,0.0770515828467494,0.039233097820728,-0.0228980534811963,-0.10471449607571,-0.0931210348693067,0.220728045038259,0.159056657528336,-0.104165300265673,-0.0493334529431317,0.0629361657286663,0.0434320922054553,-0.053100982085659,0.0833644000299567,-0.0176623008215256,0.0354899121798993,0.00120020350165115,-0.0250310260085145,-0.164553402972806,0.099708466657202,0.0941987075242864,-0.147625461893181,-0.10604452881789,0.202702053041784,0.110189441789853,0.254752754503938,0.00045844552918688,-0.175360764454503,0.0624622546223222,0.0560743084089922,0.0168242345884896,0.0566149737819868,0.142694647349389,-0.195920922532793,0.288156660380773,0.0306128485141387,-0.00696758896552496,-0.202699427815676,-0.0708618618151552,0.0731145719012436,-0.128064324608984,0.0249865302310943,-0.00698932149506027],
[0.0368685739892246,-0.0617170100715589,0.0477535595706249,0.0213637572543437,0.00355145686954315,-0.0760388971964029,-0.0264386201100073,0.00897683971522454,0.0514924561164893,-0.0423194494132706,-0.0108771396718054,-0.0714135464903632,0.0301373569463068,-0.0245337614630987,-0.0446111658125919,0.0655414136504202,-0.0963083169828769,-0.02118592885095,0.0407861161940649,0.085689870256214,0.069245771864159,0.15799722179332,-0.154086853017146,0.0991128020658834,0.112901609717705,0.00450170865623856,0.176559005416068,-0.160013848580992,-0.119590125538225,-0.241169042168162,-0.0223832328257742,0.0185212959106507,-0.0941694077020713,-0.0262480219365895,-0.124519603352102,8.44917098558612E-05,-0.0552674649764099,0.0934224788857624,0.0400518805947225,-0.129659086067031,-0.270340233118899,0.0346303678115232,0.11226297251389,-0.03981206797652,-0.0158739335673528,-0.0269416558879502,-0.110489999204819,0.0571271993324013,-0.0184064919237951,-0.048433293136976,-0.00669158826476248,-0.0569552488252672,-0.0330262546835394,0.0461029458666126,0.0567510338383216,0.0503731412768087,0.0318201041162754,-0.0778486172674734,-0.0443113346486842,0.0605136886867928,-0.128718682764714,-0.0331017471068283,-0.0198626261475258,-0.0633054146666814,0.0625364918528072,0.0237944218992871,-0.206882522030331,-0.193405477433051,-0.163701888458653,0.0479372220536071,-0.0400243497738179,-0.240407124229495,0.00254868950575422,0.0570433106911365,-0.0272666076750029,-0.217716492060111,-0.0835037861262982,-0.0565863402233998,-0.072522751989457,-0.0349940999896664,0.0329048907341685,-0.0318507418513765,-0.0724571566520175,-0.120799675318091,-0.1131266039861,-0.31065652264195,0.0341444652569948,-0.240875934853727,0.0411344136454505,-0.127565817057253,0.0283870864658597,-0.22019131650205,0.0119404935531915,-0.0677720452744371,-0.01274972660491,-0.0569725067360171,0.00928608622608639,-0.0828965128821381,0.0188803829670694,0.0172667397756679,0.0983954084994691,-0.157515759970062,0.0673898475180043,0.0803750343389978,-0.0657469621977003,-0.00601859414359281,-0.0249272306784842,0.012591448562139,0.0159702551932473,-0.00172186764079482,-0.0513260984888194,-0.119353346640642,0.0175415608572705,0.0226751760215161,-0.0648176987206364,-0.106169740921018,0.0602072660096294,0.00547840941856388,-0.0557738284100017,0.0286216831098137,0.160448830083665,-0.0182812647550858,-0.0240015325083138,-0.170348253022203,-0.0436289136672782,-0.218153515489717,-0.0170715986085342,0.00526569912710845,0.0103169601411432,0.333718601338746,0.0464655874772606,-0.0937001134535036,-0.033130991280194,0.0114517633271485,0.0519442634780211,0.00662230770889483,0.0267333812958363,0.0324584121006224,-0.0108215279098478,-0.0839430945674301,-0.121501387840272,0.0674619068360599,0.0494128395632514,0.0921174415133956,0.0437156629823789,-0.050504392713645,-0.00480083357988841,0.0340767996382522,-0.0466603324016091,0.0336059556253378,-0.224183678377304,-0.0460450102420528,-0.023283642402065,0.00171257367201428,-0.00764696020172487,-0.000485063575298253,0.0571372789411318,-0.178373648374664,-0.0318874557282371,-0.0533723612697896,0.0600132172696425,-0.18776223710982,-0.179024142563983,-0.0628349635840794,0.00959449442848857,-0.0333552220163856,0.0147469817477663,-0.0104998757666038,0.0833788815888273,-0.0144664070705207,0.0289782610728877,-0.0802526046983734,-0.0666732582575046,-0.0533042433392089,-0.00581467242712503,-0.0447465279834329,-0.0554188208334944,0.0151826466525428,-0.0018554232381412,0.0263403947430015,0.0415713249394086,-0.0485584743678492,-0.0552897428514716,0.0852042822058549,-0.0190023244506875,-0.0372436862187845,0.0561322209181086,0.0195326089970304,0.0393393603207857,-0.026488930290943,-0.0236468663563067,-0.0679552851983797,-0.021512137276003,-0.0398392860009181,-0.0174706335597566,-0.0750296809415085,-0.0246766563186353,-0.0677676931671725,0.0852076038328964,-0.0233855519743989,-0.356333784761585,-0.576576400411655,-0.417465665194906,-0.113625432522859,-0.0289439599047318,-0.142724390522266,-0.0304807381495369,0.0418062350135521,0.0173499367505299,-0.359073896845759,0.055510053084393,-0.0379985152806597,-0.334696398340712,0.0438292256402433,-0.189204730571461,-0.126152010175713,-0.0103223804424763,-0.0581202712959288,0.00508547109765112,0.0234461869859428,-2.31554226991545,-0.00617437795681818,-0.0115233310139403,-0.329234936307765,0.152131148517815,-0.0362481324811969,0.169589002210855,-0.0889572129497796,0.147809390848521,-0.027696274169747,0.0573801417384906,0.0461627920398348,-0.162060806367861,-0.0849528323804849,-0.104332296557342,-0.289003212789818,-0.483047891517898,-0.427039710257823,-0.53291132523941,-0.308971960771461,-0.147392576685587,-0.0244837251021709,0.0259648313076544,-0.033980723143779,-0.0228610607318808,-0.027199729893121,-0.0174663565023382,0.0216298598538347,-0.0483856831337645,0.0792148883860433,-0.03192539357961,0.0279028403512238,-0.0505364219974667,0.0592646890643453,0.00896448180593733,-0.00651444792359099,0.0180448405170556,0.0247129402842993,-0.00964186708083015,-0.0107488173205187,-0.0326140293321683,0.07654921139506,0.0949874241347735,0.00400708064888588,0.038795181291138,0.0684724210160707,0.0778683602925283,0.0738076197237061,-0.154528120383766,-0.0332529859228478,-0.000227296312708531,0.0554181141659602,0.0353408037911477,0.0268990088399036,-0.0620314463191884,-0.0285745002360873,0.021595335673477,0.0024005418463239,0.0265863621390309,-0.0248119284631502,-0.076059394240424,0.0590306481578792,0.069613190830299,-0.0114685954072492,0.030313496318826,-0.0705928229787477,-0.00820947368597734,-0.137466613255527,0.0254138531745048,0.00864203028347433],
[-0.00573089825356029,0.0480919730980455,0.0474027990662357,-0.0637134075488429,0.10019229670441,-0.0187190156798779,-0.0426396099313935,0.0417953652019906,-0.0924428301100123,-0.113016333335552,0.0256022130950462,-0.103646581207755,0.0310741012666957,0.0179402948822873,0.0589007988442054,-0.0789575759759698,-0.172989861602109,-0.044983630347217,0.0738992987593627,0.00735457513749889,-0.00518555538745571,0.113796959495016,-0.210546990036199,-0.0474304464498542,0.0474865845588789,-0.157306332244973,0.104788018214146,-0.321592576566562,-0.158637708732119,-0.174620369284015,-0.0493024170670153,-0.0419011848439384,-0.00344208339524289,-0.121012549243232,-0.0285744415950911,0.025151664629287,0.127150261166795,-0.0277169114853407,-0.0876855034249078,-0.243947819068666,0.19558337579521,-0.0351560358514098,0.117296704026089,-0.106110318738859,0.0401958119479169,0.0868558792925829,0.0499945816507551,-0.0146751976857432,0.126572326672949,-0.0169104722611305,0.0161376477088329,-0.124577306633066,0.00333151741170281,0.0503889077598326,0.0708013585032487,0.0112612836182718,0.0374436252324039,-0.115039491758533,0.0953557802892126,0.0319619311905625,-0.150062161252436,0.00219879081912986,-0.0526986126811355,-0.111468329305279,0.0563541905855199,0.0134743996563694,-0.198834435214779,-0.146108065466945,-0.318999496916184,0.0065120348037397,-0.0429380620700932,-0.331129005372949,-0.0453643288640238,-0.0280273985111614,-0.100448565230285,-0.242555149987064,-0.138451791656232,0.00290162387315071,-0.055492969035465,0.0680156984314819,-0.035447339934462,0.0252902307482533,-0.100106945691061,-0.0510618667630938,-0.0668540468578278,-0.343444373521672,0.0761313362056105,-0.222581344036729,-0.0311340024830087,-0.119827615168639,-0.00183955483141453,-0.142361072699255,-0.115075788539765,0.00332210564965366,-0.0238810367294306,-0.0324872407920904,-0.0642000310122139,0.00524886862472699,-0.0341109186900569,0.109744909265576,0.0621262485573091,0.00569585060578038,-0.00562130916700758,0.0467786687855384,-0.115302124815662,0.0173963956359539,-0.0325109193637708,-0.0112319381763785,-0.00928159937505605,0.247687507334674,-0.162029735496551,-0.126981786473831,0.0366662856762793,0.117775014066162,0.0996346029716226,-0.0717641026009894,-0.0106910799519,0.0366589842948544,-0.0525573673913355,0.0974860051631583,0.183478089000564,-0.0318267166520263,-0.0488462471681195,-0.0819912713048124,-0.0261204199110611,-0.190815702045159,0.00241370324878498,0.0955695750109926,-0.0826029048373757,0.279476607442412,0.0299690076507439,-0.0736457621058599,0.0691623816892635,0.00139946912384906,0.0446996912395442,0.0548421849131618,0.102862923295063,0.0542329587142799,0.0870869540860341,-0.0609898587743868,-0.135784002499854,0.0378143701421642,-0.0282179683498797,0.04603543426871,-0.209231096433282,-0.0830449662993716,0.0432338439881411,-0.0403594257199329,0.0124495405714795,0.0122100146227207,-0.110964668669874,-0.0776652184661901,0.0202342265327373,0.167439079234558,-0.0100044335244948,0.10452652768834,0.125078970921602,-0.120435005050281,-0.0438751915983936,-0.0719459029288328,0.107901733841551,-0.152086239381285,-0.165579004944783,-0.0727777784665391,-0.00879160747766734,-0.0462411425096205,-0.085773499243182,-0.0282111081606626,0.0611211351770728,-0.0925179075547705,0.0318644129703118,-0.145196375005195,-0.0693630178760874,-0.0740391085268141,0.0597799113012531,-0.0148228815668606,-0.0328069401164895,-0.0941010306328344,-0.0146924230897,0.00372062092826067,-0.121419621385221,-0.0405536379666856,-0.00992825053492904,-0.017703998324455,0.00300662120812762,-0.0158877352105468,-0.111868061578923,0.0320344438509912,0.0354212151060844,0.0072953001939796,-0.0255288483589886,-0.0122442764785096,0.0482485412337964,0.0201309949144527,0.054044794229875,0.026798279326803,0.042078501532239,0.00527788732214357,0.0464904644018966,-0.00293857363953981,-0.389402243792526,-0.788131404269472,-0.410748394594636,-0.112716224210823,0.0480133135142847,-0.149777909229453,0.0512367272492269,0.141308429252178,0.0557966375763225,-0.502967673894161,0.0117085972024178,0.0210025261781474,0.00125025030231037,0.492905244593814,-0.201371369302043,-0.128918311315903,0.53030257083742,-0.135535751954297,-0.0348192136376522,0.0045211828973028,-1.98795346872228,-0.0303132758949554,0.0247655695817269,-0.323255388305218,0.210449226313231,-0.132365523830814,0.15895535108457,-0.0635810655229181,0.180381960860293,-0.0949565701738075,0.0735224711678035,0.101927025459798,-0.0874888096849799,-0.0326889023921434,0.016462981546294,-0.187554137210414,-0.304925685632321,-0.298346277065642,-0.478090631748452,-0.140474912314831,0.0365401162752312,0.0615881221923736,0.0286908589798371,0.0215073283648271,-0.00562764102022444,-0.00102210894011291,0.0961976563539781,0.0734017731623454,0.0463957567692929,0.0341240013948594,-0.134704851021942,-0.030936087631349,0.0355222410119564,-0.00375140480697036,-0.123286890840413,-0.00883643792342267,-0.0161120849506032,-0.0923986943699181,-0.00830034049143361,0.00390039548052029,0.0224105404959015,-0.0105963163138261,0.0290042195766739,-0.0403237226951401,0.0107275891512218,-0.0681270308551443,0.0272441536683407,-0.00657753005664299,-0.0904007404762011,-0.0410384421620644,-0.0558910750027232,0.0696008073068267,-0.256781782091167,-0.11791925919246,-0.0658969539540774,-0.0351714553360908,0.0635247386684084,0.0769687083464277,-0.091845281776579,-0.0499499343226976,-0.0947085682126384,-0.0448420329313627,0.122644232191713,-0.00143595002554831,-0.0745928967929081,0.0726545279849126,0.0134483617045162,-0.122029768959899,0.025451698862647,-0.073446034331681],
[0.0269895795206691,-0.225995872474774,0.226720002238688,-0.132659583216933,0.0959305437348327,0.432307288812574,-0.115748116404441,0.0266158577792124,-0.227360694202187,0.0231712288114562,0.0215993768221145,0.00464449415146768,0.00225490425860593,0.0810725527154301,0.0672814894310069,-0.0253279173773628,-0.262690083779502,-0.22302778768303,0.249761221171595,-0.11084937342799,-0.142638043417683,0.214154180994884,-0.34566800087271,-0.0937971940130839,0.0722949955785091,0.0857165080557554,0.173910177233898,-0.614517998902434,-0.226127116328801,0.0422664581539396,-0.0344355813656408,0.240782919631295,-0.0937492326003365,0.110526350167603,-0.0736606858374761,-0.204571516653471,-0.101055162952001,0.124007114934478,0.0856050396295753,-0.0706258387036444,-1.0534058938049,-0.0125470364106927,0.219639212539542,-0.35852813816273,-0.0318206004577505,0.535193169863369,0.115457713405122,-0.0414908030230924,0.219778732707234,0.245426370832325,-0.0172045211588424,-0.0273920125101909,0.0571233235711986,-0.0505927947748798,-0.0193081799260357,-0.0353585833616131,-0.0307434682901717,-0.209023270695074,-0.382399489076837,0.0123608566201753,-0.31239595997911,0.0607400824118636,-0.0415347558605667,0.0127164291799963,-0.266236351048392,0.304654282286673,-0.160643988062896,0.0884180020079726,-0.146925376580695,0.0842296086866252,-0.00295647107753427,-0.334656672163323,0.0177823874397102,-0.0639269174543212,-0.126491508866347,-0.207519932348483,-0.31408746820985,0.0663719022981608,-0.0695625258616741,0.371302953762677,-0.103448656378768,0.104020079824873,-0.0965206640028242,0.0127969970114405,-0.167083813765583,-0.302726205401995,0.268522915549828,-0.35541530284806,0.0170940117271964,-0.0859549673035123,0.0863681574486792,-0.144951000715188,-0.140190559763823,-0.152942470482283,-0.0824014132269704,-0.0378983428553235,-0.0761089104861761,0.141182151599931,-0.0726864418445561,0.28018422934451,0.00297027170440374,0.180484461038167,0.227692786496577,0.15899260660348,0.130085190140186,0.154557986744614,-0.0594959667756606,0.14512559739432,0.0157395442534868,0.329595025760979,-0.300213428496327,-0.128810539676455,0.0448348173691826,0.211008480062003,-0.111246893408124,-0.0998642251298352,-0.0291366777302266,0.119737835034166,-0.0633918094496281,0.166031465715737,0.0477361938493711,0.0698698766038144,-0.031693521036048,-0.137490510351872,-0.0841587508048267,-0.0603319548565925,0.228870703104626,0.56575209123603,0.3688016137663,0.421675380418035,0.145775622821658,-0.00254258019550897,0.0184232473373693,0.250893819074555,0.149995582899323,0.0425468770139033,-0.0307637240089074,0.244891865563362,0.295594869987862,-0.00231274114539726,-0.153148221395465,-0.0221626760765665,0.0797620469019963,0.15152109650633,-0.303770516402667,0.0269957782656572,-0.0133315540106415,0.108549923264924,0.230139575892376,0.0590504618250592,-0.204984859378902,-0.178605355172264,0.0023204363203087,-0.0930038489356508,-0.118960141892105,-0.0855469378729548,-0.023192022229604,-0.164832068310109,0.0738307473044735,-0.0807454481381034,0.0880566914523643,-0.281018020963706,-0.232166101225316,-0.0760961248768561,-0.0076260875158704,-0.111938795763339,-0.0311721869376886,-0.0138302852970248,0.0968621438255852,-0.135830678691643,0.0194782997326415,-0.106326834961246,-0.164115812900098,-0.171012008822371,0.160170274623136,0.0158465860560882,-0.0317682012269367,0.0108764629016244,-0.00823131900531485,0.0305269528665849,-0.265659310937592,-0.0377037137071945,-0.0192979253217032,0.0169505691777511,-0.0386918414189314,-0.0835221872323435,-0.0508406959599437,-0.000282048895882608,0.0419883490626886,0.00784458593314754,0.0812564280852877,-0.0603031738433217,0.360514085935025,0.182407235554114,0.174819523465318,0.148781160815678,0.134986374074833,0.0404455960937897,-0.0762133851281102,-0.113963088724833,-0.433559922081831,-0.666971001744853,-0.437705848707054,-0.0632954711701779,0.149064044126089,-0.159385325738509,0.168584548685183,0.111788273247424,0.176035167283289,-0.478947997965059,0.169281737098847,0.0841964147194819,0.318729754114468,0.94239058105733,0.0569271538200001,0.0132273720095337,1.57622579240701,-0.222776989311746,-0.0480162306724444,-0.00202773567199277,-2.21351493860364,0.244645383625476,-0.57073478747036,-0.245454730878323,0.351703201619894,0.0665365337952319,0.392313285451104,-0.086839483310579,0.288805355532715,0.0788049048105779,-0.00365481695727811,0.121625897384028,-0.343036442440231,-0.140721902165816,-0.0601632570679733,-0.461263286462549,-0.315697153687403,-0.203933736468309,-0.452528742473725,-0.729441453927012,0.0488395128269311,-0.248191668326199,0.050045289708223,-0.0756068259667798,-0.00021044186330936,-0.0473791959243933,-0.0202431402366659,0.0534357129848036,-0.0229858409689392,0.109036048397479,0.186518850535785,0.00787203528929377,0.067375602603005,-0.0998339194106716,0.0298045069927479,0.00384054313075032,0.00402114694746569,-0.123150706527937,0.0966502429555116,-0.00918636870512592,0.0471182176741948,-0.05933145601067,0.0960556601663577,0.0468232937174781,-0.047898599735125,0.0014598103848562,-0.0046296893669052,0.142632294144427,-0.16991422123663,-0.0499057967808313,0.428902028649302,0.342009548703583,0.578728918625514,0.52219914303075,-0.093847117243248,0.135038082368473,0.0585517700681105,0.104259450867301,0.175745205943615,0.412782821548502,0.0297289937820816,0.277123630703707,-0.0246160585489403,-0.167456612283118,-0.00056712072921386,-0.30495215646201,0.220390419316283,-0.133387967861291,0.0959641105531037,-0.140747816144684],
[-0.0250854543945863,0.0808902698679381,0.0167450782593516,0.0184390857635214,-0.0148200893886462,-0.0228421220862768,-0.0489778454055139,-0.0598262569094473,0.00562035388728,-0.0990383742465579,-0.0161629998115146,-0.0766858432808357,-0.0112020522479264,-0.0481085145330324,-0.0582377934939265,0.0794517978541877,-0.165634087565384,-0.107847504784696,0.00377862077670675,0.000114009101949879,-0.0491460058216136,0.0953681835269466,-0.211411693531349,0.0644554849223785,0.0469100138968673,-0.262647156262341,0.220263553229609,-0.314011071461473,-0.0930673429550296,-0.323255169913204,-0.0411195089346272,0.0926864803592152,0.0150682473043903,0.012209711902419,-0.12481227624479,0.0230308596753123,-0.145243447821416,0.140338981563947,-0.10416084529143,-0.133224643077842,-0.288553230729945,-0.220206270859039,0.255101974559778,-0.0820922894827783,-0.0690200772569033,-0.0469769208959608,-0.199302776813577,-0.0066675209040865,-0.0488622961327024,-0.0160198023084914,-0.0229545159562119,-0.0635164501171548,-0.0938209278250043,0.037062205283488,0.0839810199949,-0.0160526869529932,0.0217078673515637,-0.135895992857415,0.0687912949846769,0.0543050434595808,-0.207129006562149,0.0424592022856787,0.0194168089063547,-0.00761313292273317,-0.00221690803626668,0.0930739730408526,-0.130859609774711,-0.234953115184537,-0.135049077837052,-0.0169627238196141,0.0442269241047446,-0.340943802212785,0.0378081758925283,0.143549249688499,-0.0635743770238987,-0.217210225062074,-0.12721696751073,-0.0108493705203699,-0.0528131614020524,-0.0286754921761983,-0.0189741847045521,-0.0358609498337105,-0.0351189712282589,-0.092145637143255,-0.128503943712509,-0.262068989717299,0.00704710048072767,-0.228566681750015,-0.0124390734817525,-0.057337912045915,6.81935000524054E-05,-0.144539213441255,-0.0374168121916018,-0.0544629567091676,-0.0335307331124335,-0.00735800607866362,-0.0244418718804958,-0.122438027708326,-0.100057531285479,0.13341595150477,0.0881149515290754,-0.180830390310187,0.0862635280577775,0.0931007402318066,0.0422053096091438,-0.0814732081159504,-0.0301875205329082,-0.0568924023722857,-0.0142937763080288,-0.0736955180447701,-0.135143338508501,-0.100753748366193,-0.0220814994089224,0.0562174456349314,-0.0443243222650257,-0.121660411666233,-0.0345676453841373,-0.0103732973638267,-0.0900614030178028,0.0352073820903233,0.146668582864118,-0.052426687145795,-0.0172772428110116,-0.130053478237433,-0.00285172242810867,-0.116855474693006,-0.00899239308876355,-0.10885997242997,-0.0113862150432806,0.190181915340416,-0.0252422566006957,-0.0877785741030683,-0.132988768414687,-0.00254203798149214,0.109816669183649,0.0595015765571343,0.0811943544204824,-0.0187349835430949,0.0665791347117034,0.0262581233248825,-0.0932959151865186,-0.163923878620786,0.068953019095759,0.173293888056648,0.0593024212015114,0.00577731989067873,0.00465584704550002,0.0544956888858799,0.181418107691598,0.0185012843317548,-0.217321587510443,0.191371533301744,0.123353304695609,-0.292445604464784,0.0198541009216972,0.105134097620504,0.0934897208814763,-0.197158433097249,0.0609320716619624,-0.190675985293293,0.080494655924603,-0.258589910284558,-0.169943121888054,-0.0797619249010867,0.0301139274680248,-0.141823834017352,-0.0723542636860714,-0.0292046836747097,0.00482066430942506,-0.120237348752974,0.0291106636553835,0.0399421593742536,-0.078361968945365,-0.15807057440503,0.16013580194627,-0.0389019218220034,-0.0391382686955676,0.119062098083121,-0.0483479666484786,-0.0755475869592218,0.0270283228129814,-0.0464542497120473,-0.0118143589129893,0.00297493262036608,-0.0381774983110668,-0.0120665379099704,-0.0552262429376282,-0.0712508906777722,-0.0555649377120986,-0.0232574604686004,-0.0105125334535437,-0.136128921581494,-0.0378297071008745,-0.015447700898282,-0.0446224247330258,-0.0537615968700058,-0.109589187640676,-0.124232864644415,0.00717274749031053,-0.0520213710411014,-0.417938480329467,-0.751251856415262,-0.482288303016926,-0.139025329582369,0.0533357266462345,-0.166690146914224,0.0163867782808661,0.210646268970284,0.0581722488019091,-0.468985841398818,0.16572162389887,-0.019812025303245,-0.100913903831085,0.314273594820176,-0.165180848870448,-0.122966035783726,0.0833148758053223,-0.137497961136899,-0.0620482149757028,0.0136768754244099,-2.69661367483322,0.156676328736984,0.0999086699582613,-0.201275605845338,0.350151298035225,-0.154022128125388,0.100235487297889,-0.150923967315359,0.190255784229834,-0.206002280754148,-0.0596109535513859,-0.00392570962028656,-0.277577579352537,-0.20053647223684,-0.180910624369342,-0.43033530958668,-0.673403996887793,-0.607541308710407,-0.715410065821646,-0.205587576020683,-0.114784987459705,0.0324322957792392,0.055924920417399,-0.00396998107732525,0.0304473514056454,0.0316510087973866,0.0632167139188641,0.0477645616385791,-0.0140476712935,0.0890178579506797,-0.0722335819201372,0.111628178850023,-0.038595932450751,0.0174510301712129,-0.0503352699690459,0.101706333839365,0.0163082241391655,-0.0235239399920156,0.0431229253579502,0.0373780682857775,0.0379693004363555,-0.0411848958301328,0.0106728069224739,0.0168533822596779,-0.0162841589751366,-0.127157314987477,0.0193729142129854,0.139853641517076,-0.160720240831895,-0.050844984395552,-0.0195666038446201,-0.00814181202969378,-0.0782283383202987,-0.0689000712198766,-0.145937057791917,-0.0917991194173379,0.0695797594911946,-0.0448344393277357,-0.0532334671288688,-0.0281581502659663,-0.175990472665291,0.034159183026069,0.165579283507716,0.0602339054490555,0.0743199596597972,0.0263641288604269,0.0645230732501361,-0.151545990568086,0.0571473651295238,-0.0873780243227162],
[-0.0449894272460765,-0.0123157508188268,0.0794313741040395,-0.00608752529462566,-0.0143084957383885,0.00613868621474584,-0.0154547543653233,0.0210643653955001,-0.0466479810681625,-0.0343303052769265,-0.0141137091940793,-0.0828803438066188,0.0215408770415175,0.0015108038012913,-0.0144767408137036,-0.0375620853000704,-0.114568295926071,-0.0796819637306505,0.00867562757922736,-0.0466358723400774,-0.0648113389224053,0.0405453711609984,-0.170592089644438,-0.0432191602024927,0.0214645892962806,-0.129823663198477,0.0838755046552028,-0.200352663788261,-0.0480326982645262,-0.150630914927044,-0.012512499104111,0.0263453124534543,-0.067275835826218,-0.0141070432171492,-0.066266239110239,0.032181848672889,0.011926987911612,-0.0332394852960702,-0.0546239044621893,-0.102453155414289,-0.13915340295484,-0.0418079550882334,0.134497864301676,-0.0682974578787988,-0.0434332127652818,0.0280901267811988,0.00389954683229166,0.00536693438283653,-0.0777789559650576,-0.0425831069163694,0.00561150961313482,-0.0968414857111584,0.0273104859318596,-0.0765132107286628,-0.0193560703433683,-0.00329761704525134,0.00450314627033272,-0.0499070404979877,0.00825798670946088,-0.0131547517474166,-0.0723936279150189,0.00708328315283485,-0.00247184208731255,-0.0389185674847044,-0.0332048829088744,0.00574728373242188,-0.0818576291280804,-0.0595664588878089,-0.183090444760656,0.040195676470777,-0.0419137306961142,-0.161458215786985,-0.0128494163856533,-0.00662106143018395,-0.092051667014232,-0.100625322279372,-0.144071655456504,-0.0228691505301417,-0.0529068863583014,0.0173739518788663,-0.0480283991470619,-0.0556756558621924,-0.0843834830380968,-0.0797922816438365,-0.0482842143875137,-0.184459328706077,-0.0454637729760127,-0.206043421494445,0.0202932803223966,-0.0560005895093197,0.0160120945383344,-0.0913072756132399,-0.0259851825936195,-0.0501246782011729,-0.0772179812475108,0.0328144036158307,0.000804652745569066,-0.08020778063115,-0.018809663068117,0.0988069533979883,0.0198920472373087,-0.0911371666459053,-0.00709668946547931,0.019141315832565,-0.0548155657481596,-0.0680644864308179,-0.018867914653385,0.00320779674442048,0.0270027284406576,0.0468077195499461,-0.0501818048631467,-0.0238609568246525,0.00804551788782691,0.0376628738204361,0.0208991456650241,-0.0431881977853311,0.00448354391601012,-0.00502254514429491,-0.0616948235934619,0.0335648655342386,0.0489076530622507,0.00829813117250408,0.025584483863895,-0.0444333321593668,-0.0243199627100102,-0.085818318482363,-0.0748822159684967,-0.0117800379195197,-0.0408721352496086,0.127004428249292,-0.0351895783113722,-0.0789311987406784,0.0021337904070739,0.0276006039155389,-0.00237207551186287,0.00289373798512219,-0.0304191167540265,0.070629337052483,0.104713428391646,-0.0593794231946377,-0.0452863321348222,-0.00277134311700644,-0.03794677684433,0.0779901342005677,0.0102940497715332,0.00334069689833071,-0.0756862823720698,-0.0502064132380612,-0.0359334503828457,0.0100944638185234,-0.106005000606992,-0.131887451530168,-0.0563015526013891,0.130879797689183,-8.81272805570545E-05,-0.030713963903296,0.00782040224427834,-0.0784227614822005,0.0409601222101661,-0.0119727538719364,-0.0158853327367154,-0.105389709601456,-0.104375982663041,-0.00224458914059118,-0.00953964788605472,-0.0431779007286854,0.0287686770501672,-0.0568245954540351,0.00507123446508919,-0.0401682941923855,0.0158414357124377,-0.0799345019198302,-0.0267313581144736,-0.051500193707013,0.0643627590491094,0.0448180071135714,0.0227846101274141,-0.032581409335105,0.00374384419349944,0.0146384570125241,-0.0568886251245585,0.0114234478669536,-0.00259481678222141,-0.0284095415270936,0.00187492271239154,-0.0676636776486176,-0.0865025583497946,0.0328351926070985,0.0447502182620376,0.0367086908180818,-0.0312446829835871,-0.000514497222037562,0.0530719492663289,-0.0498513256938385,0.0187730421274266,-0.0260146772724916,0.0215916355051208,-0.0348685470125745,0.00792092092864849,-0.00675123788923257,-0.23343402954568,-0.520015110546973,-0.222772566518602,-0.115175513569499,0.0559732813326372,-0.0836567194481502,-0.01434816582466,0.0669515067906361,-0.00508483460553753,-0.281805611509194,0.00828981602843618,-0.0547564651638163,-0.0397825066373355,0.186420980290872,-0.124131451674211,-0.112292841419574,0.231809904838303,-0.0951540425287987,-0.0553449696145093,-0.0180649969696765,-1.31697965665769,-0.0640263307241411,-0.0907126194186822,-0.228522560637881,0.0400186125030989,-0.0403121483538361,0.137757063366079,-0.0609332329371658,0.0475267762027042,-0.025924109078365,-0.0141580330965545,0.0191196365670146,-0.0881745928510878,-0.0727868220885441,-0.0856475171119987,-0.160012847572103,-0.197914866523792,-0.247914795028014,-0.273780816064812,-0.124094472490134,-0.00304645970954682,0.0232694736584464,0.0684373758152661,-0.0139950887556379,-0.0502259717186715,-0.0272160836665821,-0.0576955289767618,0.0758243178860583,0.0491885278270082,0.00798128254307001,-0.0630516122346903,0.0233009722869281,-0.0176881999501802,-0.0770209945754979,-0.00120341589996899,0.0267191993881295,0.0693486561458574,-0.0576227176051289,-0.0178950556916521,0.0873505230903417,0.0547555970437554,0.0343339722717057,0.0254885092001042,0.0222681391754833,-0.0351634756791456,-0.0272074657463832,-0.0525783725357597,0.0310105475188962,-0.11632958905798,-0.0019825530181415,-0.032554681993158,0.0612344345690671,-0.0636449777974066,-0.0560222087924194,-0.0767697193096037,0.0115921939902489,-0.0339729445567024,0.004405807100238,-0.0628099136019909,0.0391133335679381,-0.0364266401753885,-0.0291392932958548,0.0541938801589963,0.00854180197482159,-0.091302590485732,-0.00327755743443334,-0.0103653304236511,-0.0543355387444617,-0.000643167472168596,-0.053680904215366],
[0.0143863138518034,0.0424378642941654,0.0190031100134921,-0.0286751798686811,-0.0121194655298738,-0.0586190659137485,-0.0486328835045591,-0.028732238490836,-0.0445855459173814,-0.00490936293483249,-0.0446184667347059,-0.0148600019459775,0.0295384025503896,-0.0188182163943057,0.0274211288243042,0.00484686873421988,-0.112783542030809,-0.0892677716171628,0.0396599477637981,0.015389799997993,-0.0666875691279123,0.0665299041261488,-0.185647959345855,-0.021962466052024,-0.0144837852978907,-0.123879147490159,0.0156412413912674,-0.257022718103264,-0.0718852206207323,-0.158748779206537,-0.047329228630401,-0.00821905977251349,0.017863944025795,-0.0574859271739717,-0.0383758419298569,0.0295170852131299,0.0595221387202944,0.0120281772905513,-0.0540155164784436,-0.151372178113827,-0.0296024574280517,-0.0834300765386544,0.0511512210704928,-0.101360628808878,0.0284484928000001,-0.000550998393980555,-0.038733351751741,-0.0625042941985927,0.0302663174326449,-0.0646278802299946,-0.0439209377516973,-0.0451397060292929,-0.0534401457871197,-0.000215814613436987,0.0116420842977276,-0.0731125459781108,0.0177411206447979,-0.0451553369334773,0.031968645131813,-0.013860992989726,-0.1670766376688,0.0112964213217848,-0.0482503317980833,-0.0287724225394421,-0.0090200280225845,0.0452874834657501,-0.0954405283561899,-0.0594064154095729,-0.173322008717969,-0.039407524799282,-0.0642837061234657,-0.273751105891591,-0.0450203604972313,-0.0443865369577319,-0.113190266609784,-0.207281226933473,-0.132755988458723,-0.0138439155798026,-0.0595865579492609,0.0564086171239525,-0.0915183731858557,0.00138470294677058,-0.0780997370632391,-0.101085342949865,-0.0830076304510215,-0.26385692771629,-0.0103970526641463,-0.215733937874389,0.0139858868084204,-0.0440798229066232,-0.0105162845097001,-0.0974845568946888,-0.0337790978375236,-0.039730787136909,-0.0695550032350788,-0.0408668080127987,-0.00258713163663893,-0.0911566958593472,-0.0229723390511792,0.0507698669634235,0.0121625974809393,-0.0638293304471402,0.0337786058952418,0.0710829331022732,-0.0796464232274546,-0.0843082960946913,-0.0464544543828881,-0.0799188074160064,-0.0745060065026746,0.11285379522823,-0.105655866996086,-0.0409903373762726,0.0128632933773387,0.0894944230465808,-0.0754430928041801,-0.0959322872619974,-0.0272761967579687,-0.0632544601323224,-0.0662061177612061,0.00487220601242014,0.0672327542862204,-0.0580954296223931,-0.0326127342609728,-0.0683479916221142,-0.0549467729097025,-0.157994109314204,-0.0543262955148064,0.0478220384577328,0.0272353304631546,0.210044474914568,-0.0271741653293668,-0.0670076398316931,0.0440913205408915,-0.00727394000294791,0.0358209847284261,0.039745524544955,0.00484159097450145,0.0299754642338248,0.0751246595382346,-0.0903691243479215,-0.0768475765694941,-0.0651588851950129,-0.00633768849896886,0.0920154695490926,-0.0842538692514266,-0.0258599932886718,-0.032307593654989,0.0317836430347571,0.0204794928005255,-0.0301406333142122,-0.161704397206853,-0.02788010678716,0.0332533597140006,0.00831408513016869,-0.00934207408137617,0.0889444175752655,-0.00167062445733837,-0.0778079250688831,-0.032433366726461,-0.102647246254984,0.048860830902404,-0.0852544134280842,-0.130866911283245,-0.0752947060377966,0.0273317910084192,-0.0904782201999342,-0.0817014079580443,-0.0348869344771204,0.000615018930005575,-0.0629306003489842,0.0505203094676395,-0.0687805803967137,-0.0862165845768915,-0.121041290167737,0.0528173494925693,-0.013242890537482,-0.0559054401828766,0.0141156867395333,-0.0264295827890576,0.0322657884204199,-0.00723626733726719,-0.0279386631760377,-0.0520988124658442,-0.0266844659755342,0.005738265727229,0.00932629908539216,-0.0457543329978707,-0.0466696598007611,-0.00587584974175017,0.0179158623280138,0.0287813609266181,-0.0400219021334773,-0.031828688806699,0.0124852998702455,0.00531329391519427,0.0135479090757608,0.000578161866857443,-0.0934896142574012,0.0831282007096953,0.031024391910642,-0.306407479024833,-0.504358725067946,-0.303904387660589,-0.123347610339643,-0.0117750844933697,-0.118205605827095,-0.0154870954351115,0.121641013459142,-0.0134026664328381,-0.358120516615663,0.089052904645252,-0.0317859148239714,-0.0663743854363427,0.283369959116355,-0.156064625250528,-0.128361952317773,0.277238970218278,-0.0805113266246495,-0.0206481840035167,0.0375141862718481,-1.54966816858391,0.0176221596067802,0.0177722803498843,-0.231861580354794,0.120251429893477,-0.132065917441836,0.0796666749215526,-0.0812598659932485,0.0853756774097076,-0.0500937450484609,0.0346905787343919,0.0788114423670915,-0.105171158714826,-0.0391420673899219,-0.0438622070809376,-0.253354046949833,-0.360656521390287,-0.271721937532512,-0.381429684112001,-0.133773119649252,-0.0103887325675283,0.0685401907359409,0.00356125050554677,-0.0349544784388166,0.00913834376241017,0.0137652651872327,-0.000758829548677008,0.0781883594161777,0.00267735469120784,0.0234028491576132,-0.0246736373413138,-0.0211063856063048,0.0276161446825627,-0.0543850257969518,-0.085813797798815,0.0126481726680272,-0.0106779432919119,0.0275724231785871,0.0503119313607665,0.0594714402496913,-0.0139595992655812,0.0338330959798998,0.0263406984974536,-0.0203704217185047,-0.00217661358930345,-0.0740053673452747,0.00687346313178403,0.0366903897516713,-0.0585588649932653,-0.0560459176542187,-0.0662162666533497,0.0163526473804879,-0.176574317119339,-0.0880472148810299,-0.113747527082654,-0.0919015510778953,0.0192443368122995,0.0137723685678158,-0.0117546755913785,-0.0728021541667307,-0.156852472880274,-0.0115332967764331,0.0751412269363875,-0.0452920501078715,-0.0227332823747447,-0.0140169239225095,-0.0201765889548752,-0.0585127542159794,0.013506112726131,-0.0957180221611773],
[0.0489751216744185,-0.136664458747602,0.185306792594229,-0.244756122995842,-0.0739922394452184,0.451875575654497,-0.0332732244126398,0.0111706375951294,-0.234713364009448,0.204916561727824,0.0599081780754032,0.0417500868183732,0.0653163749638813,0.0820556429483971,-0.0867013272619907,0.0579744753009347,-0.303782836961865,-0.153717532461583,0.306589649319387,-0.0569315559714596,-0.0218833555970216,0.233472371434747,-0.421212914032364,0.0299994815164673,0.178459102783982,0.246352505959699,0.13830453609543,-0.703044889300105,-0.338131284495539,0.0580322606896712,-0.0256856160484408,0.084603435199449,-0.0203886531897878,0.106705899934526,-0.161161264017364,-0.0691987119367789,-0.0164086026418462,0.117044761070143,0.0675668779964938,-0.0114630570338491,-1.03900660428364,0.0838369123055074,0.257233941683455,-0.56798486408353,-0.149125825766291,0.473783012547689,0.337152015780444,-0.0279844654004208,0.0319351691321167,0.315602811674148,-0.0188467203287673,0.00665897868665915,-0.0174910055735716,0.0513309783244429,-0.013913350855897,0.0242992374916609,-0.0626372329832683,-0.285683332990104,-0.202046847655167,0.0402148938883497,-0.276955451029725,0.0822940205727133,0.0747124086530868,0.11114860093105,-0.128365246979183,0.225331692223167,-0.266704266887681,0.0634376191481439,-0.010347370212103,0.130872382864897,-0.128393693716542,-0.325290925944833,-0.00463712655705187,-0.00886303131652002,-0.254478326724975,-0.124718935125565,-0.209446468986328,0.138210232549751,-0.00595670957313486,0.399338621061781,-0.0479020344303346,0.132408712491081,-0.074045021039059,-0.0793530356541301,-0.111664856018297,-0.346032644927697,0.0827995877172357,-0.374005535070333,-0.00987722039669126,-0.0256289161818746,0.0659473624811637,-0.195539597684634,-0.0415761240150825,-0.193980939305053,-0.0683172691555905,-0.134653971680813,-0.0865686210939574,0.127691123431926,-0.102355572581031,0.396266379811311,-0.0164805700007983,0.263750710809107,0.277761774959465,0.0733539483413715,0.238386084425225,0.10295835023153,-0.0918949056398673,0.0780455412204259,0.05186355552707,0.321404939673898,-0.137417587604623,-0.102631125256333,0.0359210261125797,0.244334172553441,-0.153503832909356,-0.136972292020792,0.0256210662570589,0.074525702999865,-0.103290595875848,0.181835012945054,0.01345917328007,0.0354421653186016,-0.045163054168603,-0.0251689167737942,0.00073200720016149,-0.0978064250609631,0.16476109231901,0.607042110845106,0.36388967593265,0.226153973540522,-0.0300403721414481,0.017840662520038,0.14491145861506,0.301118774736843,0.0777939222672196,0.0819002869059745,-0.0692719809896159,0.497839367759978,0.357422815263034,0.067730114719795,-0.162526862105745,-0.047166764887424,0.0356295502394175,0.224037584781016,-0.45065077119852,0.0247397302991893,0.0862879054844313,0.0691453280345882,0.285205872252272,0.0664979970235942,-0.212097529803226,-0.0298820075584829,0.100847919196115,-0.0735419279390666,0.0552121054361684,-0.10948069266075,0.0438807659184492,-0.134165591896232,0.302858138136576,-0.126038327425025,0.155786965229061,-0.299182956821834,-0.199957796172008,-0.0908401379978838,-0.0242348897009861,-0.101392884745375,0.0887255793141301,-0.0127932367561354,0.170934822497592,-0.130841084434847,-0.0382286979268773,-0.0633807536991744,-0.159340493051857,-0.175263289988788,0.224444169034974,-0.044522152575794,-0.0153418406865306,0.183529599132269,-0.0244459554970966,0.031364055263336,-0.227189978834433,-0.0525545080429773,-0.0144039105478078,-0.0990544004288102,-0.045357285562838,-0.121623957038681,-0.0123897020354296,-0.0618945482634803,0.0109312808324574,0.0066840191295598,0.118394971925739,-0.107928345766757,0.277566468494517,0.190724423214289,0.220891731660182,0.198170923000891,-0.00236972981470386,-0.0305474561097197,-0.143079963338444,-0.177943670518043,-0.401745013399411,-0.593085069982885,-0.438911126544746,-0.124642094746893,0.103013464067431,-0.128863461876127,0.141038585962762,0.187141122145825,0.120355507995164,-0.412607774134625,0.25032532566387,0.0638109732313432,0.334730285000956,0.890708608908396,0.0624901148783475,-0.0122536576435291,1.26852492124044,-0.129106200160141,-0.0393346589199141,0.0549378478671582,-2.09371368850448,0.424908416616185,-0.485860377579544,-0.174413042158772,0.44988356734021,-0.229501421970635,0.198508820185578,-0.186335470024004,0.176067470517497,0.0302568195859748,-0.0244837678870493,0.0496718429728052,-0.507957604952789,-0.191574201977638,-0.0622430658612092,-0.506699514971593,-0.396172853352868,-0.340006185932204,-0.454871190541052,-0.838722144012047,0.0809442439340629,-0.301472765611059,0.0525413984973248,0.0669887052815262,0.025098326287758,0.00955715885103467,0.040492362243108,-0.0039402150191379,-0.0996078721540238,0.123657135523334,0.200806064921627,0.178316563834173,0.140384512736729,-0.0494348605973537,0.16120610147524,-0.00354373456513318,-0.000482277649878226,-0.260594013990648,0.0856420981714565,0.0528667535554735,0.0855617156141643,0.0852745876014338,0.107527930486132,0.0349211800735785,0.0341145270200458,-0.240981098785752,0.176183965647557,0.172073806715788,-0.0899692194199092,-0.0425114854412186,0.442495799827456,0.342588765417942,0.698526554206866,0.4333148887693,-0.106557950797888,0.130984842372076,0.0372086162461459,-0.0250043518107053,0.0815774858582814,0.370428209135255,-0.00903260460600813,0.281357424431107,-0.0344027385911979,-0.119182399449651,0.00402478074560974,-0.246360594801341,0.248914245967533,-0.147284417892494,0.203310876294043,-0.170610296999143],
[-0.0341859320358296,-0.166453859096226,0.177832620311846,-0.186619816930165,0.039023531699181,-0.382997328052685,0.182392671574126,0.0501818604912139,0.0743827259368643,0.224702332989587,-0.0159545927780265,0.0129898211168169,-0.00521772392032767,0.0171019146471735,0.319359062413505,0.157297141962192,-0.313958667775416,0.0155756484125819,-0.198443500019192,-0.0763886567232222,-0.0541589889065623,0.151452821580187,-0.499611038564541,0.0309672304526895,0.0272913356119413,0.347155831266784,-0.247114433460571,-1.66806714873933,-0.104509355746637,-0.584099322712257,-0.0377068082507819,0.0612376037103965,-0.143226174197119,-0.0225888520983207,-0.416843967613783,-0.0455785595485063,-0.185058121476161,0.0960430565023348,-0.0679497210543426,0.163730106361514,-0.991792875197327,0.218283102326247,0.125204569682091,-0.0354293444992593,-0.242963505715622,-0.178181225325304,0.00682540258923195,-0.151701916992127,0.24599278750193,0.275252268286567,0.0543605910104903,-0.0853658504396117,-0.114948710034696,-0.00169998362035229,-0.0172925804862715,-0.0509169742581899,0.0588850434164753,-0.213681105576866,0.0976562351319466,-0.0338050540991361,-0.380327103450649,-0.00895147323852396,-0.0146437373064433,0.39322378016794,-0.0421104021448363,0.0925079137009614,0.105522023222924,-0.686250270811763,0.084201989320381,-0.698163800679898,-0.249273341655562,-0.499358093058934,-0.214175110225981,-0.241235220604088,-0.366891429913576,-0.35995112149606,0.166564967510529,-0.11658474148883,-0.100955385852648,-0.293948541012923,-0.049657573147062,-0.165959200285211,-0.208984441700048,-0.221008433517661,-0.225082171118923,-0.739738376154222,0.115599638062324,-0.496426051623527,-0.0580380169615597,-0.133396906951683,0.051864356991275,-0.497535088313572,0.0417265729121778,-0.112216765211409,-0.0803312058859144,-0.0321142527240413,-0.0742710992120908,-0.143343566713227,-0.224153722067586,0.111649261633372,0.0900501608938288,-0.298419226998168,0.144778701412084,0.0257302469540202,-0.215967236746931,-0.19388040090224,-0.194223770172595,-0.167433172511143,-0.0583003293554559,-0.482659535826564,0.0142479940459267,-0.00641162574257842,-0.0238961581302661,0.171836219943475,-0.137636746262236,-0.097427238668554,-0.052316491379417,-0.15660335942765,-0.250917151444778,0.098929187963939,0.270085240602189,-0.201781715280179,-0.144727606699224,-0.280560627486347,-0.155048244200594,-0.13298775939642,-0.0953386117863884,-0.239007751849755,0.19180320378476,0.359681592013201,0.0746739649728269,-0.129177075011386,-0.125817360066861,0.200154528137597,-0.0752614659942295,0.286386944144398,0.421964964463805,-0.463093614348227,-0.218058177209287,0.103850806168113,-0.112955342244088,-0.178841063467088,0.361174442118859,0.332364613907123,-0.546190196358039,-0.0141526604043318,-0.0670537306674322,0.62944036236559,0.769265447124937,0.00628460556701124,-0.351178466071733,0.374302702505808,-0.0835231847127517,-0.375221259722233,0.041676244690527,0.471381909882442,-0.0340292724030449,-0.458888726595729,0.0247679419100649,-0.350539061517722,-0.059158230405744,-0.541098535475342,-0.152282151768323,-0.219304994611672,-0.0102435453089704,-0.25298648805623,-0.0751497394445023,-0.0748583780367384,0.100160420504458,-0.162218233267985,0.00111948416177512,0.317980732691764,-0.162124137894822,-0.245916413924898,0.261970735839601,-0.0447937367981561,-0.0587315190724711,0.0677974157797323,-0.0774860825717419,-0.0553402149322173,0.241524387149894,-0.0936363588360574,-0.0215708377550175,-0.0138898100802321,-0.0594547005026603,-0.0595131521126508,-0.0302172175946154,-0.0822761334561341,-0.0487509885701294,0.0102110303188448,0.164261221570872,-0.361162233880724,0.0477090494916513,0.0404957691959995,-0.0235430204009007,-0.00949074027718388,-0.172949829545049,-0.225411843355254,-0.0140146214352217,-0.16159809682162,-0.636052112477913,-0.971795019034829,-0.619993057765283,-0.123298133317299,0.0155336048124556,-0.17462804028969,-0.0765742889864222,0.126210277538266,0.0707998680193817,-0.676228917719591,0.651551142585467,-0.0485284970216638,-0.640320701786751,-0.19566396649361,-0.439422300084048,-0.372217770553015,-0.209094699111199,0.170649576935189,-0.00245050450468675,-0.0131348333159719,-4.55804535338302,0.293553977144064,0.308393605137413,-0.470980705201374,0.395558782178208,-0.388804724182833,0.0960614389843653,-0.372002555813593,-0.000532097038232521,-0.110579802669982,-0.193218651485619,-0.23501449425107,-0.649886042052758,-0.367347071831369,-0.38895285592382,-0.936824826085133,-1.44024132041456,-1.31455208049573,-1.42502569276697,0.178164270114226,0.0727831716372191,0.239177067530973,0.0297954195144791,0.24412126826493,0.192639689002063,0.196467138411648,0.0386040264408471,-0.148562214560614,0.0673232155019767,0.14030799626679,-0.106911269210681,0.0674208788794023,0.0967555820224349,-0.0770989157064459,-0.00687964624983837,0.106541997581804,0.143462892552217,0.34727358959748,-0.0171559745590598,-0.145850912336974,-0.0439066001911484,0.00284081884181515,0.174219776982624,0.0540770007768101,-0.0484463654116477,-0.0598788352400167,0.158071429371892,0.246832626429202,-0.242009305360781,-0.123354899244475,0.18746762291834,0.122945219886539,-0.330315445377641,-0.173560903973049,-0.248025863998562,-0.0325542577781994,0.127544478329184,-0.0612405205939869,-0.03158735162142,-0.275041663728989,-0.252656905717826,0.043793061283012,0.101546077142116,0.131703159819653,0.0772916419287297,-0.138431182100204,-0.102379613946349,0.0645060113452775,-2.36448357553737E-05,-0.0146698883399783],
[-0.0149673450965799,0.118414963258812,0.0855422778294941,0.479661028274147,-0.0725346502906039,-0.125816765350711,-0.115909509522196,0.0191589825434864,0.438824980669208,-0.0376282241321778,0.0216747350030629,-0.109693227287617,0.00901159586296224,-0.0637276711817635,-0.178707090273898,0.141778933132411,-0.0359111687413643,-0.0223333569347655,-0.0212913225590089,-0.0704246264721422,-0.0520139886191059,0.145638342418352,-0.335655115122434,0.0201171377873193,0.00108866176335999,-0.0464834580106491,0.390579479892532,0.0786782215586337,-0.242454882407147,-0.550579321883521,-0.0150115597424643,-0.0850370959013245,-0.0309573209121871,0.0155989639703578,-0.163344841439303,-0.027516474930989,0.00799160536109785,-0.027987931080002,0.130249877853126,-0.207125423140029,-0.558435139006688,-0.165857752455585,0.385803744006186,0.151515772117982,-0.114790000474984,-0.0613178622160611,-0.105123810504473,0.0842383846130157,-0.0732635042971159,0.00046755824359476,0.0514193387435633,-0.0382144903192715,-0.114511851499403,0.0176400336161553,0.100494743648935,0.0646896683003594,0.117917838680092,-0.147767647491618,0.114221420985217,0.0729469544607606,-0.28569823090767,0.0411529609033618,0.0232860819850952,0.267395690152918,0.112230347555698,0.0620367009302095,-0.156579710941928,-0.135579806083451,-0.0136283995306267,0.298839006300477,0.0726010794403094,-0.210146938891688,-0.0350748963562213,0.36291238736648,0.0513529239984128,-0.142993349515449,-0.0159604526351535,-0.125032860727883,-0.0295476062509097,-0.231618522519192,-0.0411939293290482,-0.159020944483008,-0.0681140363569421,-0.213200276963254,-0.170701488570891,-0.180228787221761,-0.0647295402744762,-0.130517327758449,-0.0315502262807437,-0.113908301165374,0.00379676811049184,-0.121962563981564,0.0699773932043522,-0.186619859733263,-0.138484974517389,-0.0667688083890342,-0.073811388387139,-0.32195157040752,-0.107251505795603,-0.0991633302106383,-0.00494366960747628,-0.358652507581431,0.0853411131826362,-0.0221877713862022,-0.0775702631728646,-0.226854865268436,-0.0295036957794686,-0.159200697799107,0.0333239509295856,-0.286061990611679,0.0994237545854927,-0.145154866596581,-0.00516765803538517,-0.0235720805459185,-0.122342025564739,-0.118514762348147,0.0338345815344195,-0.183481315857338,-0.0397361592666182,-0.0431057157187992,0.0880943859490521,-0.11965650291907,0.0247404643084636,-0.129673377995589,0.0279643663031194,-0.254862085583255,-0.0542413398035681,-0.0414521572260865,0.0398281946561997,0.398187431741419,-0.00527384854187892,-0.105576716912796,-0.116649213204892,0.00170223610759126,0.0112928424835704,-0.0163832006244273,-0.0411033749027746,-0.12603496898198,0.108163449021689,-0.0845063332978836,-0.169781397989275,-0.0296392745279719,0.0723782084627129,0.139475458148134,0.400678042369841,-0.121438512405948,-0.00316608166624768,0.0578692811403793,-0.0753969498745067,0.0644424032990226,-0.255059844775136,-0.13764118809582,0.126917117792793,0.0328218747001804,0.000935049369991171,-0.0817332109484766,0.0169909990680392,-0.189195316082455,-0.0858984780388191,-0.154406345101506,0.0215965670529938,-0.337026829448304,-0.292206036054911,-0.0544561467883948,-0.0175135299478409,-0.0710576603284181,-0.120491697670084,0.0174818898353658,0.0307626520916533,-0.0976630512697488,0.00742222742895144,-0.233648580917171,-0.154632223388683,-0.177211619543299,0.0640964027351152,-0.0426639459286316,-0.0548925103034108,0.0393225231289702,-0.0552352483299649,0.00826719581613668,-0.0604926783694895,-0.0749819921278573,-0.0559697282811949,0.0761567184450917,-0.0383164713553928,-0.020112875507779,0.00681115605802858,-0.0602069437167746,-0.0140502996827122,0.0131513963268811,-0.0682578364385922,-0.130368742043275,0.0167930006997045,-0.0680661707967766,0.088802722355809,-0.0848612165399207,-0.0275142601358416,-0.110359965210996,0.0920190022870947,0.00379539647197297,-0.528081803157623,-0.733394033649658,-0.447963378981476,-0.170125251427112,-0.035565598198405,-0.204004636108827,-0.0158682966385858,0.0866268652662339,0.0565534288535641,-0.528894898449665,-0.107970840091201,-0.0916111019150781,-0.31465553389494,0.143877960059609,-0.150782464793815,-0.116460133183473,-0.305191315027385,-0.317837293444773,-0.0482009512631429,-0.00951456545322434,-3.7001794328098,0.263597155849879,0.165646159316382,-0.131630317236357,0.395777995021527,-0.115193655928337,0.198966027874351,-0.127159923856399,0.172381285676446,-0.291955043547028,-0.182997675260372,-0.149318256553784,-0.481339239786549,-0.264568426911201,-0.241379703755964,-0.652233335603069,-1.01411257713529,-0.890214127844524,-1.03225717244058,-0.613157808541124,0.0207298263003506,-0.126202621698149,0.0612330261082945,-0.0361038287815406,-0.0460192165338972,0.0294027364716408,0.0923173188760639,0.0511323308395047,0.0309023509941043,0.0135581184018024,-0.0955609216724594,0.0278999942915587,0.032146023960803,0.0287784130970704,0.119996429937765,0.0849457610677068,-0.00285115684306634,-0.0911778435865295,-0.0405870896586769,-0.0234787257640167,0.0130853047030495,0.14991915200082,0.0796245472701047,0.0346790021548583,-0.0267279978210939,0.061686007442007,0.0265483070011772,0.103748446153293,-0.155552392569719,-0.00721832593690153,0.188667944886072,-0.0496218859467009,0.337966267898282,0.0719086298513342,-0.150232459644301,-0.0718118796757215,0.0430271326042604,-0.127293949157481,0.0323439557097137,-0.0182081636234388,-0.183645263134887,0.0501701612910205,0.153899135776197,-0.17684907729275,-0.048080236030142,-0.171940903741122,0.126003252885595,-0.188310730099258,0.0452755183539259,-0.128621663479687],
[0.0429856963079168,0.0341379753176152,0.104187329000071,-0.0540560887555806,0.111593681585156,0.0491660427349303,-0.0108658368155245,-0.000763604205609744,0.0135857098950277,-0.0911497033673496,0.00956917310153569,-0.0746258230646621,-0.0145234330108956,-0.0334546279581346,0.145272398357176,-0.144865160405475,-0.175109236743841,-0.128065462042483,0.0985282452157917,0.0854676950902907,0.0507501847750584,0.0996791184817792,-0.195536337369292,0.0875012880440736,0.151064321819878,-0.137753511849312,0.0719394499388608,-0.452283129408095,-0.306559376901225,-0.349966129299088,-0.0396632470537196,0.0361265749773239,-0.0162691012177186,-0.141998865261904,-0.0914809212287626,-0.0016666722256091,0.0141740375488338,-0.00572967384018368,-0.21920164706993,-0.242352366030079,0.189857371677204,-0.170947866354068,0.207719152332363,-0.078736323393312,0.124426375689466,0.154511886916689,0.17354525422992,0.0413108230400475,0.0145621819118758,-0.154239892876862,-0.0315262655274848,-0.20875033857676,0.0206439654448911,0.0871307527252864,0.13962726104698,-0.0450006131530473,0.0468085535608459,-0.127977535886266,-0.0531719359920551,0.0427759851356967,-0.1865656692497,0.0372543200860465,0.0984555117759027,-0.0505482214062773,0.0310281184019382,-0.0130653362570068,-0.260212998697198,-0.127252950792501,-0.362816067742769,-0.100785720432964,-0.114712428107327,-0.375351137913956,-0.0283866134785329,0.0192196403251894,-0.223472290382651,-0.317099187173486,-0.248113570157039,0.00636334429558119,-0.121175724149803,0.176442078354481,-0.1249500399069,-0.0532071119551077,-0.11526911392374,-0.107885361901444,-0.166851707695976,-0.422294210112313,0.105253417758491,-0.332604805084398,-0.0383268715966927,-0.129801831274453,0.0712350632066901,-0.181395145883481,-0.171864104140727,-0.038970509789184,-0.12413444253775,-0.0490511934331114,-0.0620049625858992,-0.0262640887323744,-0.0412323386996603,0.215602939227798,0.240445352952316,-0.153740407044727,0.175820110499574,0.13436060989856,-0.129682326857342,-0.102900216711696,-0.16999382703467,-0.0547335394546491,-0.111144734417753,0.367417607229664,-0.185399156683888,-0.100218977589603,-0.0119387886331118,0.198159452374855,0.149775206792244,-0.0602847299775275,0.0388553116171718,-0.073692254785173,-0.147556236710244,0.114242461794701,0.283744437059055,-0.0769590190169483,-0.0576168898174504,-0.169685415516134,-0.0862554151480591,-0.169875854703811,-0.104714765070103,0.161297405453729,-0.064322860456385,0.467015989288938,0.133115391810621,-0.00433375825701312,-0.0470961433344673,0.0605449558772488,0.112103013330021,0.107110189173896,0.17358019832048,0.13027327312505,0.157325392655806,-0.155141602900315,-0.214155202222584,0.116617472357302,0.0213200625926264,0.0970409000003608,-0.277999409974156,-0.0768988263401697,0.0433685919985768,0.0152183834432091,-0.0521574085054638,0.00876722308432814,-0.270501840995422,-0.0407477616563976,-0.101692855405169,0.0508154814487189,0.101658497401988,0.188224604280125,0.0141597365226124,-0.193352053004318,-0.157516849836725,-0.150941978765928,0.0139845295241307,-0.212766685715873,-0.200145141575554,-0.0215839556325606,0.160451751311114,-0.153995304908442,0.000321217632458123,-0.0458167004931924,0.117137503330678,-0.0285497995402014,-0.00329457658756008,-0.0740967609893459,-0.09205093045356,-0.0907423794145593,-0.0237204094081146,-0.0128081525627012,0.0139815976845941,0.0509117237689145,-0.055397321916161,-0.0517855704625605,-0.0171854967207269,-0.0513757544396877,0.0250049130042517,0.0660996135697829,-0.0500864411756374,-0.0117256188592566,-0.033376906287724,0.0251614986915401,-0.0185910727989306,-0.0316805316320019,-0.0227045694755384,-0.116922972687486,-0.0765582439005864,-0.0940518076453798,-0.0657560472399258,-0.0566030604820743,-0.0938599218060548,-0.192992452952368,0.112844572037902,0.0666421715455489,-0.655011346123237,-1.18147092803865,-0.612049380836868,-0.201722291144516,0.0304845611394994,-0.237194155527244,0.0559781427664727,0.151024041606416,0.0794961543771298,-0.63190233351758,0.07804301683185,0.0384453294501001,0.039210308825205,0.819340536972252,-0.195136002670706,-0.210557518847141,0.887706812515136,-0.207961797048805,-0.042843000583702,-0.00305527043366099,-2.97550214157438,0.0545704715035975,0.108456290067876,-0.38485878069562,0.237817165061908,-0.143689882096448,0.273777494822154,-0.110343884218037,0.201306926783898,-0.10024191661183,0.21203617705146,0.203950207477081,-0.15326529906428,-0.00230530763990862,0.0269427809460201,-0.34434296975536,-0.538601332283731,-0.564161991916071,-0.752897412681571,-0.0172855921615636,0.0408916369190985,0.175101839808915,0.0200355651596935,-0.03146055291488,0.0195964375004122,0.0616226856989567,-0.00986433339107544,0.0368724300285022,0.0190599841388524,0.037278853580043,-0.185371706308632,-0.048052902146352,0.0019959894449717,0.0324766066387927,-0.182183310925321,-0.0562384243127402,0.052375361832888,0.19037716930774,0.0385126981705738,-0.00501812149339895,0.0131382002109285,0.0159787337096252,0.0782517115680962,0.0295516808867544,0.0446985251972352,0.081600056854927,0.0585172907751032,0.0445010232393596,-0.207632657323596,-0.0404861146222375,-0.129013064213631,0.0751281909216454,-0.455423599487585,-0.263206554759384,-0.136960039916867,-0.0289076684821615,0.0494501414887927,0.0833133610031089,-0.169836898571887,-0.171060918789788,-0.0628109229008694,-0.0264281588848753,0.14668095665569,0.0673643728887361,-0.136379479964089,0.182070553473812,-0.00603650271673501,-0.0229198449984439,-0.00745148423013477,0.0287422368906099],
[0.0418395182216694,0.0539677878941668,0.0704884715361115,0.253270089349498,-0.0527777760695928,-0.160094064990812,0.0680608583492147,0.0183567370598119,0.235206691356847,0.0412152744224689,-0.0132735879332445,-0.0912297408656752,0.0720203938759947,-0.0362913677287089,-0.0231530050131113,0.0859560760264953,-0.146110466503334,-0.0436916003957853,-0.00974093592929351,-0.127127084705319,-0.0716483924156783,0.0893761218259875,-0.30635604415466,-0.0879549695390697,0.00487789117473761,0.171905249972846,0.129898089111422,-0.566458990372042,-0.283212591717011,-0.509705167124347,-0.0765918865631472,-0.0668711393570288,-0.0534469531046067,-0.0395673936504888,-0.242629424526413,0.112501588888776,0.215004789751131,-0.14701974186824,0.0202082340797658,-0.236500572639082,-0.224773459560828,0.0255131227299621,0.337849770337183,0.105498766317096,-0.0595477270099184,0.0108240951280598,0.169756543322202,0.0887134265895232,-0.0597622541142172,0.155588846799726,-0.00899738490151943,-0.114190420846447,-0.123548274670062,-0.0207615347340747,0.152272001760891,0.0623836821023355,0.103378375909642,-0.167151458812348,0.0318370175419387,-0.0147783428772533,-0.324716195062272,0.0817557779557924,0.0118121173019309,0.0690803584421277,-0.0572835550173503,0.0730086330931838,-0.216029363057808,-0.316613543166011,-0.145569706050955,-0.0783740748743729,-0.0854769701984269,-0.373030924518898,-0.0224959664107404,0.150229854271044,-0.126842452082074,-0.229554219509979,-0.0226498763894122,-0.146582203508272,-0.0259072782076647,-0.137771173296724,-0.0814965073645674,-0.0986167747652636,-0.144016596257094,-0.240213910496143,-0.128603899890455,-0.368075386871633,-0.0111405146762583,-0.349054545942793,-0.112651695253338,-0.12979590917339,-0.015922144440185,-0.263318055661652,0.00749496949583069,-0.0720954907013745,-0.134277230132806,-0.0101925436148162,-0.0396748831211134,-0.239744513556651,-0.214667865672646,0.00339269575875603,0.0132299742954997,-0.409746845387318,0.0342128441959024,0.0685413117212865,-0.190878855449375,-0.211732788719977,-0.121703160994392,-0.124308964334297,-0.0748434655701954,-0.221107365178506,-0.0444594603678771,-0.075052821876341,0.00367390562633622,-0.00296015796132193,-0.0908982041537109,-0.0991885375764459,-0.0112769100548489,-0.126413315227684,-0.141313274211586,-0.0150165589974406,0.0783638825825796,-0.159495147112028,-0.072071515203132,-0.197919973493106,-0.0312988997733246,-0.262842630219562,-0.0595830562998307,0.0267965335997988,0.0142676651777685,0.571359863521214,0.0725642126898994,-0.0384974493678532,0.0692707393791211,0.0141852395513211,0.116396069387257,0.0922033205621035,0.0957311658032822,-0.250513726145184,0.16746571819922,-0.104810646380373,-0.182811174256629,-0.0474061109852668,0.0226308094383517,0.154380944446276,-0.147594761200289,-0.126243475649106,-0.195716309548262,0.220181274029592,0.0460597045918305,0.0926864303546059,-0.314047695219475,-0.167604857825409,-0.00948045148371619,0.240586174040806,-0.0393252229534182,0.0716248898606871,0.0881075929554524,-0.153181888455041,-0.0548643518430686,-0.153762922426449,0.0106396269960424,-0.334368803966097,-0.3856518734306,-0.094072108338298,0.0746013754634081,-0.147792836356887,-0.0622546797108528,-0.0481526360740183,0.109152477537365,-0.0580353833444615,0.0664249454211583,-0.26391042224421,-0.145570086559771,-0.191627831564754,0.101642346493955,0.0210795234797132,-0.0510290833185237,-0.0855980641865809,-0.0547610041405941,-0.0656736318871369,-0.114729387986068,-0.0743410131591885,-0.0384808224796129,0.00714254224117501,-0.0210832296441078,0.00333684563680666,-0.0451621760419373,-0.0314265147324299,-0.0399954027534742,0.124891389711186,-0.0510384459426715,-0.110159903874541,0.147681496532485,-0.0344992914322467,0.132498856143861,-0.0932764295635222,0.0668574806161953,-0.0200699210441256,0.119364485948579,-0.00739388219948636,-0.60213592516931,-0.976091075403919,-0.523756779227515,-0.199618999660785,0.0475332977212759,-0.270238110753171,0.00770005759560481,0.0619952051530628,0.00600439171932924,-0.628465645207458,0.106755967183454,-0.0550492817379112,-0.464001268046691,0.135116820395167,-0.294143387498524,-0.287834484298743,-0.0467867858295691,-0.105341435370675,-0.00959046334241936,-0.0360808787361819,-4.13838615461444,0.206258056054714,0.316189559819174,-0.345657592395019,0.359141756350443,-0.119889923625333,0.250280762995945,-0.188575081704172,0.197496690925064,-0.213202975710286,-0.145534001630434,-0.108586157655183,-0.394744062277412,-0.219096464348656,-0.226772362879071,-0.666712433527573,-0.980054147301185,-0.859152198604954,-1.11675372897933,-0.206380325323673,0.162612748891798,0.152289556405609,0.108651105407025,0.0800981367259612,-0.0207635820637903,0.054320716564791,0.0734479454679014,0.0758241044071691,0.0493942571048642,0.0685371270007392,-0.0942820870623217,-0.0104208156309726,-0.175808652794943,-0.0697722866745686,0.00788074155491284,-0.00192725772497904,-0.00722156051940349,-0.0528419459038044,-0.0331011455338739,0.0881344354849515,0.00248542512270855,0.0710112392390088,0.103233852487052,0.047205786176847,-0.0108974135433614,0.170748531252454,-0.091961959837469,0.165983065706972,-0.218980034479066,-0.062314882607739,0.0822325118452014,0.126025536625382,0.146410416687406,-0.0704550162246176,-0.0127182453902927,-0.0527917495201053,0.0710266055019727,-0.0314493610831625,0.060543359746202,-0.0765611999397179,-0.121666495373754,0.0902900722847609,0.218566124407843,-0.0585963037089828,0.111655427407563,-0.0541656395042396,0.0117408614204992,-0.155370345594348,0.00634583962074682,-0.0993910522782523],
[0.0486221130981147,-0.0319519951361722,0.0701431790537889,0.0425528606987658,-0.0414910124355915,0.0894980973729545,-0.129367804664475,-0.0155850029515956,-0.0468892942843483,-0.065781836281968,0.0504950776381188,-0.0160476754964882,0.0518569902944944,-0.00303042311432315,-0.0815759012713282,-0.00294374728387688,-0.0665053273540125,-0.00747118007869838,0.0558087166845027,-0.0571366613989246,-0.104354736434502,0.0629518385929023,-0.174808536055484,-0.0520285596540406,-0.0402403048814766,-0.266053513695635,0.132722426859959,-0.134005360822898,0.0148580454498222,-0.112088756009903,-0.040191764745228,0.0232543272515348,-0.0563819045516623,-0.0860448009192226,-0.0994497728616759,-0.0435536078848447,-0.0386427028444857,0.0160405027300667,-0.014027426057345,-0.123478626528915,-0.097121506608996,-0.158914834884269,0.0745152678616854,-0.0147974979191227,0.00189802417214045,0.0650101524624048,-0.0830542706280747,-0.0273836178791661,0.0186049626494284,-0.0590680871185228,-0.0142538889710064,-0.0592804945362539,-0.0275315523192912,-0.000518011195657516,-0.0103473552230395,-0.0423762899809349,-0.0323741785895226,-0.078257580815008,0.0438221496627124,-0.0434017805924791,-0.0518803442046318,0.0389612071024494,-0.0736320849461747,-0.0286455658429798,-0.12986408903174,0.0427352247427309,-0.0501242440622281,-0.0139766674703162,-0.117207279928616,0.040754575008124,0.0173250905293558,-0.0747058577407556,-0.0436672651933019,0.027682182684933,-0.0111858262252562,-0.0681028383873707,-0.105247590644024,-0.059656927261478,-0.000440987423312425,0.0278199377602998,-0.0534905217038044,-0.030775482000734,-0.039850840035582,-0.0628721490993308,-0.0336375319328593,-0.0574642909692179,-0.0202207585657429,-0.124735360719431,0.0574602081522491,-0.033074428854222,-0.00508216056775974,-0.0481452793220062,0.0146221792213584,-0.0313461542384663,-0.0870267309054813,-0.0109449783457122,-0.065786010746218,-0.0568718514494297,-0.0472236906906336,0.00573456304482181,-0.0286745240870631,-0.0119939616927353,0.0829857891048396,-0.00691942704557205,0.0345121208186924,-0.0230212097064519,-0.04848170609743,0.000313867754738831,-0.0543739041517338,0.00172599453925289,-0.101700539692128,-0.0193092638405128,0.0316563989459954,0.0756131681516531,0.0113959392543008,-0.0594991424060348,0.00800983832777598,-0.0111823211523734,-0.0618150112752406,-0.00177326079192187,0.0272931789947639,-0.0742719252237154,0.0302345220616172,-0.0117197332365901,-0.036862726848968,-0.0268270789959954,-0.0460008849916358,-0.0181487874941753,0.0352553770948755,0.0842580726403887,0.00167959606412621,-0.0052307867605392,-0.00863098118632784,0.0199777392112026,0.0271158116279329,-0.00330808206115345,-0.00109717583591054,0.027073344722318,0.0268081398142622,-0.0718458124160259,-0.0399710996698263,-0.0844714837661232,-0.021498864379589,0.0887507919479395,0.086620227766748,-0.00488532870163651,0.0314897227981489,0.0106603436592242,-0.0427479734139357,0.00512801301597814,-0.103660100782374,0.0169602308456283,0.0290117354883675,-0.103096409893669,0.0731257325442933,0.0780976139247095,0.0694469485416233,-0.114355265296508,0.0755251832949379,-0.0768247325484621,0.0610168562647838,-0.132536742573968,-0.0857399000603813,-0.0711369232566947,0.0139198096282275,-0.0753862867598586,-0.0264612532480521,-0.0495888687626947,-0.00734031708904834,-0.0819215656841314,0.022056820204271,-0.0221469823609148,-0.0856482886424407,-0.098742836842642,0.0736263202371416,0.0395192651207863,0.0362803578831114,0.00908014663983018,-0.0573222804575678,0.0377460401475914,0.00915741023439956,-0.0695013523685122,0.0459012402066981,-0.0927551101893922,-0.00304321811584469,-0.0383749790476026,-0.059469536080361,0.00694060884665148,-0.00751888030737617,0.0128791843196462,0.0468317508945277,-0.0416746004022668,-0.0428402310791057,-0.0197477753162428,-0.0275989332300287,0.00161142404655039,-0.0523630955612362,-0.0726549031318389,-0.0563942931889971,-0.0329269044415322,-0.211736750910255,-0.350199934593362,-0.188422638206498,-0.11491525118869,0.0299564358128546,-0.0532265025969523,-0.0141641236035606,0.0725716971283237,0.00454636631901514,-0.168315949190597,0.0750770507273274,0.000850442766305618,0.0608150763838215,0.282238616692548,-0.0674365370119834,-0.034481839107834,0.0979440331466323,-0.144932878347894,-0.00658459465511467,-0.016218871068848,-0.982784021120686,-0.0330552290213236,-0.00954045177841309,-0.0804052690403927,0.0918368372048278,-0.0765308440818761,0.0157133809754752,-0.0161613131718529,0.0771582662159966,-0.133612376891169,0.0216781487573389,0.0281105820915467,-0.146193030063552,-0.0357654365326271,-0.0324424348662591,-0.179254388285815,-0.215946494968202,-0.154716869064695,-0.238398048635166,-0.183341410562365,-0.129776437027971,-0.0702124419913276,0.0112863938565505,-0.0260432465954168,0.0122179616991119,-0.0201978526552419,-0.0547153882679409,-0.0216780119336803,-0.0988652029126328,0.0593600742941426,-0.0433176184374554,-0.00355388005463069,0.0261391410401152,0.0168347832864286,-0.0557840327974931,0.0547318180019669,0.0594838045368733,-0.124375361335438,0.0530979397688395,-0.0336471947674053,0.042908849416519,-0.0313417339897276,0.0494108262570258,-0.0423642025304541,-0.0352235475596375,-0.118631536420632,-0.0183196335704358,0.0263666678359306,-0.0694480302774265,-0.0437847455664147,-0.0440716341947227,-0.00267322016897827,-0.0413691142988266,0.000524324719211932,-0.104103138502183,-0.0446451481169728,0.0301292316622384,-0.0372474595823597,-0.0145933317835625,-0.0429520624450098,-0.139507662111944,0.0414612819448258,0.0720094758161867,-0.0115355425065747,0.0137759792371746,-0.0186593438542595,-0.019350244744254,-0.0918032628981557,0.0384407691668581,-0.054359404823591],
[-0.0254783191568187,-0.0258728188501489,-0.00791601686711203,-0.18528380163701,-0.0993060593397819,-0.0945960992829363,-0.017388950237107,-0.0479751609258739,-0.118068191386599,0.00312858995398299,0.0337463718209877,-0.0720808360673873,0.056571215398503,0.0349827087377201,-0.0329636687334859,-0.0105773269015738,-0.254335800355583,-0.0246674443179802,-0.00402108135086154,-0.00945881521424505,-0.050521805539149,0.2800309960516,-0.249698960617525,0.0525085464141088,0.0290090117042598,-0.0793048659107503,0.11303296863082,-0.586875194522853,-0.0841571873102484,-0.238898162018596,-0.0782104249469101,0.0467148630645263,-0.124034915457728,0.0522658323154393,-0.0358084004103124,-0.024954101519894,-0.117458002111636,0.00495033447232696,-0.100256487120197,-0.110113982179123,-0.726159246487973,0.145664876142798,0.139447248772904,-0.137144964013046,-0.186179271012676,0.031772483462774,-0.0191883104443026,0.0485398703789084,0.0273257276540649,0.0571639230995727,-0.0911873150830556,-0.0434939664252877,0.0264926287684659,0.0293241950904245,-0.0462459451671593,-0.137651951158672,0.0742984064289044,-0.112311021729738,-0.0563335362378122,0.099631316635353,-0.12694472690682,-0.0603185307118585,-0.0427483290474798,-0.220579811880334,0.0183636538407633,0.166280330728018,0.0333306835896764,-0.272006486176664,-0.0666420328328238,-0.141567140083611,0.0306385744301348,-0.316540928570313,-0.108096797396165,-0.0330061311364781,-0.0374273004900286,-0.177987823421511,0.0178802304150346,0.170182639388113,0.102951288359301,0.0368379047183718,0.0896679907586424,0.142915502300767,0.0268276148662954,-0.00690495498399138,-0.0523109653454562,-0.364779581615021,0.055457020338689,-0.273395910866032,-0.0829174445718159,-0.0564584204328131,0.00789323084381749,-0.291437863549927,0.0144377316666958,-0.0387973879614631,0.0832323023650106,0.0436386603455563,0.0441363808298811,0.00775016631989875,0.0477709901827278,0.139575021980896,0.0985359153020809,0.0439112451684463,0.120139460486414,0.0945853339669996,0.000420716784648692,0.121124036222305,0.105541166724717,0.0560169238989181,0.13899931780888,-0.0449795818613949,0.0353796692468111,-0.0951203088883053,0.0238051217175331,0.0588868303569757,-0.0216907235465822,-0.0606144627375087,-0.0225527688390028,0.122551695253764,0.0985784692142727,0.0860204536874962,0.179802932105599,0.101728575133403,0.0613141295604058,-0.127515039652796,-0.0295097109228633,-0.0578651662706828,-0.0234669180414058,-0.223586904447905,0.0925978223901007,0.155250015078471,-0.0326610822824386,-0.148267221446726,-0.129226023573457,0.264590754477179,-0.0408211807364754,-0.0667817263223618,0.0898971475521291,-0.152503219108765,-0.0796097348338754,-0.0274313091269757,-0.0195515696128,-0.102443539191341,0.0308833538301222,0.221832255999729,-0.124918982246067,-0.0396090330656252,-0.211346866159317,0.0907179795802486,0.155127702329051,-0.0154712655895408,-0.164427299594528,0.089152233406083,0.00830647585065301,0.0192051360241711,0.0935630594303508,0.112623698924493,-0.0888003984896797,-0.166554797727852,0.169490000707079,-0.149009034494585,0.00700048751420373,-0.207322160089699,-0.121433912702565,-0.0416426510201145,0.165703635727324,-0.125318867071678,0.108140207002314,-0.0593721897096804,0.185965517178219,-0.0341885441379728,0.135451228342191,-0.0122779433444644,-0.0762553644904082,-0.111144916518974,-0.0281775298113637,-0.0681217437259752,0.0353477070067826,0.0749233710854228,-0.0547431171113057,-0.00827590275672854,0.200784628906593,-0.0157292439839726,-0.0070136706903036,0.0673143205152067,0.0103459152506756,-0.0111115443407868,-0.0176151829751369,-0.00764628737097323,-0.055129559659587,0.0129967596963497,0.0155710694920359,-0.0608063555399479,0.0623203846419392,0.0108408317698655,0.0523998382456128,-0.0470653644399154,0.0748302563350754,0.00712154196425178,-0.0548270286457167,0.00223248674120879,-0.42457418045072,-0.729260376978966,-0.373212670635568,-0.160905181960373,-0.0130177059369164,-0.127435534069094,0.0262440340609199,0.105599874701089,0.0665595554579214,-0.459856274545308,0.142379410890065,-0.0143323743790049,-0.332036089499753,-0.0124257686368144,-0.202744781456369,-0.120352814836738,-0.06066372058913,0.113548466483653,0.016608102028484,0.0577770348028655,-2.28000156276971,0.153452756286755,-0.140654540521765,-0.352524046281174,0.124318968729309,-0.186688649455637,0.193923448156409,-0.131996614195235,0.230969339528679,0.0496028331436226,-0.0751931899366422,-0.022241224199607,-0.354862688977058,-0.169063898331433,-0.142622181445148,-0.373721866040499,-0.47520541868359,-0.49160548063257,-0.470837018007731,-0.359756919529472,-0.120427233874066,-0.0963599189691754,0.0294132095707969,0.0399153695308581,0.0185631709463187,0.0564007541408461,0.00568968981626143,-0.083128790072019,-0.0560904342492029,0.0813952617649275,0.0193397544566843,-0.00295792469964509,-0.0409081168541213,-0.030216951625947,0.0964010797670648,-0.0472314053744582,0.0291506525438276,-0.0798166466089281,-0.00484841170815323,0.150730590202843,0.0899295170741696,0.0874657804721735,0.0423098600032231,0.0460591667211511,0.0461452526286337,-0.127589036124237,-0.0903769860938502,0.143422396995076,-0.121650256926808,-0.0944163659133009,0.122490674040606,0.153642811128987,0.168987397393951,0.0140884719036171,-0.0664164971887029,0.0660689917765417,0.0796718660330237,0.0417835137874856,-0.0202921476395369,-0.0163626266352593,-0.0770288984895139,0.148008790943628,0.0914181009994801,0.0669562962261984,0.0454340270577689,-0.0429933550034616,0.0339913859253656,-0.0487019700443995,0.0163310210960368,0.0457482645124815],
[0.0452532384038351,0.0244919498546345,0.00631271665949818,0.472534495738108,-0.1696539543969,-0.159093694628322,-0.117408191040081,0.0732168327244749,0.408500882992806,-0.00644962911898748,-0.0101401539452319,-0.0977608796272525,0.0410652143213648,-0.0317481078971094,-0.155431107462658,0.0480532713513614,-0.0659590822829787,-0.0586505723342871,-0.0602573787839367,-0.097913998998907,-0.00882155751396688,0.217033061123188,-0.33132243955335,0.0656145723847053,0.067946412549678,0.151655851180404,0.330798807606766,-0.0470391141561059,-0.0504329859801636,-0.402110337778736,0.00306091543650817,-0.00336914193801451,-0.0856700922503279,0.0851137536581715,-0.227404556719899,-0.0203469770697858,-0.0975700493645308,-0.0706058535527121,0.0827079363811703,-0.151045121963334,-1.19001274009976,-0.240381360411702,0.326628682812284,0.211374074312214,-0.109511070729903,-0.0366612488100788,-0.0655125875293418,0.0313439763607618,0.0211802908439598,0.122400796419424,0.0476872494706307,-0.00243173376969697,-0.0623361759743952,0.0165615656206265,-0.043233736955014,0.0118783583732319,0.132849905064968,-0.211899569056939,0.0380103318549664,0.112644838145349,-0.297279251946398,0.0451081323391702,0.116586887544541,0.23102145837769,-0.119030716615464,0.145983288995288,-0.170956936946296,-0.114452301895394,0.0501468233648251,0.345884763340119,0.0121801144248585,-0.232720590947347,-0.00688137515316991,0.361297659560749,0.130350555964025,-0.102422246539414,0.0956591976780498,-0.0909796120493463,0.0220343244070315,-0.1880905029261,0.10276163224675,-0.124586311566591,-0.00246074771033083,-0.27876628556479,-0.125575305644229,-0.122538688977947,-0.0980685036010714,-0.151349439898758,0.0208455541186636,-0.122007931659884,0.0266991049799957,-0.159126664483016,-0.00201114808680666,-0.0913734737934356,-0.0561627936476889,-0.00814374995907796,-0.000291174578287839,-0.258334884785162,-0.0733683753613953,0.0280872077238116,0.0854117115265264,-0.244230062632781,0.24670702687019,0.053779359377219,0.0707127414806823,-0.112256678127557,0.0224531290403763,-0.0666535672478241,0.0875490721261221,-0.250760000193023,0.163057128414932,-0.128322159879107,-0.0411041231390976,-0.018223722766732,-0.111454569470766,-0.129693088354358,0.0320028216676054,-0.102303550226528,0.0466743294606347,-0.026047363848788,0.175900895966016,-0.152969329681078,0.0507744195978018,-0.057080394140979,-0.060659992269698,-0.156427846523997,-0.0448414313413649,-0.269415316669929,0.0572440192259738,0.317824211816909,0.112375232329991,-0.0970643241498532,-0.268930807379734,0.0815960325621455,-0.0399364251500636,0.0362908555321906,0.0524392555096947,-0.23427924291136,-0.0157380136256378,-0.00879053508676368,-0.150397162918267,-0.202012307091581,0.0961430010875328,0.197715342081539,0.217678625080021,-0.054366436453657,-0.112925983499372,-0.00598948133003768,0.00388195096814665,-0.014329118182213,-0.304862118585311,-0.0228741293940562,0.117846450553084,-0.0549332140426011,0.130753768464395,0.0455061445260253,-0.0234891238936703,-0.212023716647395,-0.00348123853266095,-0.21132367861711,-0.00668274876117096,-0.292435963599561,-0.248542745486699,-0.0872458271557184,-0.0077448346733956,-0.106271442935938,0.0412620309266472,-0.0378643848430548,0.0435059289085772,-0.0746714659035577,0.0306724456398406,-0.17915240663046,-0.143064685556595,-0.158005761816558,0.0444091728932029,-0.000352144830523242,-0.00630540532099462,0.111569129742046,-0.0697684658088861,-0.0943676079702731,0.0699595193978701,-0.0394003910528408,-0.0434233586110384,0.00872330832109359,-0.0894159587092555,-0.0417595208187623,0.0256591172583092,0.00309991540559973,-0.0310218034761553,-0.113517271646293,-0.082402535473061,-0.27933498042664,-0.0493260676141319,-0.106909200610288,-0.0652678422402496,-0.122396606469718,-0.0434744062932323,-0.175509013769351,0.0762300439530725,0.00710502505196396,-0.424668142507813,-0.609878017597621,-0.360112529945306,-0.143958719317642,0.0367614763756047,-0.21396050372634,-0.078573577997229,0.134454342626374,0.00754160013931939,-0.530231957295726,-0.0624986938546014,-0.0860344040131868,-0.361034638875192,0.00544038283244127,-0.150001925275413,-0.165154097921943,-0.415995551857661,-0.269841910727598,-0.0131723434488603,0.00276674967584807,-3.32605938819997,0.0952702215066027,0.0706770014648153,-0.197986731772716,0.376212603009625,-0.118029714534425,0.176412518978055,-0.235326540115748,0.122430625834331,-0.304663206846196,-0.185949156597444,-0.193295049826445,-0.435442690276603,-0.279134918137196,-0.286325576604491,-0.606310023247163,-0.895012139275581,-0.84273644735224,-0.958799268597985,-0.460211634413448,0.0735360469806453,-0.0284997508053274,0.10759881569009,0.0089616325779666,0.0344963446715327,0.0375691176587201,0.00422367354061821,0.0578015818704103,0.055824786331531,0.0620623105276418,0.000555054672204638,0.145273776896333,-0.00601867015917341,0.0125728400067302,0.0574797679350527,0.043560960882636,0.0710011682183599,-0.0185462102117863,-0.0548952895176038,-0.0221128342503662,-0.0018752571338947,0.0991233900101911,0.054640318048856,-0.018158704105781,0.0293400437378561,0.151013416773172,-0.00226648892267753,0.186716205074339,-0.177534772864725,-0.0216843156463282,0.110047457658745,-0.0484649294866495,0.23230016528873,0.0258521306021498,-0.157069645034252,-0.0845371932501063,-0.026715665815971,-0.133756763093328,-0.115870632029182,-0.0789857012489216,-0.224518221922053,0.140548209498185,0.203143777671102,-0.086775529480756,-0.0472243582596806,-0.0587027996988001,0.135806995886514,-0.218519181711288,0.0904488290853592,-0.064653238764879]]

NNBias = [-0.0728910754370082,-0.0909152260700693,0.344105169652013,0.00753101047565907,0.0764508845081174,0.0775021718009604,0.00369620451652809,-0.0410633303384169,0.00691188464808595,0.278992939014204,-0.0347776142602203,0.081360301021834,0.0695810224342359,0.039989772672935,0.199746290141499,-0.0450135584688674,-0.0229551932929695,0.32465331119479,0.00588477759920555,-0.00998570487633481,0.0425428178268461,0.457707160523873,0.197070921037334,0.0692409280916833,0.0584856835222493,0.0325087365081867,-0.0396003505577713,0.154869642765331,0.133728990961694]

HiddenWeights = [-0.54616320390661,-0.619661947784085,-0.850467157070209,-0.911139124466649,-0.841536141690271,-0.475035482732281,-0.41031020287759,-1.13874135419099,-0.94136176781249,-3.42833835667672,-1.10837543553376,-0.664354377421957,-0.96954587194875,-1.19319524330645,-1.13421929485495,-0.746017378969497,-0.638725031967233,-0.820678478404795,-0.904950608975147,-0.50654493802209,-0.557437616794335,-1.04223651198024,-2.02624904082589,-1.31472937680244,-0.854084156956335,-1.33943963990329,-0.442477673898412,-0.897214500879627,-1.22511487467703]

HiddenBias = [1.43640817097865]

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
  count_input_size = 290
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
        "bert_output_bias", [count_hidden_size], initializer=tf.constant_initializer(NNBias),
        dtype=dtype)
#[29,290]
    count_output_weights = tf.get_variable(
        "count_output_weights", [count_hidden_size, count_input_size],
        initializer=tf.constant_initializer(NNWeights),
        dtype=dtype)		
#[1,29]
    final_output_weights = tf.get_variable(
        "final_output_weights", [num_labels, count_hidden_size],
        initializer=tf.constant_initializer(HiddenWeights),
        dtype=dtype)
#[1]
    final_bias = tf.get_variable(
        "final_output_bias", [num_labels], initializer=tf.constant_initializer(HiddenBias),
        dtype=dtype)		
		
  with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    count_input = tf.reshape(extra_vec, [FLAGS.train_batch_size, count_input_size])
    count_input = tf.saturate_cast(count_input, dtype)	  
#[290]*[290,29]=[29]
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
          tf.TensorShape([290]),
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
