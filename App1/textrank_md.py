

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import math
import time
import random
import spacy
import pytextrank
# import modeling
# import create_pretraining_data as cpd
# import optimization_multi_gpu
# import tokenization
# import tensorflow as tf
# import horovod.tensorflow as hvd
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
parser.add_argument("--do_train",action='store_true', default=False, help="Whether to run training.")

parser.add_argument("--do_eval", action='store_true', default=False, help="Whether to run eval on the dev set.")

parser.add_argument("--do_predict", action='store_true', default=False, help="Whether to run pre dict on the dev set.")
parser.add_argument("--read_local", action='store_true', default=False, help="read local")
parser.add_argument(
    "--output_dir", type=str, default="",help=
    "The output directory where the model checkpoints will be written.")
parser.add_argument(
    "--data_dir", type=str, default="",help=
    "The output directory where the model checkpoints will be written.")

(args, unknown) = parser.parse_known_args()

# tf.app.flags.DEFINE_string('phillyarg','blobpathconv', 'fix phillyarg issue')

# flags = tf.flags

# FLAGS = flags.FLAGS
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
    if args.read_local:
      #dst = "./train.tsv"
      #if not tf.gfile.Exists(dst):
      #  tf.io.gfile.copy(input_file, dst)
      #  print("Download file Done!")
      #__tmp__ = hvd.mpi_ops.allgather(tf.constant(0.0, shape=[4, 1]))
      with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        for index, line in enumerate(reader):
        #   if index % hvd.size() == hvd.rank():
            yield line
    else: 
      with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        for line in reader:
          yield line

class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""
  def get_train_and_dev_line_cnt(self, train_data_dir, dev_data_dir):
    train_line_cnt = 0
    dev_line_cnt = 0
    if args.do_train:
      #train_line_cnt = 10000000 #478585308 #TODO
      with open(os.path.join(train_data_dir, "train.tsv"), "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for _ in reader:    
          train_line_cnt += 1
    if args.do_eval:
      with open(os.path.join(dev_data_dir, "dev.tsv"), "r") as f:
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
        self._read_tsv(data_dir), "dev")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    for (i, line) in enumerate(lines):
      #print("origin line:[%s]" % line)
      guid = "%s-%s" % (line[0], '0')
      text_a = line[1]
      text_b = ''
      label = [float(0)]*40 #Teacher Predict Emb40
      pos = 'ml-1'
      weight = float(1)
      perc = float(0)
      pos_bias = float(7.50)
      yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, pos=pos, perc=perc, weight=weight, pos_bias = pos_bias)


if __name__ == "__main__":
    print("***** create processor *****")
    # processor = MrpcProcessor()
    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)
    output_predict_file = os.path.join(args.output_dir, "predict_results.txt")
    nlp = spacy.load("en_core_web_md")

    # add PyTextRank to the spaCy pipeline
    nlp.add_pipe("textrank")

    with open(output_predict_file, "a") as writer:
      with open(args.data_dir, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for idx, line in enumerate(reader):
        # for idx, example in enumerate(processor.get_test_examples(args.data_dir)):
            # example text
            # text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."

            # load a spaCy model, depending on language, scale, etc.
            doc = nlp(" ".join(line[1].split("[SEP]")))
            result = [phrase.text for phrase in doc._.phrases]
            # # examine the top-ranked phrases in the document
            # for phrase in doc._.phrases:
            #     print(phrase.text) 
            #     print(phrase.rank, phrase.count)
            #     print(phrase.chunks)
            writer.write("%s\t%s\n" % (line[0]+"-0", ' '.join(result[:50])))
            if idx % 50000 == 0:
                print("Predict [%s]->[%s]", line[0], ' '.join(result[:50]))
