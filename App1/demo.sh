python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=./tmp/tf_examples.tfrecord \
  --vocab_file=./vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
