#!/bin/bash

# cd $1

# cp ../kaldi-scripts/path.sh path.sh


##### ASR BUILDING #####
# initialization commands
. ./cmd.sh || { echo -e "\n cmd.sh expected.\n"; exit; }
. ./path.sh || { echo -e "\n path.sh expected. \n"; exit; }

NSTE=$(cut data/test/utt2spk -d' ' -f2 | sort | uniq -c | wc -l)
if [ "$NSTE" -gt 14 ]; then NSTE=14; fi
NSTR=$(cut data/train/utt2spk -d' ' -f2 | sort | uniq -c | wc -l)
if [ "$NSTR" -gt 14 ]; then NSTR=14; fi
echo $NSTE
echo $NSTR


### Triphone + LDA and MLLT + SGMM
# echo
# echo "===== UBM5b2 (UBM for SGMM experiments) TRAINING ====="
# echo
# steps/train_ubm.sh --cmd utils/run.pl 600 data/train data/lang exp/system1/tri3b_ali exp/system1/ubm5b2 || exit 1;
## SGMM
# Training
# echo
# echo "===== SGMM2_5b2 (subspace Gaussian mixture model) TRAINING ====="
# echo
# steps/train_sgmm2.sh --cmd utils/run.pl 11000 25000 data/train data/lang exp/system1/tri3b_ali exp/system1/ubm5b2/final.ubm exp/system1/sgmm2_5b2 || exit 1;
# echo
# echo "===== SGMM2_5b2 (subspace Gaussian mixture model) DECODING ====="
# echo
# Graph compilation
# utils/mkgraph.sh data/lang exp/system1/sgmm2_5b2 exp/system1/sgmm2_5b2/graph
# Decoding
# steps/decode_sgmm2.sh --nj $NSTE --cmd utils/run.pl --transform-dir exp/system1/tri3b/decode_test exp/system1/sgmm2_5b2/graph data/test exp/system1/sgmm2_5b2/decode_test
# SGMM alignments
# echo
# echo "===== SGMM2_5b2 (subspace Gaussian mixture model) ALIGNMENT ====="
# echo
# steps/align_sgmm2.sh --nj $NSTR --cmd utils/run.pl --transform-dir exp/system1/tri3b_ali  --use-graphs true --use-gselect true data/train data/lang exp/system1/sgmm2_5b2 exp/system1/sgmm2_5b2_ali  || exit 1; 

## Denlats
# echo
# echo "===== CREATE DENOMINATOR LATTICES FOR SGMM TRAINING ====="
# echo
# steps/make_denlats_sgmm2.sh --nj $NSTR --cmd utils/run.pl --sub-split $NSTR --transform-dir exp/system1/tri3b_ali data/train data/lang exp/system1/sgmm2_5b2_ali exp/system1/sgmm2_5b2_denlats  || exit 1;

## SGMM+MMI
# Training
echo
echo "===== SGMM2_5b2_MMI (SGMM+MMI) TRAINING ====="
echo
steps/train_mmi_sgmm2.sh --cmd utils/run.pl --transform-dir exp/system1/tri3b_ali --boost 0.1 data/train data/lang exp/system1/sgmm2_5b2_ali exp/system1/sgmm2_5b2_denlats exp/system1/sgmm2_5b2_mmi_b0.1  || exit 1;
# Decoding
echo
echo "===== SGMM2_5b2_MMI (SGMM+MMI) DECODING ====="
echo
for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd utils/run.pl --iter $iter --transform-dir exp/system1/tri3b/decode_test data/lang data/test exp/system1/sgmm2_5b2/decode_test exp/system1/sgmm2_5b2_mmi_b0.1/decode_test_it$iter 
done

# Training
echo
echo "===== SGMM2_5b2_MMI_Z (SGMM+MMI) TRAINING ====="
echo
steps/train_mmi_sgmm2.sh --cmd utils/run.pl --transform-dir exp/system1/tri3b_ali --boost 0.1 data/train data/lang exp/system1/sgmm2_5b2_ali exp/system1/sgmm2_5b2_denlats exp/system1/sgmm2_5b2_mmi_b0.1_z
# Decoding
echo
echo "===== SGMM2_5b2_MMI_Z (SGMM+MMI) DECODING ====="
echo
for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd utils/run.pl --iter $iter --transform-dir exp/system1/tri3b/decode_test data/lang data/test exp/system1/sgmm2_5b2/decode_test exp/system1/sgmm2_5b2_mmi_b0.1_z/decode_test_it$iter
done

# MBR
echo
echo "===== MINIMUM BAYES RISK DECODING ====="
echo
cp -r -T exp/system1/sgmm2_5b2_mmi_b0.1/decode_test_it3{,.mbr}
local/score_mbr.sh data/test data/lang exp/system1/sgmm2_5b2_mmi_b0.1/decode_test_it3.mbr

# SGMM+MMI+fMMI
echo
echo "===== SYSTEM COMBINATION USING MINIMUM BAYES RISK DECODING ====="
echo
local/score_combine.sh data/test data/lang exp/system1/tri3b_fmmi_indirect/decode_test_it3 exp/system1/sgmm2_5b2_mmi_b0.1/decode_test_it3 exp/system1/combine_tri3b_fmmi_indirect_sgmm2_5b2_mmi_b0.1/decode_test_it8_3
echo
echo "===== run.sh script is finished ====="
echo


## DNN
echo
echo "===== DNN DATA PREPARATION ====="
echo
# Config:
gmmdir=exp/system1/tri3b
data_fmllr=data
stage=0 # resume training with --stage=N
train_cmd="run.pl --gpu 1"
cuda_cmd="run.pl --gpu 1"
decode_cmd="run.pl --gpu 1"
# End of config.
. utils/parse_options.sh || exit 1;
#
if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # test
  dir=$data_fmllr/test
  steps/nnet/make_fmllr_feats.sh --nj $NSTE --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_test \
     $dir data/test $gmmdir $dir/log $dir/data || exit 1
  # train
  dir=$data_fmllr/train
  steps/nnet/make_fmllr_feats.sh --nj $NSTR --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir data/train $gmmdir $dir/log $dir/data || exit 1
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

echo
echo "===== DNN DATA TRAINING ====="
echo
# Training
if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  dir=exp/system1/dnn4b_pretrain-dbn
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
     steps/nnet/pretrain_dbn.sh --hid-dim 1024 --rbm-iter $NSTR $data_fmllr/train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/system1/dnn4b_pretrain-dbn_dnn
  ali=${gmmdir}_ali
  feature_transform=exp/system1/dnn4b_pretrain-dbn/final.feature_transform
  dbn=exp/system1/dnn4b_pretrain-dbn/2.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
  $dir/log/train_nnet.log \ 
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_tr90 $data_fmllr/train_cv10 data/lang $ali $ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj $NSTE --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph $data_fmllr/test $dir/decode_test || exit 1;
fi

# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
dir=exp/system1/dnn4b_pretrain-dbn_dnn_smbr
srcdir=exp/system1/dnn4b_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj $NSTR --cmd "$train_cmd" \
    $data_fmllr/train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj $NSTR --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $data_fmllr/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 2 iteratioNSTR of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
    $data_fmllr/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 2 3 4 5 6; do
    steps/nnet/decode.sh --nj $NSTE --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph $data_fmllr/test $dir/decode_test_it${ITER} || exit 1;
  done 
fi

echo Success
# exit 0


# Getting results [see RESULTS file]
for x in exp/system1/*/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done > exp/system1/RESULTS

echo
echo "===== See results in 'exp/system1/RESULTS' ====="
echo
