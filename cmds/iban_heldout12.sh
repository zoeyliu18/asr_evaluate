bash path.sh

mkdir data/iban/heldout_speaker/trainibf_013/lang/

cp -R data/iban/lang/* data/iban/heldout_speaker/trainibf_013/lang/

lm_order=3

echo $lm_order

echo
echo "===== LANGUAGE MODEL CREATION ====="
echo "===== MAKING lm.arpa ====="
echo
loc=`which ngram-count`;
if [ -z $loc ]; then
        if uname -a | grep 64 >/dev/null; then
                sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
        else
                        sdir=$KALDI_ROOT/tools/srilm/bin/i686
        fi
        if [ -f $sdir/ngram-count ]; then
                        echo "Using SRILM language modelling tool from $sdir"
                        export PATH=$PATH:$sdir
        else
                        echo "SRILM toolkit is probably not installed.
                                Instructions: tools/install_srilm.sh"
                        exit 1
        fi
fi
mkdir data/iban/heldout_speaker/trainibf_013/local
local=data/iban/heldout_speaker/trainibf_013/local
cat data/iban/heldout_speaker/trainibf_013/corpus data/iban/local/corpus.txt > data/iban/heldout_speaker/trainibf_013/local/corpus.txt
mkdir $local/tmp
ngram-count -order $lm_order -write-vocab $local/tmp/vocab-full.txt -wbdiscount -text $local/corpus.txt -lm $local/tmp/lm.arpa
echo
echo "===== MAKING G.fst ====="
echo
original_lang=data/iban/lang
mkdir data/iban/heldout_speaker/trainibf_013/lang
lang=data/iban/heldout_speaker/trainibf_013/lang
src/lmbin/arpa2fst --disambig-symbol=#0 --read-symbol-table=$original_lang/words.txt $local/tmp/lm.arpa $lang/G.fst

#cat data/iban/heldout_speaker/trainibf_013/corpus.1 data/local/hupa_texts.txt > data/local/corpus.txt

#rm -r -f data/local/tmp

#bash lm.sh $lm_order ### can be whatever numbers

#echo $lm_order


train_cmd="run.pl --gpu 1"

# monophones
echo
#echo "===== MONO TRAINING ====="
echo
# Training
steps/train_mono.sh --nj 1 --cmd "$train_cmd" data/iban/heldout_speaker/trainibf_013 data/iban/heldout_speaker/trainibf_013/lang exp/iban/heldout_speaker/systemibf_013/mono
echo
echo "===== MONO DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh --mono data/iban/heldout_speaker/trainibf_013/lang exp/iban/heldout_speaker/systemibf_013/mono exp/iban/heldout_speaker/systemibf_013/mono/graph
# Decoding
steps/decode.sh --nj 1 --cmd "$train_cmd" exp/iban/heldout_speaker/systemibf_013/mono/graph data/iban/heldout_speaker/devibf_013 exp/iban/heldout_speaker/systemibf_013/mono/decode_dev
echo
echo "===== MONO ALIGNMENT ====="
echo
steps/align_si.sh --boost-silence 1.25 --nj 1 --cmd "$train_cmd" data/iban/heldout_speaker/trainibf_013 data/iban/heldout_speaker/trainibf_013/lang exp/iban/heldout_speaker/systemibf_013/mono exp/iban/heldout_speaker/systemibf_013/mono_ali

## Triphone
echo
echo "===== TRI1 (first triphone pass) TRAINING ====="
echo
# Training
echo -e "triphones step \n"
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 4200 40000 data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/mono_ali exp/iban/heldout_speaker/systemibf_013/tri1
echo
echo "===== TRI1 (first triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri1 exp/iban/heldout_speaker/systemibf_013/tri1/graph
# Decoding
steps/decode.sh --nj 1 --cmd "$train_cmd" exp/iban/heldout_speaker/systemibf_013/tri1/graph data/iban/heldout_speaker/devibf_013 exp/iban/heldout_speaker/systemibf_013/tri1/decode_dev
echo
echo "===== TRI1 (first triphone pass) ALIGNMENT ====="
echo
steps/align_si.sh --nj 1 --cmd "$train_cmd" data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri1 exp/iban/heldout_speaker/systemibf_013/tri1_ali

## Triphone + Delta Delta
echo
echo "===== TRI2a (second triphone pass) TRAINING ====="
echo
# Training
steps/train_deltas.sh --cmd utils/run.pl 4200 40000  data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri1_ali exp/iban/heldout_speaker/systemibf_013/tri2a
echo
echo "===== TRI2a (second triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri2a exp/iban/heldout_speaker/systemibf_013/tri2a/graph
# Decoding
steps/decode.sh --nj 1 --cmd utils/run.pl exp/iban/heldout_speaker/systemibf_013/tri2a/graph data/iban/heldout_speaker/devibf_013 exp/iban/heldout_speaker/systemibf_013/tri2a/decode_dev
echo
echo "===== TRI2a (second triphone pass) ALIGNMENT ====="
echo
steps/align_si.sh --nj 1 --cmd utils/run.pl data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri2a exp/iban/heldout_speaker/systemibf_013/tri2a_ali

## Triphone + Delta Delta + LDA and MLLT
echo
echo "===== TRI2b (third triphone pass) TRAINING ====="
echo
# Training
steps/train_lda_mllt.sh --cmd utils/run.pl --splice-opts "--left-context=3 --right-context=3"  4200 40000 data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri2a_ali exp/iban/heldout_speaker/systemibf_013/tri2b
echo
echo "===== TRI2b (third triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri2b exp/iban/heldout_speaker/systemibf_013/tri2b/graph
# Decoding
steps/decode.sh --nj 1 --cmd utils/run.pl exp/iban/heldout_speaker/systemibf_013/tri2b/graph data/iban/heldout_speaker/devibf_013 exp/iban/heldout_speaker/systemibf_013/tri2b/decode_dev
echo
echo "===== TRI2b (third triphone pass) ALIGNMENT ====="
echo
steps/align_si.sh --nj 1 --cmd utils/run.pl --use-graphs true data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri2b exp/iban/heldout_speaker/systemibf_013/tri2b_ali

## Triphone + Delta Delta + LDA and MLLT + SAT and FMLLR
echo
echo "===== TRI3b (fourth triphone pass) TRAINING ====="
echo
# Training
steps/train_sat.sh --cmd utils/run.pl 4200 40000 data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri2b_ali exp/iban/heldout_speaker/systemibf_013/tri3b
echo
echo "===== TRI3b (fourth triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri3b exp/iban/heldout_speaker/systemibf_013/tri3b/graph
# Decoding
steps/decode_fmllr.sh --nj 1 --cmd utils/run.pl exp/iban/heldout_speaker/systemibf_013/tri3b/graph data/iban/heldout_speaker/devibf_013 exp/iban/heldout_speaker/systemibf_013/tri3b/decode_dev
echo
echo "===== TRI3b (fourth triphone pass) ALIGNMENT ====="
echo
# HMM/GMM aligments
steps/align_fmllr.sh --nj 1 --cmd utils/run.pl data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri3b exp/iban/heldout_speaker/systemibf_013/tri3b_ali

echo
echo "===== CREATE DENOMINATOR LATTICES FOR MMI TRAINING ====="
echo
steps/make_denlats.sh --nj 1 --cmd utils/run.pl --sub-split 14 --transform-dir exp/iban/heldout_speaker/systemibf_013/tri3b_ali data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri3b exp/iban/heldout_speaker/systemibf_013/tri3b_denlats || exit 1;

## Triphone + LDA and MLLT + SAT and FMLLR + fMMI and MMI
# Training
echo
echo "===== TRI3b_MMI (fifth triphone pass) TRAINING ====="
echo
steps/train_mmi.sh --cmd utils/run.pl --boost 0.1 data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri3b_ali exp/iban/heldout_speaker/systemibf_013/tri3b_denlats exp/iban/heldout_speaker/systemibf_013/tri3b_mmi_b0.1  || exit 1;
# Decoding
echo
echo "===== TRI3b_MMI (fifth triphone pass) DECODING ====="
echo
steps/decode.sh --nj 1 --cmd utils/run.pl --transform-dir exp/iban/heldout_speaker/systemibf_013/tri3b/decode_dev exp/iban/heldout_speaker/systemibf_013/tri3b/graph data/iban/heldout_speaker/devibf_013 exp/iban/heldout_speaker/systemibf_013/tri3b_mmi_b0.1/decode_dev



## DNN
echo
echo "===== DNN DATA PREPARATION ====="
echo
# Config:
gmmdir=/data/liuaal/kaldi/exp/iban/heldout_speaker/systemibf_013/tri3b
data_fmllr=/data/liuaal/kaldi/exp/iban/heldout_speaker/systemibf_013/tri3b
stage=0 # resume training with --stage=N
train_cmd="/data/liuaal/kaldi/utils/run.pl --gpu 1"
cuda_cmd="/data/liuaal/kaldi/utils/run.pl --gpu 1"
decode_cmd="/data/liuaal/kaldi/utils/run.pl --gpu 1"
# End of config.
bash /data/liuaal/kaldi/utils/parse_options.sh || exit 1;
#

if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # dev
  dir=$data_fmllr/dev
  bash new_make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_dev \
     $dir data/iban/heldout_speaker/devibf_013 $gmmdir $dir/log $dir/data || exit 1
  # train
  dir=$data_fmllr/train
  bash new_make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir data/iban/heldout_speaker/trainibf_013 $gmmdir $dir/log $dir/data || exit 1
  # split the data : 90% train 10% cross-validation (held-out)
  /data/liuaal/kaldi/utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi
#  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
#  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \

#cp pr exp/iban/heldout_speaker/systemibf_013/tri3b/dev/* exp/iban/heldout_speaker/systemibf_013/tri3b/train_cv10
#cp pr exp/iban/heldout_speaker/systemibf_013/tri3b/train/* exp/iban/heldout_speaker/systemibf_013/tri3b/train_tr10

echo
echo "===== DNN DATA TRAINING ====="
echo

# Training

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  dir=/data/liuaal/kaldi/exp/iban/heldout_speaker/systemibf_013/dnn4b_pretrain-dbn
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
     /data/liuaal/kaldi/steps/nnet/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 14 $data_fmllr/train $dir || exit 1;
fi

chmod -R 777 /data/liuaal/kaldi/exp/iban/heldout_speaker/systemibf_013/dnn4b_pretrain-dbn_dnn/log/train_nnet.log

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=/data/liuaal/kaldi/exp/iban/heldout_speaker/systemibf_013/dnn4b_pretrain-dbn_dnn
  ali=${gmmdir}_ali
  feature_transform=exp/iban/heldout_speaker/systemibf_013/dnn4b_pretrain-dbn/final.feature_transform
  dbn=/data/liuaal/kaldi/exp/iban/heldout_speaker/systemibf_013/dnn4b_pretrain-dbn/2.dbn
  TRAIN_DIR=train
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    /data/liuaal/kaldi/steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/${TRAIN_DIR}_tr90 $data_fmllr/${TRAIN_DIR}_tr90 data/iban/lang $ali $ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  /data/liuaal/kaldi/steps/nnet/decode.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph $data_fmllr/dev $dir/decode_dev || exit 1;
fi

# Sequence training using sMBR criterion, we do Stochastic-GD
# with per-utterance updates. We use usually good acwt 0.1
dir=/data/liuaal/kaldi/exp/iban/heldout_speaker/systemibf_013/dnn4b_pretrain-dbn_dnn_smbr
srcdir=/data/liuaal/kaldi/exp/iban/heldout_speaker/systemibf_013/dnn4b_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  /data/liuaal/kaldi/steps/nnet/align.sh --nj 1 --cmd "$train_cmd" \
    $data_fmllr/train data/iban/lang $srcdir ${srcdir}_ali || exit 1;
  /data/liuaal/kaldi/steps/nnet/make_denlats.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $data_fmllr/train data/iban/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 2 iterations of sMBR
  /data/liuaal/kaldi/steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
    $data_fmllr/train data/iban/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 2 3 4 5 6; do
    /data/liuaal/kaldi/steps/nnet/decode.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph $data_fmllr/dev $dir/decode_dev_it${ITER} || exit 1;
  done
fi

echo
echo "===== See results in 'exp/iban/heldout_speaker/systemibf_013/RESULTS' ====="
echo

for x in exp/iban/heldout_speaker/systemibf_013/*/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done > exp/iban/heldout_speaker/systemibf_013/RESULTS


echo Success
exit 0


## UBM for fMMI experiments
# Training
echo
echo "===== DUBM3b (UBM for fMMI experiments) TRAINING ====="
echo
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 1 --cmd "$train_cmd" 600 data/iban/iban/heldout_speaker/trainibf_013 data/iban/iban/lang exp/iban/heldout_speaker/systemibf_013/tri3b_ali exp/iban/heldout_speaker/systemibf_013/dubm3b

## fMMI+MMI
# Training
echo
echo "===== TRI3b_FMMI (fMMI+MMI) TRAINING ====="
echo
steps/train_mmi_fmmi.sh --cmd utils/run.pl --boost 0.1 data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri3b_ali exp/iban/heldout_speaker/systemibf_013/dubm3b exp/iban/heldout_speaker/systemibf_013/tri3b_denlats exp/iban/heldout_speaker/systemibf_013/tri3b_fmmi_a || exit 1;
# Decoding
echo
echo "===== TRI3b_FMMI (fMMI+MMI) DECODING ====="
echo
for iter in 3 4 5 6 7 8; do
  steps/decode_fmmi.sh --nj 1 --cmd utils/run.pl --iter $iter --transform-dir exp/iban/heldout_speaker/systemibf_013/tri3b/decode_dev exp/iban/heldout_speaker/systemibf_013/tri3b/graph data/iban/heldout_speaker/devibf_013 exp/iban/heldout_speaker/systemibf_013/tri3b_fmmi_a/decode_dev_it$iter
done

## fMMI + mmi with indirect differential
# Training
echo
echo "===== TRI3b_FMMI_INDIRECT (fMMI+MMI with indirect differential) TRAINING ====="
echo
steps/train_mmi_fmmi_indirect.sh --cmd utils/run.pl --boost 0.1 data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri3b_ali exp/iban/heldout_speaker/systemibf_013/dubm3b exp/iban/heldout_speaker/systemibf_013/tri3b_denlats exp/iban/heldout_speaker/systemibf_013/tri3b_fmmi_indirect || exit 1;
# Decoding
echo
echo "===== TRI3b_FMMI_INDIRECT (fMMI+MMI with indirect differential) DECODING ====="
echo
for iter in 3 4 5 6 7 8; do
  steps/decode_fmmi.sh --nj 1 --cmd utils/run.pl --iter $iter --transform-dir  exp/iban/heldout_speaker/systemibf_013/tri3b/decode_dev exp/iban/heldout_speaker/systemibf_013/tri3b/graph data/iban/heldout_speaker/devibf_013 exp/iban/heldout_speaker/systemibf_013/tri3b_fmmi_indirect/decode_dev_it$iter
done

### Triphone + LDA and MLLT + SGMM
echo
echo "===== UBM5b2 (UBM for SGMM experiments) TRAINING ====="
echo
steps/train_ubm.sh --cmd utils/run.pl 600 data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri3b_ali exp/iban/heldout_speaker/systemibf_013/ubm5b2 || exit 1;
## SGMM
# Training
echo
echo "===== SGMM2_5b2 (subspace Gaussian mixture model) TRAINING ====="
echo
steps/train_sgmm2.sh --cmd utils/run.pl 11000 25000 data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri3b_ali exp/iban/heldout_speaker/systemibf_013/ubm5b2/final.ubm exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2 || exit 1;
echo
echo "===== SGMM2_5b2 (subspace Gaussian mixture model) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data/iban/lang exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2 exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2/graph
# Decoding
steps/decode_sgmm2.sh --nj 1 --cmd utils/run.pl --transform-dir exp/iban/heldout_speaker/systemibf_013/tri3b/decode_dev exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2/graph data/iban/heldout_speaker/devibf_013 exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2/decode_dev
# SGMM alignments
echo
echo "===== SGMM2_5b2 (subspace Gaussian mixture model) ALIGNMENT ====="
echo
steps/align_sgmm2.sh --nj 1 --cmd utils/run.pl --transform-dir exp/iban/heldout_speaker/systemibf_013/tri3b_ali  --use-graphs true --use-gselect true data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2 exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_ali  || exit 1;

## Denlats
echo
echo "===== CREATE DENOMINATOR LATTICES FOR SGMM TRAINING ====="
echo
steps/make_denlats_sgmm2.sh --nj 1 --cmd utils/run.pl --sub-split 1 --transform-dir exp/iban/heldout_speaker/systemibf_013/tri3b_ali data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_ali exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_denlats  || exit 1;

## SGMM+MMI
# Training
echo
echo "===== SGMM2_5b2_MMI (SGMM+MMI) TRAINING ====="
echo
steps/train_mmi_sgmm2.sh --cmd utils/run.pl --transform-dir exp/iban/heldout_speaker/systemibf_013/tri3b_ali --boost 0.1 data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_ali exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_denlats exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_mmi_b0.1  || exit 1;
# Decoding
echo
echo "===== SGMM2_5b2_MMI (SGMM+MMI) DECODING ====="
echo
for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd utils/run.pl --iter $iter --transform-dir exp/iban/heldout_speaker/systemibf_013/tri3b/decode_dev data/iban/lang data/iban/heldout_speaker/devibf_013 exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2/decode_dev exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_mmi_b0.1/decode_dev_it$iter
done

# Training
echo
echo "===== SGMM2_5b2_MMI_Z (SGMM+MMI) TRAINING ====="
echo
steps/train_mmi_sgmm2.sh --cmd utils/run.pl --transform-dir exp/iban/heldout_speaker/systemibf_013/tri3b_ali --boost 0.1 data/iban/heldout_speaker/trainibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_ali exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_denlats exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_mmi_b0.1_z
# Decoding
echo
echo "===== SGMM2_5b2_MMI_Z (SGMM+MMI) DECODING ====="
echo
for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd utils/run.pl --iter $iter --transform-dir exp/iban/heldout_speaker/systemibf_013/tri3b/decode_dev data/iban/lang data/iban/heldout_speaker/devibf_013 exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2/decode_dev exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_mmi_b0.1_z/decode_dev_it$iter
done

# MBR
echo
echo "===== MINIMUM BAYES RISK DECODING ====="
echo
cp -r -T exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_mmi_b0.1/decode_dev_it3{,.mbr}
local/score_mbr.sh data/iban/heldout_speaker/devibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_mmi_b0.1/decode_dev_it3.mbr

# SGMM+MMI+fMMI
echo
echo "===== SYSTEM COMBINATION USING MINIMUM BAYES RISK DECODING ====="
echo
local/score_combine.sh data/iban/heldout_speaker/devibf_013 data/iban/lang exp/iban/heldout_speaker/systemibf_013/tri3b_fmmi_indirect/decode_dev_it3 exp/iban/heldout_speaker/systemibf_013/sgmm2_5b2_mmi_b0.1/decode_dev_it3 exp/iban/heldout_speaker/systemibf_013/combine_tri3b_fmmi_indirect_sgmm2_5b2_mmi_b0.1/decode_dev_it8_3

echo
echo "===== run.sh script is finished ====="
echo
