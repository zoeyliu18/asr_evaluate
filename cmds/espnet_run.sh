
#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train_mtlalpha1.0.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode_ctcweight1.0.yaml

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=10000    # effective only for word LMs
lmtag=              # tag for managing LMs
lm_resume=          # specify a snapshot file to resume LM training

# decoding parameter
recog_model=model.loss.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# data
datadir=./downloads
an4_root=${datadir}/an4
data_url=http://www.speech.cs.cmu.edu/databases/an4/

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train1"
train_dev="train1"
lm_test="dev1"
recog_set="dev1"

#if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#    echo "stage -1: Data Download"
#    mkdir -p ${datadir}
#    local/download_and_untar.sh ${datadir} ${data_url}
#fi

#if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
#    echo "stage 0: Data preparation"
#    mkdir -p data/{train,test} exp

#    if [ ! -f ${an4_root}/README ]; then
#        echo Cannot find an4 root! Exiting...
#        exit 1
#    fi

#    python3 local/data_prep.py ${an4_root} sph2pipe

#    for x in test train; do
#        for f in text wav.scp utt2spk; do
#            sort data/${x}/${f} -o data/${x}/${f}
#        done
#        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
#    done
#fi

#bash 04_data_prep.sh

feat_tr_dir=${dumpdir}/top_tier/random/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/top_tier/random/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev1 train1; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 1 --write_utt2num_frames true \
            data/top_tier/random/${x} exp/top_tier/random/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/top_tier/random/${x}
    done

#    # make a dev set
#    utils/subset_data_dir.sh --first data/top_tier/random/train1 100 data/top_tier/random/${train_dev}
#    n=$(($(wc -l < data/top_tier/random/train1/text) - 100))
#    utils/subset_data_dir.sh --last data/top_tier/random/train1 ${n} data/top_tier/random/${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/top_tier/random/${train_set}/feats.scp data/top_tier/random/${train_set}/cmvn.ark

    # dump features
    dump.sh --cmd "$train_cmd" --nj 1 --do_delta ${do_delta} \
        data/top_tier/random/${train_set}/feats.scp data/top_tier/random/${train_set}/cmvn.ark exp/top_tier/random/dump_feats/train1 ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 1 --do_delta ${do_delta} \
        data/top_tier/random/${train_dev}/feats.scp data/top_tier/random/${train_set}/cmvn.ark exp/top_tier/random/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/top_tier/random/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 1 --do_delta ${do_delta} \
            data/top_tier/random/${rtask}/feats.scp data/top_tier/random/${train_set}/cmvn.ark exp/top_tier/random/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/top_tier/random/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/top_tier/random/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/top_tier/random/${train_set}/text | cut -f 2- -d" " > temp
    cat hupa_texts.txt temp > for_dictionary.txt
#    text2token.py -s 1 -n 1 data/top_tier/random/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    text2token.py -s 1 -n 1 for_dictionary.txt | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/top_tier/random/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/top_tier/random/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/top_tier/random/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            data/top_tier/random/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi


# It takes about one day. If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/top_tier/random/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"

    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/top_tier/random/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data/top_tier/random/${train_set}/text > ${lmdatadir}/train.txt
        cat hupa_texts.txt >> ${lmdatadir}/train.txt
        cut -f 2- -d" " data/top_tier/random/${train_dev}/text > ${lmdatadir}/valid.txt
        cut -f 2- -d" " data/top_tier/random/${lm_test}/text > ${lmdatadir}/test.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/top_tier/random/lm_train
        lmdict=${dict}
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 data/top_tier/random/${train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train.txt
        cat hupa_texts.txt >> ${lmdatadir}/train.txt
        text2token.py -s 1 -n 1 data/top_tier/random/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 data/top_tier/random/${lm_test}/text \
                | cut -f 2- -d" " > ${lmdatadir}/test.txt
    fi

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --dict ${lmdict}
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/top_tier/random/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=8

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=${dumpdir}/top_tier/random/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
    #    ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            python3 asr_recog_update.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --batchsize 1 \
            ${recog_opts}

    #    score_sclite.sh ${expdir}/${decode_dir} ${dict}
        bash score.sh ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi
