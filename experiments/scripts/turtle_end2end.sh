#!/bin/bash
# Usage:
# ./experiments/scripts/turtle_end2end.sh [options args to {train,test}_net.py]
#

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=0
NET=VGG16

ITERS=70000
# The names in factory.py
DATASET_TRAIN=turtle_train
DATASET_TEST=turtle_test

NET_INIT=data/imagenet_models/${NET}.v2.caffemodel

#array=( $@ )
#len=${#array[@]}
#EXTRA_ARGS=${array[@]:2:$len}
#EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/turtle_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/turtle/solver.prototxt \
  --weights ${NET_INIT} \
  --imdb ${DATASET_TRAIN} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/turtle/vgg16_end2end_test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${DATASET_TEST} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
