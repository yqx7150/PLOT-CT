#!/usr/bin/env bash


#CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=4338 DiffIR/train.py -opt options/train_DiffIRS1.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=4321 \
    DiffIR/train.py \
    -opt options/train_DiffIRS1.yml \
    --launcher pytorch