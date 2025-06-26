#!/usr/bin/env bash
export tag='effnet_lr1e-4_bs8_ep150'
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12355 main.py \
  --name $tag \
  --lr 1e-4 \
  --epochs 150 \
  --rand_p 0.4 \
  --train_bs 8 --val_bs 8 --test_bs 8 \
  --train_img_dir /data/train \
  --val_img_dir   /data/val \
  --test_img_dir  /data/test \
  --csv_root /data/csv \
  >> logs/effnet/$tag 2>&1 &

tail -f logs/effnet/$tag
