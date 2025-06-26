#!/usr/bin/env bash
export tag='m3t_depth12_lr1e-5_bs4'
CUDA_VISIBLE_DEVICES=2 \
python main_m3t.py \
  --name $tag \
  --lr 1e-5 \
  --depth 12 \
  --drop_rate 0.15 \
  --epochs 250 \
  --rand_p 0.25 \
  --train_bs 4 --val_bs 4 --test_bs 4 \
  --train_img_dir /data/train \
  --val_img_dir   /data/val \
  --test_img_dir  /data/test \
  --csv_root /data/csv \
  >> logs/m3t/$tag 2>&1 &

tail -f logs/m3t/$tag
