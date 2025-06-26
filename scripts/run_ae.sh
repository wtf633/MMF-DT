#!/usr/bin/env bash
export tag='ae_lr5e-4_dr0.25_ep200'
CUDA_VISIBLE_DEVICES=1 \
python main_ae.py \
  --name $tag \
  --lr 5e-4 \
  --drop_rate 0.25 \
  --epochs 200 \
  --train_bs 12 \
  --val_bs 6 \
  --test_bs 6 \
  --train_img_dir /data/train \
  --val_img_dir   /data/val \
  --test_img_dir  /data/test \
  --csv_root /data/csv \
  >> logs/ae/$tag 2>&1 &

tail -f logs/ae/$tag
