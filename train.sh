#!/usr/bin/env bash
screen -S exp
nvidia-smi
conda env create -f environment.yml
source activate skillearn
pip install Pillow
mkdir readytorun
python darts-LPT/train_search_ts.py --batch_size 6 --gpu 0,1,2,3 --is_parallel 1
