#!/usr/bin/env bash
sleep infinity
screen -S exp
bash
nvidia-smi
conda env create -f environment.yml
source activate skillearn
cd /dianjiao-pvc/LPT-COVIDx/darts-LPT
python train_search_ts.py python train_search_ts.py --layers 6 --batch_size 8 --init_channels 6
