#!/usr/bin/env bash
sleep infinity
screen 
bash
nvidia-smi
conda env create -f environment.yml
source activate skillearn
cd /dianjiao-pvc/LPT-COVIDx/darts-LPT
python train_search_ts.py python train_search_ts.py --layers 8 --batch_size 12 --init_channels 6
