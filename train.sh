#!/usr/bin/env bash
nvidia-smi
conda env create -f environment.yml
source activate skillearn
cd /dianjiao-pvc/LPT-COVIDx/darts-LPT
python train_search_ts.py --batch_size 12 --layers 8 --init_channels 8
