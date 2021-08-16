#!/usr/bin/env bash
sleep infinity
nvidia-smi
conda env create -f environment.yml
source activate skillearn
cd /dianjiao-pvc/LPT-COVIDx/darts-LPT
python train_search_ts.py --batch_size 6 --layers 6