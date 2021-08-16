#!/usr/bin/env bash
nvidia-smi
conda env create -f environment.yml
source activate skillearn
cd /dianjiao-pvc/LPT-COVIDx/darts-LPT
python train_search_ts.py --batch_size 4 --gpu 0,1,2,3 --is_parallel 1 --layers 6