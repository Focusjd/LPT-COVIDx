#!/usr/bin/env bash
nvidia-smi
conda env create -f environment.yml
source activate skillearn
pip install Pillow
mkdir readytorun
python darts-LPT/train_search_ts.py --batch_size 8 --gpu 0,1,2,3,4 is_parallel 1