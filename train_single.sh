#!/bin/bash
#SBATCH --mem=8g
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=1-20
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT


module load torch
module load tensorflow
module load nvidia

dir=/cs/labs/daphna/idan.azuri/unsupervised-reptile

cd $dir
source /cs/labs/daphna/idan.azuri/venv3.6/bin/activate


python3 run_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_m153
