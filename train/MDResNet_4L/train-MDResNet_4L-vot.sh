#!/bin/bash
#SBATCH --account=def-choang
#SBATCH --gres=gpu:lgpu:4
#SBATCH --mem=16G
#SBATCH --time=0-10:00
#SBATCH --mail-user=<computecanada@hi.mranas.net>
#SBATCH --mail-type=ALL

module load python/3.7 cuda
source $TRK_WD/ENV/bin/activate
cd $TRK_WD/TrackerBenchmark/trackers/MDResNet_4L
python train.py --dataset vot

