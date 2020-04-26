#!/bin/bash
#SBATCH --account=def-choang
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-5:00
#SBATCH --mail-user=<computecanada@hi.mranas.net>
#SBATCH --mail-type=ALL

module load python/3.7 cuda
source $TRK_WD/ENV/bin/activate
cd $TRK_WD/TrackerBenchmark
python run_trackers.py -t RoialMDNet -s tb100 -e OPE -n tb100

