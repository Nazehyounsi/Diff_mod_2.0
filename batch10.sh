#!/bin/bash


#SBATCH --job-name=Config10

#SBATCH --nodes=1

#SBATCH --gpus-per-node=2

#SBATCH --time=1600

#SBATCH --output=%x-%j.out

#SBATCH --error=%x-%j.err

python Training.py --config config10.json --train --gpu --cycle
