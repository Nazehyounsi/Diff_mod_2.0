#!/bin/bash


#SBATCH --job-name=Eval13

#SBATCH --nodes=1

#SBATCH --gpus-per-node=2

#SBATCH --time=2600

#SBATCH --output=%x-%j.out

#SBATCH --error=%x-%j.err

python Training.py --config config13.json --evaluate --gpu --evaluation_param 3
