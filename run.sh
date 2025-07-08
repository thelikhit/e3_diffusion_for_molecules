#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --output=output.txt
#SBATCH 

source .e3_diffusion_for_molecules/bin/activate

python finetuned_learned_noise_schedule_qm9.py