import os
import subprocess

LR = 0.0005
NUM_HIDDEN_LAYERS = 64
NUM_LAYERS = 1
number = 0

for lr in [LR + (i/5000) for i in range(0, 20)]:
    for hs in [NUM_HIDDEN_LAYERS + j for j in range(0, 1000, 100)]:
        for nl in [NUM_LAYERS + k for k in range(0, 7)]:
            shell_path = f'/ubc/cs/research/plai-scratch/virtuecc/GitHub/launch/{number}.sh'
            file = open(shell_path, 'w+')
            file.write('#!/bin/bash\n' +
                       f'#SBATCH --job-name=job_{number}\n' +
                       '#SBATCH --nodes=1\n' +
                       '#SBATCH --cpus-per-task=4\n' +
                       '#SBATCH --time=10-20:00:00\n' +
                       '#SBATCH --partition=plai\n' +
                       '#SBATCH --gres=gpu:1\n' +
                       '#SBATCH --mem=8G\n' +
                       '#SBATCH --mail-user=yungdexiong@gmail.com\n' +
                       '#SBATCH --mail-type=begin\n' +
                       '#SBATCH --mail-type=end\n' +
                       f'#SBATCH --error=logs/job_{number}.err\n' +
                       f'#SBATCH --output=logs/job_{number}.out\n' +
                       'source /ubc/cs/research/plai-scratch/virtuecc/venv/bin/activate\n' +
                       'cd /ubc/cs/research/plai-scratch/virtuecc/GitHub/NLPNoiseModel\n' +
                       f'python Train.py --lr={lr} --num_layers={nl} --hidden_size={hs}')
            file.close()
            command = 'sbatch ' + shell_path
            subprocess.call()
            os.remove(shell_path)
