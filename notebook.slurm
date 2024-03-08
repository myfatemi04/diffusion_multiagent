#!/bin/bash

# // SBATCH --nodelist gv01
#SBATCH --time=6:00:00
#SBATCH --job-name=Notebook
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=collabrobogroup
#SBATCH --partition=gpu
#SBATCH -e slurm-%j-Notebook_GPU.err
#SBATCH -o slurm-%j-Notebook_GPU.out

source ~/.bashrc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

mamba activate py310

export JUPYTER_PORT=2596

# echo Command to create an "SSH tunnel", so you can open localhost:2596 locally:
echo ssh -J $USER@rivanna.hpc.virginia.edu -L $JUPYTER_PORT:localhost:$JUPYTER_PORT $USER@$(hostname)

jupyter notebook --no-browser --ip 127.0.0.1 --port $JUPYTER_PORT --NotebookApp.max_buffer_size=1000000000 --ServerApp.allow_remote_access=true --NotebookApp.token='' --NotebookApp.password=''
