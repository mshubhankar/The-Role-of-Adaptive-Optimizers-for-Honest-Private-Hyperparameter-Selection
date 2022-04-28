#!/bin/bash
#SBATCH --array=0-719
#SBATCH --gres=gpu:t4:1         # Request GPU "generic resources"
#SBATCH --cpus-per-task=6       # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=8000M             # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-02:00:00       # DD-HH:MM:SS

# If not using sbatch, use the salloc command followed by the rest:
#salloc --account=rrg-xihe --gres=gpu:1 --cpus-per-task=4 --mem=8000M --time=100:00

module load python/3.6 cuda cudnn

SOURCEDIR=/home/ssasy/projects/rrg-xihe/ssasy/DPAdam-WOSM/dp_optimizers

# Prepare virtualenv
rsync -av $SOURCEDIR/ $SLURM_TMPDIR/ 
virtualenv --no-download $SLURM_TMPDIR/env

source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r $SOURCEDIR/requirements.txt

# Prepare data
mkdir $SLURM_TMPDIR/data
tar xf /home/ssasy/projects/rrg-xihe/ssasy/data/data.tar -C $SLURM_TMPDIR/

# Start training
#python $SLURM_TMPDIR/main.py $SLURM_TMPDIR/data
python $SLURM_TMPDIR/main.py $SLURM_TMPDIR/data $SLURM_ARRAY_TASK_ID
