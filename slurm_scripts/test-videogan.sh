#!/bin/sh
#SBATCH --job-name=test-videogan
#SBATCH --output=test-videogan.out
#SBATCH --error=test-videogan.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michaelperez012@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=2
#SBATCH --distribution=block:block
#SBATCH --mem-per-cpu=5gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=100:00:00

echo "Date	= $(date)"
echo "host	= $(hostname -s)"
echo "Directory = $(pwd)"

module purge
ml conda
ml ffmpeg
export PATH=/blue/ctolerfranklin/michaelperez012/videogan-venv/bin:$PATH
conda list

T1=$(date +%s)
python3 test_golf-ucf101.py
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
