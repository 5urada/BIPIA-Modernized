#!/bin/bash
#SBATCH --job-name=bipia_generate
#SBATCH --output=bipia_generate_%j.out
#SBATCH --error=bipia_generate_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu

# ============================================================
# BIPIA Generation: Adaptive prompt generation with simulation loop
# ============================================================

cd /scratch/user/u.ss197555/CSCE704/BIPIA-Modernized

module load GCCcore/13.3.0 Python/3.12.3
module load WebProxy
export http_proxy=http://10.71.8.1:8080
export https_proxy=http://10.71.8.1:8080

source .venv/bin/activate

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start : $(date)"
echo "============================================================"

python experiment/generate_adaptive.py
if [ $? -ne 0 ]; then
    echo "ERROR: generate_adaptive.py failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
