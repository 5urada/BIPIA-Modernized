#!/bin/bash
#SBATCH --job-name=bipia_experiment
#SBATCH --output=bipia_experiment_%j.out
#SBATCH --error=bipia_experiment_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu

# ============================================================
# BIPIA Experiment: Inference + Scoring
# All OpenAI API calls, no GPU needed
# ============================================================

cd /scratch/user/u.ss197555/CSCE704/BIPIA-Modernized

module load GCCcore/13.3.0
module load Python/3.12.3
module load WebProxy
export http_proxy=http://10.71.8.1:8080
export https_proxy=http://10.71.8.1:8080

source .venv/bin/activate

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start : $(date)"
echo "============================================================"

# Step 1: Run inference (2400 API calls, resumes if partially done)
echo ""
echo "=== Step 1: Running inference ==="
python experiment/run_experiment.py
if [ $? -ne 0 ]; then
    echo "ERROR: run_experiment.py failed"
    exit 1
fi

# Step 2: Score results
echo ""
echo "=== Step 2: Scoring results ==="
python experiment/score_results.py
if [ $? -ne 0 ]; then
    echo "ERROR: score_results.py failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
