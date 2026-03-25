#!/bin/bash
# submit_NumFixPoint_CompleteLinkage.sh
# ---------------------------------------------------------------------------
# Slurm array job: each task runs _2_2_NumFixPoint_fx_k_theta_CompleteLinkage.py
# for one (k, theta) pair.
#
# NOTE: Complete Linkage is O(n_sims^2) — keep N_SIMS <= 5000 to avoid OOM.
#
# Submit with:
#   sbatch submit_NumFixPoint_CompleteLinkage.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=NumFP_CL
#SBATCH --output=logs/NumFP_CL_%A_%a.out
#SBATCH --error=logs/NumFP_CL_%A_%a.err
#SBATCH --array=0-19                           # adjust to (len(K_VALUES)-1)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --mail-user=k23017508@kcl.ac.uk
#SBATCH --mail-type=END,FAIL

echo "===== Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date) ====="

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
cd /scratch/users/k23017508/files || { echo "Directory not found!"; exit 1; }
echo "Working directory: $(pwd)"

module purge
module load anaconda3 2>/dev/null || module load python 2>/dev/null

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lv

echo "Python being used: $(which python)"

python - <<EOF
import numpy, numba, scipy
print("Python environment OK")
EOF
if [ $? -ne 0 ]; then
    echo "Python environment broken. Exiting."
    exit 1
fi

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
K_VALUES=(0.01 0.5 1.0 1.5 1.6 1.7 1.8 1.9 2.0 2.5 \
          #3.26 3.58 3.89 4.21 4.53 4.84 5.16 5.47 5.79 6.0
          )

THETA_VALUES=(0.16)

# Fixed simulation parameters
S=2000 #same to parameters from Niek
IT=100_000
N_SIMS=50           # Keep low for Complete Linkage — O(n_sims^2) memory

# ---------------------------------------------------------------------------
# Resolve (k, theta) for this array task
# ---------------------------------------------------------------------------
N_K=${#K_VALUES[@]}
N_THETA=${#THETA_VALUES[@]}

K_IDX=$(( SLURM_ARRAY_TASK_ID % N_K ))
THETA_IDX=$(( SLURM_ARRAY_TASK_ID / N_K ))

K=${K_VALUES[$K_IDX]}
THETA=${THETA_VALUES[$THETA_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K}, theta=${THETA}, S=${S}, IT=${IT}, n_sims=${N_SIMS}"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p logs
python _2_2_NumFixPoint_fx_k_theta_CompleteLinkage.py "${K}" "${THETA}" "${S}" "${IT}" "${N_SIMS}"

echo "===== Job finished at $(date) ====="
