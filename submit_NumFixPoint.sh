#!/bin/bash
# submit_NumFixPoint.sh
# ---------------------------------------------------------------------------
# Slurm array job: each task runs _2_2_NumFixPoint_fx_k_theta.py for one
# (k, theta) pair.  Edit the K_VALUES and THETA_VALUES arrays below to define
# the parameter grid.
#
# Submit with:
#   sbatch submit_NumFixPoint.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=NumFixPoint
#SBATCH --output=logs/NumFixPoint_%A_%a.out   # %A = job ID, %a = array index
#SBATCH --error=logs/NumFixPoint_%A_%a.err
#SBATCH --array=0-19                           # adjust to (len(K_VALUES)-1)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                      # numba parallel threads
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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

# Activate your conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lv

echo "Python being used: $(which python)"

# Quick environment check
python - <<EOF
import numpy, numba
print("Python environment OK")
EOF
if [ $? -ne 0 ]; then
    echo "Python environment broken. Exiting."
    exit 1
fi

# ---------------------------------------------------------------------------
# Parameter grid
# Modify K_VALUES and THETA_VALUES to define the pairs you want.
# Each array task picks one (k, theta) pair by its index.
# ---------------------------------------------------------------------------

# Example: 20 k values at a single fixed theta
K_VALUES=(0.1 0.42 0.74 1.05 1.37 1.68 2.0 2.32 2.63 2.95 \
          3.26 3.58 3.89 4.21 4.53 4.84 5.16 5.47 5.79 6.0)

# To run the same k sweep for multiple theta values, expand THETA_VALUES
# and increase --array accordingly (array index = k_idx * n_theta + theta_idx).
THETA_VALUES=(0.16)           # add more values here if needed

# Fixed simulation parameters
S=50                          # total number of species
IT=100000                     # total number of integration steps (IT)
N_SIMS=10000                  # number of trajectories per run

# ---------------------------------------------------------------------------
# Resolve (k, theta) for this array task
# ---------------------------------------------------------------------------
N_K=${#K_VALUES[@]}
N_THETA=${#THETA_VALUES[@]}

K_IDX=$(( SLURM_ARRAY_TASK_ID % N_K ))
THETA_IDX=$(( SLURM_ARRAY_TASK_ID / N_K ))

K=${K_VALUES[$K_IDX]}
THETA=${THETA_VALUES[$THETA_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K}, theta=${THETA}, S=${S}, IT=${IT}"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p logs
python _2_2_NumFixPoint_fx_k_theta.py "${K}" "${THETA}" "${S}" "${IT}" "${N_SIMS}"
