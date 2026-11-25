#!/bin/bash
#SBATCH --job-name=autocache-search
#SBATCH --partition=acc_ehpc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --time=18:00:00

set -euo pipefail
cd "${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WAN_EVALUATION_BACKEND=slurm
export OPAL_EVALUATION_BACKEND=slurm
export HF_HUB_OFFLINE=1
module_name="${SEARCH_MODULE:-variants.opal_search.search}"
config="${SEARCH_CONFIG:-experiments/qwen21-drawbench30-170/search.json}"
"${REACTIVE_PYTHON:-python}" -m "$module_name" run "$config"
"${REACTIVE_PYTHON:-python}" -m "$module_name" test "$config"
