#!/bin/bash
# Phase 0: W*_N precompute — one array task per (method, N).
# Set the array size from:  julia --project=. PEPDesign/cluster/registry.jl counts
#SBATCH --job-name=pep_phase0
#SBATCH --array=1-__NTASKS0__
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=02:00:00
#SBATCH --output=PEPDesign/cluster/logs/%x_%a.out

# CECI: adapt the module line to the cluster (see www.ceci-hpc.be/clusters.html)
module load Julia 2>/dev/null || module load releases/2023b Julia

export JULIA_DEPOT_PATH=${GLOBALSCRATCH:-$HOME}/julia_depot
export MOSEKLM_LICENSE_FILE=${MOSEKLM_LICENSE_FILE:-$HOME/mosek/mosek.lic}
export PEP_TAG=${PEP_TAG:-default}

julia --project=. --threads=$SLURM_CPUS_PER_TASK \
    PEPDesign/cluster/phase0_wstar.jl $SLURM_ARRAY_TASK_ID
