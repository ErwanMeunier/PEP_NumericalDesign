#!/bin/bash
# Phase 1: budgeted single-horizon designs — one array task per (method, policy, N).
#SBATCH --job-name=pep_phase1
#SBATCH --array=1-__NTASKS1__
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=12:00:00
#SBATCH --output=PEPDesign/cluster/logs/%x_%a.out

module load Julia 2>/dev/null || module load releases/2023b Julia

export JULIA_DEPOT_PATH=${GLOBALSCRATCH:-$HOME}/julia_depot
export MOSEKLM_LICENSE_FILE=${MOSEKLM_LICENSE_FILE:-$HOME/mosek/mosek.lic}
export PEP_TAG=${PEP_TAG:-default}

julia --project=. --threads=$SLURM_CPUS_PER_TASK \
    PEPDesign/cluster/phase1_designs.jl $SLURM_ARRAY_TASK_ID
