#!/bin/bash
# Phase 2: HRDP + anytime — one array task per (method, policy). Needs phase 0.
#SBATCH --job-name=pep_phase2
#SBATCH --array=1-__NTASKS2__
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --output=PEPDesign/cluster/logs/%x_%a.out

module load Julia 2>/dev/null || module load releases/2023b Julia

export JULIA_DEPOT_PATH=${GLOBALSCRATCH:-$HOME}/julia_depot
export MOSEKLM_LICENSE_FILE=${MOSEKLM_LICENSE_FILE:-$HOME/mosek/mosek.lic}
export PEP_TAG=${PEP_TAG:-default}

julia --project=. --threads=$SLURM_CPUS_PER_TASK \
    PEPDesign/cluster/phase2_hrdp.jl $SLURM_ARRAY_TASK_ID
