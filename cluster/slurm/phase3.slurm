#!/bin/bash
# Phase 3: aggregation — single job. Needs phases 0–2.
#SBATCH --job-name=pep_phase3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=01:00:00
#SBATCH --output=PEPDesign/cluster/logs/%x.out

module load Julia 2>/dev/null || module load releases/2023b Julia

export JULIA_DEPOT_PATH=${GLOBALSCRATCH:-$HOME}/julia_depot
export PEP_TAG=${PEP_TAG:-default}

julia --project=. PEPDesign/cluster/phase3_aggregate.jl
