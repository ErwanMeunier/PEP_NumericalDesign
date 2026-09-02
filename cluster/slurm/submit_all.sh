#!/bin/bash
# Submit the full 4-phase PEPDesign pipeline on a CECI cluster with dependency
# chaining. Run from the repository root:
#   bash PEPDesign/cluster/slurm/submit_all.sh [tag]
#
# For the IGDM ε-grid, resubmit with different PEP_TAG / PEP_IGDM_EPS:
#   PEP_IGDM_EPS=0.1 bash PEPDesign/cluster/slurm/submit_all.sh eps01
set -euo pipefail

export PEP_TAG=${1:-${PEP_TAG:-default}}
mkdir -p PEPDesign/cluster/logs

# Instantiate the environment once on the login node (compiles + resolves).
julia --project=. -e 'import Pkg; Pkg.instantiate()'

# Task counts from the registry (stable ordering).
eval "$(julia --project=. PEPDesign/cluster/registry.jl counts | sed 's/^/N_/')"
echo "array sizes: phase0=$N_phase0 phase1=$N_phase1 phase2=$N_phase2 (tag=$PEP_TAG)"

sed "s/__NTASKS0__/$N_phase0/" PEPDesign/cluster/slurm/phase0.slurm > /tmp/pep_p0.slurm
sed "s/__NTASKS1__/$N_phase1/" PEPDesign/cluster/slurm/phase1.slurm > /tmp/pep_p1.slurm
sed "s/__NTASKS2__/$N_phase2/" PEPDesign/cluster/slurm/phase2.slurm > /tmp/pep_p2.slurm

J0=$(sbatch --parsable --export=ALL /tmp/pep_p0.slurm)
J1=$(sbatch --parsable --export=ALL /tmp/pep_p1.slurm)                    # independent of phase 0
J2=$(sbatch --parsable --export=ALL --dependency=afterok:$J0 /tmp/pep_p2.slurm)
J3=$(sbatch --parsable --export=ALL --dependency=afterok:$J1:$J2 PEPDesign/cluster/slurm/phase3.slurm)

echo "submitted: phase0=$J0 phase1=$J1 phase2=$J2 phase3=$J3"
