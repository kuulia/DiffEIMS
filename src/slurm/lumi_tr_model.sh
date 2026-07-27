#!/bin/bash
#SBATCH --job-name=tr_diffeims_e2e
#SBATCH --output=outfiles/e2e_ft_%j.out
#SBATCH --time=24:00:00
#SBATCH --account=project_462001155
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
REAL=$(realpath /scratch/project_462001155/lindl)
VENV=$REAL/pyg_venv

cd $REAL/DiffEIMS || exit 1

start_time=$(date +%s)

echo "Running spec2mol_main.py"
srun singularity exec --bind $REAL:$REAL:rw $SIF bash -c "
source $VENV/bin/activate
cd $REAL/DiffEIMS
python src/spec2mol_main.py
"

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Total runtime: $runtime seconds"


