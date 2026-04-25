#!/bin/bash
#SBATCH --job-name=clique_coarsening
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output/coarse_%j.out
#SBATCH --error=output/coarse_%j.err

module purge
module load cuda-12.1
source /data/himanshu/env_him/bin/activate

mkdir -p output

EPOCHS=50
DATASET="reddit"

echo "--- STARTING CLIQUE-BASED COARSENING COMPARISON ---"

# 1. Baseline: ORIGINAL Graph (Training on unreduced graph)
echo "Step 1: Training Baseline on Original Graph..."
python node_classification.py --dataset $DATASET --epoch $EPOCHS --mode puregpu --baseline > ./output/baseline_${DATASET}_${SLURM_JOB_ID}.txt

# 2. Loop through Retention Ratios
# R=0.3 means keep 30% of nodes (high compression), R=0.7 means keep 70%
RATIOS=(0.3 0.5 0.7 0.85)

for R in "${RATIOS[@]}"
do
    echo "----------------------------------------------------------"
    echo "PROCESSING RETAIN_FRACTION=$R"
    echo "----------------------------------------------------------"
    
    # Generate coarsened graph
    echo "--> Coarsening graph (Clique-Based) with ratio $R..."
    python him_file.py --dataset $DATASET --method coarsening --retain_fraction $R --boost_h 0
    
    # Check if the output DGL file exists before training
    COARSE_GRAPH="${DATASET}.dgl"
    
    if [ -f "$COARSE_GRAPH" ]; then
        echo "--> Training SAGE on coarsened graph..."
        python node_classification.py --dataset $DATASET --epoch $EPOCHS --mode puregpu > ./output/coarse_${DATASET}_R${R}_${SLURM_JOB_ID}.txt
        echo "--> Done. Results in ./output/coarse_${DATASET}_R${R}_${SLURM_JOB_ID}.txt"
    else
        echo "--> Error: Could not generate $COARSE_GRAPH for ratio $R. Skipping."
    fi
done

echo "--- ALL COARSENING EXPERIMENTS COMPLETE ---"
