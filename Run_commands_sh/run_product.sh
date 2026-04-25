#!/bin/bash
#SBATCH --job-name=coarse_products
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output_2/products_%j.out
#SBATCH --error=output_2/products_%j.err

module purge
module load cuda-12.1
source /data/himanshu/env_him/bin/activate

OUT_DIR="output_2"
mkdir -p $OUT_DIR

EPOCHS=50
DATASET="ogbn-products"

echo "--- STARTING CLIQUE-BASED COARSENING: $DATASET ---"

# 1. Baseline: ORIGINAL Graph
echo "Step 1: Training Baseline on Original Graph..."
python node_classification.py --dataset $DATASET --epoch $EPOCHS --mode puregpu --baseline > ./${OUT_DIR}/baseline_${DATASET}_${SLURM_JOB_ID}.txt

# 2. Loop through Retention Ratios
RATIOS=(0.3 0.5 0.7 0.85)

for R in "${RATIOS[@]}"
do
    echo "----------------------------------------------------------"
    echo "PROCESSING RETAIN_FRACTION=$R"
    echo "----------------------------------------------------------"
    
    python him_file.py --dataset $DATASET --method coarsening --retain_fraction $R --boost_h 0
    
    COARSE_GRAPH="products.dgl"
    if [ -f "$COARSE_GRAPH" ]; then
        echo "--> Training SAGE on coarsened graph..."
        python node_classification.py --dataset $DATASET --epoch $EPOCHS --mode puregpu > ./${OUT_DIR}/coarse_${DATASET}_R${R}_${SLURM_JOB_ID}.txt
    else
        echo "--> Error: Could not generate $COARSE_GRAPH. Skipping."
    fi
done

echo "--- $DATASET EXPERIMENTS COMPLETE ---"
