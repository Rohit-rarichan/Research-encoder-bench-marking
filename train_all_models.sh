#!/bin/bash

# ============================================================================
# Sequential Model Training Script
# Trains all 4 models back-to-back in tmux
# Each model: 100 epochs, batch_size 6, with checkpoint resume
# ============================================================================

# Configuration
CONDA_ENV="encoders"
DATA_ROOT="/home/tamoghno/datasets/nuimages"
EPOCHS=100
BATCH_SIZE=6
SEED=42

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p /home/tamoghno/rohit-encoders/outputs

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Sequential Model Training Pipeline${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "Conda Environment: ${GREEN}${CONDA_ENV}${NC}"
echo -e "Data Root: ${GREEN}${DATA_ROOT}${NC}"
echo -e "Epochs: ${GREEN}${EPOCHS}${NC}"
echo -e "Batch Size: ${GREEN}${BATCH_SIZE}${NC}"
echo -e "Seed: ${GREEN}${SEED}${NC}"
echo ""

# Function to train a single model
train_model() {
    local model=$1
    local checkpoint_path="./outputs/${model}/checkpoint_best.pth"
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Starting training: ${GREEN}${model}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Build command
    local cmd="cd /home/tamoghno/rohit-encoders && python train_nuimages.py"
    cmd="${cmd} --model ${model}"
    cmd="${cmd} --data_root ${DATA_ROOT}"
    cmd="${cmd} --epochs ${EPOCHS}"
    
    # SegFormer and ConvNeXt-B need smaller batch size (memory intensive)
    if [[ "${model}" == "segformer_upernet" ]] || [[ "${model}" == "convnext_b" ]]; then
        cmd="${cmd} --batch_size 4"
    else
        cmd="${cmd} --batch_size ${BATCH_SIZE}"
    fi
    
    cmd="${cmd} --seed ${SEED}"
    
    # Add resume flag if checkpoint exists
    if [ -f "${checkpoint_path}" ]; then
        echo -e "${GREEN}✓ Found existing checkpoint: ${checkpoint_path}${NC}"
        echo -e "${YELLOW}Resuming from best checkpoint...${NC}"
        cmd="${cmd} --resume ${checkpoint_path}"
        echo ""
    else
        echo -e "${YELLOW}No checkpoint found, starting fresh training${NC}"
        echo ""
    fi
    
    # Run training
    eval ${cmd}
    local exit_code=$?
    
    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}✓ ${model} training completed successfully${NC}"
    else
        echo -e "${RED}✗ ${model} training failed with exit code ${exit_code}${NC}"
        return 1
    fi
    echo ""
    
    return 0
}

# Main execution
cd /home/tamoghno/rohit-encoders

# Initialize conda
echo -e "${YELLOW}Initializing conda environment: ${CONDA_ENV}${NC}"

# Source conda initialization (miniconda)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

# Activate the environment
conda activate ${CONDA_ENV}

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate conda environment${NC}"
    exit 1
fi

# Set PyTorch memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo -e "${GREEN}✓ Conda environment activated${NC}"
echo -e "${GREEN}✓ PyTorch memory fragmentation fix enabled${NC}"
echo ""

# Training sequence
MODELS=("segformer_upernet" "resnet101" "swin_b" "convnext_b")
START_TIME=$(date +%s)

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    model_num=$((i + 1))
    total_models=${#MODELS[@]}
    
    echo -e "${YELLOW}[${model_num}/${total_models}]${NC} Training model: ${GREEN}${model}${NC}"
    
    train_model "${model}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Training failed for ${model}. Aborting pipeline.${NC}"
        exit 1
    fi
    
    # Summary after each model
    if [ $i -lt $((total_models - 1)) ]; then
        next_model="${MODELS[$((i + 1))]}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}Next model: ${GREEN}${next_model}${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
    fi
done

# Final summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ ALL MODELS TRAINED SUCCESSFULLY!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "Total training time: ${GREEN}${HOURS}h ${MINUTES}m${NC}"
echo ""

# Generate summary
echo -e "${YELLOW}Generating training summary...${NC}"
echo ""

for model in "${MODELS[@]}"; do
    results_file="./outputs/${model}/results.json"
    if [ -f "${results_file}" ]; then
        echo -e "${BLUE}${model}:${NC}"
        python3 << EOF
import json
with open('${results_file}') as f:
    results = json.load(f)
    print(f"  Best mIoU: {results.get('best_miou', 'N/A'):.4f}" if isinstance(results.get('best_miou'), (int, float)) else f"  Best mIoU: {results.get('best_miou', 'N/A')}")
    val_miou = results.get('val_results', {}).get('miou', 'N/A')
    test_miou = results.get('test_results', {}).get('miou', 'N/A')
    full_miou = results.get('full_dataset_results', {}).get('miou', 'N/A')
    print(f"  Val mIoU: {val_miou:.4f}" if isinstance(val_miou, (int, float)) else f"  Val mIoU: {val_miou}")
    print(f"  Test mIoU: {test_miou:.4f}" if isinstance(test_miou, (int, float)) else f"  Test mIoU: {test_miou}")
    print(f"  Full Dataset mIoU: {full_miou:.4f}" if isinstance(full_miou, (int, float)) else f"  Full Dataset mIoU: {full_miou}")
EOF
    fi
done

echo ""
echo -e "${GREEN}Training pipeline complete!${NC}"
