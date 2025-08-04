#!/bin/bash

# DiffInject Complete Pipeline Example
# This script demonstrates the complete DiffInject pipeline

set -e  # Exit on any error

echo "=========================================="
echo "DiffInject Complete Pipeline Example"
echo "=========================================="

# Configuration
DATASET="bffhq"           # Dataset name: bffhq, celeba, cmnist, cifar10c, bar
PERCENTAGE="0.5pct"       # Dataset percentage
GPU_ID=0                  # GPU ID to use
H_GAMMA_VALUES="0.1 0.3 0.5 0.7 0.9"  # Content injection strengths to test

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Percentage: $PERCENTAGE"
echo "  GPU: $GPU_ID"
echo "  H-gamma values: $H_GAMMA_VALUES"
echo ""

# Step 1: Train Biased Classifier
echo "Step 1: Training biased classifier..."
cd train_classifier
python train_classifier.py --dataset $DATASET --pct $PERCENTAGE
echo "✓ Biased classifier training completed"
echo ""

# Step 2: Extract Bias-Conflict Samples
echo "Step 2: Extracting bias-conflict samples..."
python top_k_loss.py --dataset $DATASET --pct $PERCENTAGE
echo "✓ Bias-conflict sample extraction completed"
echo ""

# Step 3: Generate Synthetic Samples
echo "Step 3: Generating synthetic samples..."
cd ..

# Create a custom script for the current dataset
cat > temp_generation_script.sh << EOF
#!/bin/bash
sh_file_name="temp_generation_script.sh"
gpu=$GPU_ID

config="${DATASET}.yml"
content_dir="./test_images/${DATASET}/contents"
style_dir="./test_images/${DATASET}/styles"
dt_lambda=0.9985
t_boost=200
n_gen_step=1000
n_inv_step=1000
omega=0.0

for h_gamma in $H_GAMMA_VALUES; do
    save_dir="./bin/${DATASET}_pipeline/h_gamma_\${h_gamma}"
    
    echo "Running with h_gamma = \${h_gamma}, saving to \${save_dir}"
    
    CUDA_VISIBLE_DEVICES=\$gpu python main.py --diff_style \\
                            --content_dir \$content_dir \\
                            --style_dir \$style_dir \\
                            --save_dir \$save_dir \\
                            --config \$config \\
                            --n_gen_step \$n_gen_step \\
                            --n_inv_step \$n_inv_step \\
                            --n_test_step 1000 \\
                            --dt_lambda \$dt_lambda \\
                            --hs_coeff \$h_gamma \\
                            --t_noise \$t_boost \\
                            --sh_file_name \$sh_file_name \\
                            --user_defined_t_edit 500 \\
                            --omega \$omega
    
    echo "Completed run with h_gamma = \${h_gamma}"
done
EOF

chmod +x temp_generation_script.sh
bash temp_generation_script.sh
rm temp_generation_script.sh
echo "✓ Synthetic sample generation completed"
echo ""

# Step 4: Train Debiased Classifier (for best h_gamma value)
echo "Step 4: Training debiased classifier..."
cd train_classifier

# Use the middle h_gamma value as default
BEST_H_GAMMA="0.5"
SYNTHETIC_ROOT="../bin/${DATASET}_pipeline/h_gamma_${BEST_H_GAMMA}"

echo "Using synthetic samples from: $SYNTHETIC_ROOT"
python train_classifier.py --dataset $DATASET --pct $PERCENTAGE \
    --synthetic_root $SYNTHETIC_ROOT \
    --bias_conflict_ratio 0.1

echo "✓ Debiased classifier training completed"
echo ""

# Step 5: Evaluate Results
echo "Step 5: Evaluating results..."
echo "Results are available in:"
echo "  - Biased classifier: ./train_classifier/results/${DATASET}_${PERCENTAGE}/"
echo "  - Synthetic samples: ./bin/${DATASET}_pipeline/"
echo "  - Debiased classifier: ./train_classifier/results/${DATASET}_${PERCENTAGE}_debiased/"
echo ""

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "To compare results across different h_gamma values, you can run:"
echo "for h_gamma in $H_GAMMA_VALUES; do"
echo "  python train_classifier.py --dataset $DATASET --pct $PERCENTAGE \\"
echo "    --synthetic_root ../bin/${DATASET}_pipeline/h_gamma_\${h_gamma} \\"
echo "    --bias_conflict_ratio 0.1"
echo "done"
echo ""
echo "For more detailed analysis, check the results directories and logs." 