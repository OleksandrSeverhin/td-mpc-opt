#!/bin/bash

# Define your chosen ManiSkill3 tasks
TASKS=(
    "PushCube-v1"
    "PickCube-v1"
    "StackCube-v1"
    "PegInsertionSide-v1"
)

# Define the teacher checkpoint
TEACHER_CKPT="/root/td-mpc-opt/tdmpc2/models/multitask/mt30-317M.pt"

# Loop through each task
for TASK in "${TASKS[@]}"; do
    echo "========================================================"
    echo "Starting distillation for: $TASK"
    echo "========================================================"

    # Run the script. 
    # Notice the dynamic exp_name! This guarantees separate save folders.
    python train_distill.py \
        env_id=$TASK \
        model_size=1 \
        checkpoint=$TEACHER_CKPT \
        exp_name="ms3_distill_1M_${TASK}" \
        steps=1000000

    echo "Finished $TASK."
    
    # Sleep for 10 seconds to ensure the GPU fully flushes its memory 
    # before PyTorch initializes the next environment.
    sleep 10 
done

echo "========================================================"
echo "ALL TASKS COMPLETED AND SAVED."
echo "========================================================"