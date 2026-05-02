#!/bin/bash

TASKS=(
    "PushCube-v1"
    "PickCube-v1"
    "StackCube-v1"
    "PegInsertionSide-v1"
)

TEACHER_CKPT="/root/td-mpc-opt/tdmpc2/models/multitask/mt30-317M.pt"

for TASK in "${TASKS[@]}"; do
    echo "========================================================"
    echo "Starting distillation for: $TASK"
    echo "========================================================"
    python train_distill.py \
        env_id=$TASK \
        model_size=1 \
        checkpoint=$TEACHER_CKPT \
        exp_name="ms3_distill_1M_${TASK}" \
        steps=1000000
    echo "Finished $TASK."
    sleep 10 
done

echo "========================================================"
echo "ALL TASKS COMPLETED AND SAVED."
echo "========================================================"