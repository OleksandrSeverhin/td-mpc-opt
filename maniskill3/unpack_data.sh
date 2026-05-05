#!/bin/bash

TASKS=(
    "PushCube-v1"
    "PickCube-v1"
    "StackCube-v1"
    "PegInsertionSide-v1"
    "PlugCharger-v1"
)

echo "Unpacking missing 24-dim observations using SAPIEN physics..."

for TASK in "${TASKS[@]}"; do
    echo "Replaying $TASK..."
    python -m mani_skill.trajectory.replay_trajectory \
        --traj-path ~/.maniskill/demos/${TASK}/motionplanning/trajectory.h5 \
        --use-env-states \
        --obs-mode state \
        --save-traj
done

echo "Unpacking complete!"