#!/bin/bash
# One-shot retrain: backs up the current model, then launches training in the
# background with logging. Run from the project root with your conda env active:
#     bash train.sh
#
# To use fewer epochs (faster), edit num_train_epochs in
# training/train_segformer_cpu.py before running.

set -e
cd "$(dirname "$0")"

# 1. Back up current weights so you can revert if the retrain is worse
if [ -d backend/models/segformer-b0-final ]; then
    rm -rf backend/models/segformer-b0-final-backup
    cp -r backend/models/segformer-b0-final backend/models/segformer-b0-final-backup
    echo "✅ Backed up current model -> backend/models/segformer-b0-final-backup"
fi

# 2. Launch training in the background, logging to train.log
nohup python training/train_segformer_cpu.py > train.log 2>&1 &
echo "✅ Training started (PID $!). Keep the laptop awake + plugged in."
echo "   Watch progress:   tail -f train.log"
echo "   When it prints 'Saved to ...', restart the app to load new weights."
