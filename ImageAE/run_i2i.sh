#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=3
python test_alae -c configs \
	OUTPUT_DIR training_artifacts/ \
	LANGEVIN.STEP 15 LANGEVIN.LR 1.0 \
	EBM.LR 0.01 EBM.LAYER 2 EBM.HIDDEN 1024 \
