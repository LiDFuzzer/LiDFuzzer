#!/bin/bash


# ------------------------------------------
# Params

export LABEL_PATH=/home/LiDFuzzer/tool/selected_data/semantic_kitti_labels/dataset/sequences/
export BIN_PATH=/home/LiDFuzzer/tool/selected_data/semantic_kitti_pcs/dataset/sequences/
export PRED_PATH=/home/LiDFuzzer/tool/pred_data
data=$BIN_PATH
pred=$PRED_PATH
model="cyl"


# ------------------------------------------
# Command


python modelPredTester.py -data "$data" -pred "$pred" -model "$model"


# ------------------------------------------

