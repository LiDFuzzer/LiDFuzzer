#!/bin/bash


# ------------------------------------------
# Params

export LABEL_PATH=/home/semLidarFuzz/tool/selected_data/semantic_kitti_labels/dataset/sequences/
export BIN_PATH=/home/semLidarFuzz/tool/selected_data/semantic_kitti_pcs/dataset/sequences/
export PRED_PATH=/home/semLidarFuzz/tool/pred_data
data=$BIN_PATH
pred=$PRED_PATH
model="cyl"


# ------------------------------------------
# Command


python modelPredTester.py -data "$data" -pred "$pred" -model "$model"


# ------------------------------------------

