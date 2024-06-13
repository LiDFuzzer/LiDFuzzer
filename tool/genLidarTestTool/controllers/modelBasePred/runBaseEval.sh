#!/bin/bash


------------------------------------------
# Move on to tool setup
export PYTHONPATH=/home/semLidarFuzz/tool/genLidarTestTool
export DISPLAY=":0"
# export RUNNING_IN_DOCKER=gen_lidar_tests
export MONGO_CONNECT=/home/semLidarFuzz/tool/mongoconnect.txt
export MODEL_DIR=/home/semLidarFuzz/tool/genLidarTestTool/suts
export STAGING_DIR=/home/semLidarFuzz/tool/genLidarTestTool/tmp/dataset/sequences/00/velodyne/
export PRED_PATH=/home/semLidarFuzz/tool/pred_data/
export LABEL_PATH=/home/semLidarFuzz/tool/selected_data/semantic_kitti_labels/dataset/sequences/
export BIN_PATH=/home/semLidarFuzz/tool/selected_data/semantic_kitti_pcs/dataset/sequences/
export MODELS="cyl"

labelBasePath=$LABEL_PATH
predBasePath=$PRED_PATH


# ------------------------------------------

for model in ${MODELS//,/ }
do
  python3 modelBasePred.py -bins $BIN_PATH -pred $predBasePath -model $model -modelDir $MODEL_DIR -stage $STAGING_DIR
done
source movePreds.sh
python3 modelEvaluationInitial.py -labels $labelBasePath -pred $predBasePath -models $MODELS


# ------------------------------------------



