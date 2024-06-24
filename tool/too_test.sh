export PYTHONPATH=/home/LiDFuzzer/tool/genLidarTestTool
export DISPLAY=":0"
export RUNNING_IN_DOCKER=gen_lidar_tests
export MONGO_CONNECT=/home/LiDFuzzer/tool/mongoconnect.txt
export MODEL_DIR=/home/LiDFuzzer/tool/genLidarTestTool/suts
export STAGING_DIR=/home/LiDFuzzer/tool/genLidarTestTool/tmp/dataset/sequences/01/velodyne/
export PRED_PATH=/home/LiDFuzzer/tool/pred_data/
export LABEL_PATH=/home/LiDFuzzer/tool/selected_data/semantic_kitti_labels/dataset/sequences/
export BIN_PATH=/home/LiDFuzzer/tool/selected_data/semantic_kitti_pcs/dataset/sequences/
export MODELS="cyl"

mkdir $PRED_PATH

cd /home/LiDFuzzer/tool/genLidarTestTool/controllers/modelBasePred/ && source runBaseEval.sh