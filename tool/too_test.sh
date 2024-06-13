export PYTHONPATH=/home/semLidarFuzz/tool/genLidarTestTool
export DISPLAY=":0"
export RUNNING_IN_DOCKER=gen_lidar_tests
export MONGO_CONNECT=/home/semLidarFuzz/tool/mongoconnect.txt
export MODEL_DIR=/home/semLidarFuzz/tool/genLidarTestTool/suts
export STAGING_DIR=/home/semLidarFuzz/tool/genLidarTestTool/tmp/dataset/sequences/01/velodyne/
export PRED_PATH=/home/semLidarFuzz/tool/pred_data/
export LABEL_PATH=/home/semLidarFuzz/tool/selected_data/semantic_kitti_labels/dataset/sequences/
export BIN_PATH=/home/semLidarFuzz/tool/selected_data/semantic_kitti_pcs/dataset/sequences/
export MODELS="cyl"

mkdir $PRED_PATH

cd /home/semLidarFuzz/tool/genLidarTestTool/controllers/modelBasePred/ && source runBaseEval.sh