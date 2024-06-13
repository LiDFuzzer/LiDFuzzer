import numpy as np
import random
import open3d as o3d
import subprocess
import argparse
import os,sys
import glob
import yaml
from domain.semanticMapping import learning_map
from domain.semanticMapping import learning_map_inv
from domain.semanticMapping import name_label_mapping
from service.models.mmdetection3d import run_inference
import pymongo.errors
from pymongo import MongoClient
from data.fileIO import saveLabelFile, changeJsonLabel
import math
import shutil
from compression import copy_files_to_another_directory
from os import path as osp
from pathlib import Path

import mmengine
# make lookup table for mapping
maxkey = max(learning_map.keys())

# +100 hack making lut bigger just in case there are unknown labels
remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
remap_lut[list(learning_map.keys())] = list(learning_map.values())

fold_split = {
    'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    'val': [8],
    'trainval': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'test': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}
split_list = ['train', 'valid', 'trainval', 'test']


def get_semantickitti_info(split: str, filenumber : int) -> dict:
    data_infos = dict()
    data_infos['metainfo'] = dict(dataset='SemanticKITTI')
    data_list = []
    for i_folder in fold_split[split]:
        for j in range(filenumber):
            data_list.append({
                'lidar_points': {
                    'lidar_path':
                    osp.join('sequences',
                             str(i_folder).zfill(2), 'velodyne',
                             str(j).zfill(6) + '.bin'),
                    'num_pts_feats':
                    4
                },
                'pts_semantic_mask_path':
                osp.join('sequences',
                         str(i_folder).zfill(2), 'labels',
                         str(j).zfill(6) + '.label'),
                'sample_idx':
                str(i_folder).zfill(2) + str(j).zfill(6)
            })
    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_semantickitti_info_file(filenumber: int) -> None:
    save_path = "/home/semantickitti"
    pkl_prefix = 'semantickitti'
    print('Generate info.')
    save_path = Path(save_path)

    semantickitti_infos_val = get_semantickitti_info(split='val',filenumber = filenumber)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    if os.path.exists(filename):
        # 如果文件存在，则删除文件
        os.remove(filename)
    print(f'SemanticKITTI info val file is saved to {filename}')
    mmengine.dump(semantickitti_infos_val, filename)

# sys.path.append("./")
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))

from data.fileIO import openBinFile,openLabelFile,saveLabelFile
centerCamPoint = np.array([0, 0, 0.3])

def mongoConnect(mongoConnect):

    configFile = open(mongoConnect, "r")
    mongoUrl = configFile.readline()
    print("Connecting to mongodb")
    configFile.close()
    
    client = MongoClient(mongoUrl)

    db = client["lidar"]
    return db

#获取未标注的sign和pole实例并进行标注
def getSignPoleInstances(pcds, semantics, instances):

    uniqueInst = set()
    for instance in instances:
        uniqueInst.add(instance)

    mask_sign = (semantics == 81)
    mask_pole = (semantics == 80)

    onlysigns = pcds[mask_sign, :]
    onlypoles = pcds[mask_pole, :]

    indexsSignMask = np.where(mask_sign)
    indexsPoleMask = np.where(mask_pole)

    pcdSigns = o3d.geometry.PointCloud()
    pcdPoles = o3d.geometry.PointCloud()
    pcdSigns.points = o3d.utility.Vector3dVector(onlysigns)
    pcdPoles.points = o3d.utility.Vector3dVector(onlypoles)

    sign_labels = np.array(pcdSigns.cluster_dbscan(eps = 1, min_points = 2, print_progress = False))
    pole_labels = np.array(pcdPoles.cluster_dbscan(eps = 1, min_points = 2, print_progress = False))

    uniqueSignLabel = {}
    uniquePoleLabel = {}
    uniqueSeenSignLabel = set()
    uniqueSeenPoleLabel =set()
    signsfound = True
    polesfound = True
    if sign_labels.size == 0:
        signsfound = False
    else:

        for signlabel in sign_labels:
            if(signlabel != -1 and signlabel not in uniqueSeenSignLabel):

                uniqueSeenSignLabel.add(signlabel)

                newsignlabel = 0
                while(newsignlabel < 10):
                    randSignInsLable = random.randint(3000,5000)
                    if(randSignInsLable not in uniqueInst):
                        uniqueSignLabel[signlabel] = randSignInsLable
                        uniqueInst.add(randSignInsLable)
                        newsignlabel = randSignInsLable
                    else:
                        newsignlabel += 1


    if pole_labels.size == 0:
        polesfound = False
    else:
        
        for polelabel in pole_labels:
            if(polelabel != -1 and polelabel not in uniqueSeenPoleLabel):

                uniqueSeenPoleLabel.add(polelabel)

                newpolelabel = 0
                while(newpolelabel < 10):
                    randPoleInsLable = random.randint(1000,3000)
                    if(randPoleInsLable not in uniqueInst):
                        uniquePoleLabel[polelabel] = randPoleInsLable
                        uniqueInst.add(randPoleInsLable)
                        newpolelabel = randPoleInsLable
                    else:
                        newpolelabel += 1
    if signsfound == False:
        pass
    else:
        for sign in uniqueSignLabel.keys():
            currentSignIndexs = indexsSignMask[0][sign_labels == sign]
            instances[currentSignIndexs] = uniqueSignLabel[sign]
    
    if polesfound == False:
        pass
    else:
        for pole in uniquePoleLabel.keys():
            currentPoleIndexs = indexsPoleMask[0][pole_labels == pole]
            instances[currentPoleIndexs] = uniquePoleLabel[pole]

    return signsfound, polesfound, instances

def runModel(modelRunDir, runCommand , model, binpath, labelpath, predlabelpath):
    copy_files_to_another_directory(binpath, "/home/semantickitti/sequences/08/velodyne")
    copy_files_to_another_directory(labelpath, "/home/semantickitti/sequences/08/labels")
    binFiles = np.array(glob.glob(binpath + "*.bin", recursive = True))
    create_semantickitti_info_file(len(binFiles))

    if model == "Cylinder3D":
        if modelRunDir == None:
            modelRunDir = "/home/LiDFuzzer/mmdetection3d/tools"
        runCommand = "cd {} && {}".format(modelRunDir, runCommand)
        print(runCommand)
        returnCode = subprocess.Popen(runCommand, shell=True).wait()
    elif model == "SPVCNN":
        if modelRunDir == None:
            modelRunDir = "/home/LiDFuzzer/mmdetection3d/tools"
        runCommand = "cd {} && {}".format(modelRunDir, runCommand)
        print(runCommand)
        returnCode = subprocess.Popen(runCommand, shell=True).wait()
    elif model == "MinkuNet":
        if modelRunDir == None:
            modelRunDir = "/home/LiDFuzzer/mmdetection3d/tools"
        runCommand = "cd {} && {}".format(modelRunDir, runCommand)
        print(runCommand)
        returnCode = subprocess.Popen(runCommand, shell=True).wait()
    elif model == "FRNet":
        if modelRunDir == None:
            modelRunDir = "/home/LiDFuzzer/FRNet"
        runCommand = "cd {} && {}".format(modelRunDir, runCommand)
        print(runCommand)
        returnCode = subprocess.Popen(runCommand, shell=True).wait()
    if returnCode == 0:
        delete_files_in_directory("/home/semantickitti/sequences/08/velodyne")
        delete_files_in_directory("/home/semantickitti/sequences/08/labels")
        move_files_to_another_directory("/home/semantickitti/sequences/08/prediction",predlabelpath)
    return returnCode

def predict(model, binpath, labelpath, predlabelpath):
    modelstatu = -1
    if os.path.exists(predlabelpath):
        print("Prediction alrealdy have done!")
        modelstatu = 0
    else:
        os.makedirs(predlabelpath)
        if model == "Cylinder3D":
            # runCommand = "python3 demo_folder.py"
            # runCommand += " --demo-folder {}".format(binpath)
            # runCommand += " --save-folder {}".format(predlabelpath)
            # modelstatu = runModel(None, runCommand, model, binpath, labelpath)
            runCommand = "python test.py"
            runCommand += " /home/LiDFuzzer/mmdetection3d/configs/cylinder3d/cylinder3d_4xb4-3x_semantickitti.py"
            runCommand += " /home/LiDFuzzer/mmdetection3d/checkpoints/cylinder3d_4xb4_3x_semantickitti_20230318_191107-822a8c31.pth"
            modelstatu = runModel(None, runCommand, model, binpath, labelpath,predlabelpath)
        elif model == "SPVCNN":
            runCommand = "python test.py"
            runCommand += " /home/LiDFuzzer/mmdetection3d/configs/spvcnn/spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti.py"
            runCommand += " /home/LiDFuzzer/mmdetection3d/checkpoints/spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_125908-d68a68b7.pth"
            modelstatu = runModel(None, runCommand, model, binpath, labelpath,predlabelpath)

        elif model == "MinkuNet":
            runCommand = "python test.py"
            runCommand += " /home/LiDFuzzer/mmdetection3d/configs/minkunet/minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti.py"
            runCommand += " /home/LiDFuzzer/mmdetection3d/checkpoints/minkunet34_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti_20230512_233511-bef6cad0.pth"
            modelstatu = runModel(None, runCommand, model, binpath, labelpath,predlabelpath)

        elif model == "FRNet":
            runCommand = "python test.py"
            runCommand += " /home/LiDFuzzer/FRNet/configs/frnet/frnet-semantickitti_seg.py"
            runCommand += " /home/LiDFuzzer/FRNet/frnet-semantickitti_seg.pth"
            modelstatu = runModel(None, runCommand, model, binpath, labelpath,predlabelpath)
    return modelstatu


def copy_files_to_another_directory(orginal_directory, destination_directory):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # 遍历源文件夹下的所有文件
    for filename in os.listdir(orginal_directory):
        # 构建源文件的完整路径
        source_file = os.path.join(orginal_directory, filename)
        # 构建目标文件的完整路径
        destination_file = os.path.join(destination_directory, filename)
        # 将源文件复制到目标文件夹中
        shutil.copy(source_file, destination_file)

def delete_files_in_directory(directory):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        # 构建文件的完整路径
        file_path = os.path.join(directory, filename)
        # 如果是文件而不是文件夹，则删除该文件
        if os.path.isfile(file_path):
            os.remove(file_path)

def move_files_to_another_directory(orginal_directory, destination_directory):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # 遍历源文件夹下的所有文件
    for filename in os.listdir(orginal_directory):
        # 构建源文件的完整路径
        source_file = os.path.join(orginal_directory, filename)
        # 构建目标文件的完整路径
        destination_file = os.path.join(destination_directory, filename)
        # 将源文件移动到目标文件夹中
        shutil.move(source_file, destination_file)
        print(f"File '{source_file}' has been moved to '{destination_file}'.")

# def predict(model, binpath, predlabelpath):
#     if os.path.exists(predlabelpath):
#         print("Prediction alrealdy have done!")
#     else:
#         os.makedirs(predlabelpath)
#         if model == "Cylinder3D" or model == "SPVCNN" or model == "MinkuNet":
#             run_inference(model = model, pcd_data_list = [binpath], out_dir = predlabelpath)
#             changeJsonLabel(predlabelpath + "/preds/")
#         elif model == "SPVCNN":
#             pass
#         elif model == "MinkuNet":
#             pass


def labelInatanceModify(binpath, labelpath, labelpathmodel):
    # Get label / bin files
    binFiles = np.array(glob.glob(binpath + "*.bin", recursive = True))
    labelFiles = np.array(glob.glob(labelpath + "*.label", recursive = True))
    
    # Sort
    labelFiles = sorted(labelFiles)
    binFiles = sorted(binFiles)

    for index in range(len(labelFiles)):
        pcds, intensity = openBinFile(binFiles[index])
        semantics, instances = openLabelFile(labelFiles[index])
        signfound, polefound, instances_new = getSignPoleInstances(pcds, semantics, instances)
        filename = os.path.basename(labelFiles[index])
        if signfound or polefound:
            saveLabelFile(labelpathmodel + filename, semantics, instances_new)
        else:
            saveLabelFile(labelpathmodel + filename, semantics, instances)
        



#由于模型的训练会将一些语义进行转化，比如将moving-car转化为car，所以需要将原来的语义进行转化
def changeSemantics(semantics): 
    v_map = np.vectorize(lambda x: learning_map[x] if x in learning_map else x)
    semantics_mapped = v_map(semantics)
    v_inv_map = np.vectorize(lambda x: learning_map_inv[x] if x in learning_map_inv else x)
    semantics_final = v_inv_map(semantics_mapped)
    return semantics_final


# # Asset Prechecks
# def checkInclusionBasedOnTriangleMeshAsset(points, mesh):

#     obb = mesh.get_oriented_bounding_box()

#     legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

#     scene = o3d.t.geometry.RaycastingScene()
#     _ = scene.add_triangles(legacyMesh)

#     pointsVector = o3d.utility.Vector3dVector(points)

#     indexesWithinBox = obb.get_point_indices_within_bounding_box(pointsVector)
    
#     for idx in indexesWithinBox:
#         pt = points[idx]
#         query_point = o3d.core.Tensor([pt], dtype=o3d.core.Dtype.Float32)

#         occupancy = scene.compute_occupancy(query_point)
#         if (occupancy == 1): 
#             return True

#     return False

# Asset Prechecks
def checkInclusionBasedOnTriangleMeshAsset(points, mesh, type):

    obb = mesh.get_oriented_bounding_box()

    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(legacyMesh)

    pointsVector = o3d.utility.Vector3dVector(points)

    indexesWithinBox = obb.get_point_indices_within_bounding_box(pointsVector)
    occupy_num = 0
    for idx in indexesWithinBox:
        pt = points[idx]
        query_point = o3d.core.Tensor([pt], dtype=o3d.core.Dtype.Float32)

        occupancy = scene.compute_occupancy(query_point)
        if (occupancy == 1): 
            occupy_num += 1
    if indexesWithinBox:
        if float(occupy_num/len(indexesWithinBox)) > 0.3 and type != 10:
            return True
        else:
             return False
    return False


#判断点云在哪个环内
def assetPosition(pcdArr, instances, instance, cfg):
    # pcdsequence = o3d.geometry.PointCloud()
    # pcdsequence.points = o3d.utility.Vector3dVector(pcdArr)

    instancePoints = pcdArr[instances == instance]
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(instancePoints)
    obb = pcdAsset.get_oriented_bounding_box()
    assetCenter = obb.get_center()
    assetCenter_x = assetCenter[0]
    assetCenter_y = assetCenter[1]
    dist = math.sqrt((assetCenter_x - centerCamPoint[0])**2 + (assetCenter_y - centerCamPoint[1])**2)
    mean_dist = cfg["GAconfig"]["max_radius"]/cfg["GAconfig"]["gene_numbers"]
    dist_index = int(dist/mean_dist)

    return dist_index

def assetIsValid(pcdArr, instances, instance, semantics):
    
    pcdsequence = o3d.geometry.PointCloud()
    pcdsequence.points = o3d.utility.Vector3dVector(pcdArr)

    instancePoints = pcdArr[instances == instance]

    change_semantics = changeSemantics(semantics)
    mask = (instances == instance)
    orgtype = change_semantics[mask]
    counts = np.bincount(orgtype)
    typename = np.argmax(counts)
    # Acceptable number of points
    if (np.shape(instancePoints)[0] < 100) and typename == 10:
        return False
    elif(np.shape(instancePoints)[0] < 20):
        return  False

    pcdItem = o3d.geometry.PointCloud()
    pcdItem.points = o3d.utility.Vector3dVector(instancePoints)

    # Remove asset, unlabeled, outliers, and ground (parking, road, etc)
    maskInst = (instances != instance) & (semantics != 0) & (semantics != 1) & (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
    pcdWithoutInstance = pcdArr[maskInst, :]

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(instancePoints)

    #  Get the asset's bounding box
    obb = pcdAsset.get_oriented_bounding_box()
    assetCenter = obb.get_center()
    #计算相机与物体之间的距离
    dist = np.linalg.norm(centerCamPoint - assetCenter)

    # Dist is acceptable
    if dist > 50:
        return False

    # There are no points obsuring this asset's bounding box

    # Scale box based on distance
    scalingFactor = (dist / 150) + 1
    # print("Scaling {} Distance {}".format(scalingFactor, dist))
    obb.scale(scalingFactor, assetCenter)

    # Get the points that comprise the box
    boxPoints = np.asarray(obb.get_box_points())

    # Take the two lowest and move z dim down (removes any floating cars)
    boxPoints = boxPoints[boxPoints[:, 2].argsort()]
    boxPoints[:2, 2] -= 5
    
    # Create new mesh with the box and the center
    boxVertices = np.vstack((boxPoints, centerCamPoint))
    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(boxVertices)
    hull2, _ = pcdCastHull.compute_convex_hull()

    incuded = checkInclusionBasedOnTriangleMeshAsset(pcdWithoutInstance, hull2, typename)
        
    return not incuded

def poleTosign(pcds, instances, semantics):
    flag = False
    instance_poleTosign_dict = {}
    # 选出含有sign和pole且能聚类为sign的点，则必须含有sign的点
    mask_signAndpole = (instances >= 1000)
    mask_sign = (instances >= 3000)
    if np.any(mask_signAndpole) and np.any(mask_sign):
        SignAndPolePcds = pcds[mask_signAndpole]
        instancesSignAndPoles = instances[mask_signAndpole]
        semanticsSignAndPoles = semantics[mask_signAndpole]
        index_SignAndPole = np.where(mask_signAndpole)
        pcdsSignAndPole = o3d.geometry.PointCloud()
        pcdsSignAndPole.points = o3d.utility.Vector3dVector(SignAndPolePcds)
        labels = np.array(pcdsSignAndPole.cluster_dbscan(eps = 2, min_points = 10, print_progress = False))
        max_label = labels.max()
        if max_label < 0:
            pass
        else:
            uniqueSeenLabels = set()
            for labelNum in labels:
                if(labelNum != -1 and labelNum not in uniqueSeenLabels):
                    uniqueSeenLabels.add(labelNum)

                    semanticsSignAndPole = semanticsSignAndPoles[labels == labelNum]
                    instancesSignAndPole = instancesSignAndPoles[labels == labelNum]

                    types = set()
                    for sem in semanticsSignAndPole:
                        types.add(sem)
                    if len(types) > 1:
                        pointsSignAndPole = SignAndPolePcds[labels == labelNum]
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pointsSignAndPole)
                        obb = pcd.get_oriented_bounding_box()
                        maxXy = obb.get_max_bound()
                        minXy = obb.get_min_bound()
                        maxXy[2] = 0
                        minXy[2] = 0
                        distAcross = np.linalg.norm(maxXy - minXy)
                        if (distAcross < 3):
                            #将所有pole实例的序号改为sign的序号
                            typesinstances = set()
                            flag = True
                            for instance in instancesSignAndPole:
                                typesinstances.add(instance)
                            sorted_list = sorted(list(typesinstances))
                            fristSignIndex = next((i for i, x in enumerate(sorted_list) if x >= 3000), None)
                            instance_key = sorted_list[fristSignIndex]
                            sorted_list.remove(sorted_list[fristSignIndex])
                            instance_poleTosign_dict[instance_key] = sorted_list
    return instance_poleTosign_dict, flag

# #计算每个实例预测的精度并进行挑选
# def instancesSelect(binpath, labelpath, predlabelpath, cfg, mdbInstanceAssets, model):
#     binFiles = np.array(glob.glob(binpath + "*.bin", recursive = True))
#     labelFiles = np.array(glob.glob(labelpath + "*.label", recursive = True))
#     predlabelFiles = np.array(glob.glob(predlabelpath + "*.label", recursive = True))

#     binFiles = sorted(binFiles)
#     labelFiles = sorted(labelFiles)
#     predlabelFiles = sorted(predlabelFiles)
#     instances_assets = []
#     for index in range(len(labelFiles)):
#         pcds, intensity = openBinFile(binFiles[index])
#         semantics, instances = openLabelFile(labelFiles[index])
#         change_semantics = changeSemantics(semantics)
#         pred_semantics, pred_instances = openLabelFile(predlabelFiles[index])
#         instance_poleTosign_dict, flag = poleTosign(pcds, instances, semantics)
#         instancesAcc_dict = {}
#         instantSet = set()
#         for instance in instances:
#             instantSet.add(instance)
#         for instance in instantSet:
#             if instance == 0:
#                 continue
#             mask = (instances == instance)
#             orgtype = change_semantics[mask]
#             predtype = pred_semantics[mask]
#             count = 0
#             for ele in predtype:
#                 if ele == orgtype[0]:
#                     count = count + 1
#             accuracy = float(count/orgtype.size)
#             orgtypename = name_label_mapping[orgtype[0]]
#             if orgtypename == "person" or orgtypename == "bicycle" or orgtypename == "motorcycle" or orgtypename == "truck" or orgtypename == "bicyclist" or orgtypename == "motorcyclist":
#                 print(instance, orgtypename, accuracy, orgtype.size)
#             if accuracy >= cfg["accuracy"][orgtypename]:
#                 instancesAcc_dict[instance] = accuracy
#         #将精度超过阈值的sign和pole进行实例合并
#         merge_instances = {}
#         print(instancesAcc_dict)
#         copy_instances = np.copy(instances)
#         if flag == True:
#             for key, values in instance_poleTosign_dict.items():
#                 all_flag = True
#                 if key not in instancesAcc_dict:
#                     all_flag = False
#                 else:
#                     for value in values:
#                         if value not in instancesAcc_dict:
#                             all_flag = False

#                 if all_flag:
#                     merge_instances[key] = values
#                     for value in values:
#                         instances[copy_instances == value] = key

#                         instancesAcc_dict[key] += instancesAcc_dict.pop(value)
#                     instancesAcc_dict[key] = float(instancesAcc_dict[key]/(len(instance_poleTosign_dict[key]) +1))
#         inset = set()
#         for ins in instances:
#             inset.add(ins)
#         # print(inset)
#         for accinstance in instancesAcc_dict:
#             if accinstance >= 1000 and accinstance <= 3000:
#                 continue
#             valid_flag = assetIsValid(pcds, instances, accinstance, semantics)
#             print(accinstance, valid_flag)
#             if valid_flag:
#                 # saveLabelFile(labelFiles[index], semantics, instances)
#                 asset = {}
#                 # print(accinstance)
#                 # 拆分路径
#                 sequence_path, file_name = os.path.split(labelFiles[index])
#                 # 再次拆分序列路径
#                 sequence_dir, sequence_number = os.path.split(sequence_path)
#                 # 获取上一级目录的名称
#                 parent_directory = os.path.basename(sequence_dir)

#                 accinstance_mask = (instances == accinstance)
#                 accinstance_semantic = change_semantics[accinstance_mask]

#                 asset["_id"] = parent_directory + "-" + file_name.split('.')[0] + "-" + str(accinstance) + "-" + name_label_mapping[accinstance_semantic[0]]
#                 asset["sequence"] = parent_directory
#                 asset["scene"] = file_name.split('.')[0]
#                 asset["type"] = name_label_mapping[accinstance_semantic[0]]
#                 asset["type-semantic"] = int(accinstance_semantic[0])
#                 asset["type-points"] = int(np.count_nonzero(accinstance_mask))
#                 asset["accuracy"] = instancesAcc_dict[accinstance]
#                 # asset["instances"] = int(accinstance)
#                 if accinstance in merge_instances:
#                     merge_instances[accinstance].append(accinstance)
#                     all_instances_list = [int(x) for x in merge_instances[accinstance]]
#                     asset["all-instances"] = all_instances_list
#                 else:
#                     asset["all-instances"] = int(accinstance)
#                 asset["model"] = model
#                 asset["dist_index"] = assetPosition(pcds, instances, accinstance, cfg)
#                 print(asset)
#                 instances_assets.append(asset)
#         if (len(instances_assets) >= 2000):
#             print("save batch of 2000 instances of {}".format(parent_directory))
#             # Batch insert
#             try:
#                 mdbInstanceAssets.insert_many(instances_assets)
#             except pymongo.errors.BulkWriteError as e:
#                 # https://stackoverflow.com/questions/44838280/how-to-ignore-duplicate-key-errors-safely-using-insert-many
#                 panic = list(filter(lambda x: x['code'] != 11000, e.details['writeErrors']))
#                 if len(panic) > 0:
#                     raise RuntimeError("Error writing to mongo", e)

#             instances_assets = []

#     # Batch insert any remaining asset details
#     if (len(instances_assets) != 0):
#         print('Saving remaining', len(instances_assets), 'instance assets')
#         try:
#             mdbInstanceAssets.insert_many(instances_assets)
#         except pymongo.errors.BulkWriteError as e:
#             # https://stackoverflow.com/questions/44838280/how-to-ignore-duplicate-key-errors-safely-using-insert-many
#             panic = list(filter(lambda x: x['code'] != 11000, e.details['writeErrors']))
#             if len(panic) > 0:
#                 raise RuntimeError("Error writing to mongo", e)
    
#计算每个实例预测的精度并进行挑选
def instancesSelect(binpath, labelpath, predlabelpath, cfg, mdbInstanceAssets, model):
    binFiles = np.array(glob.glob(binpath + "*.bin", recursive = True))
    labelFiles = np.array(glob.glob(labelpath + "*.label", recursive = True))
    predlabelFiles = np.array(glob.glob(predlabelpath + "*.label", recursive = True))

    binFiles = sorted(binFiles)
    labelFiles = sorted(labelFiles)
    predlabelFiles = sorted(predlabelFiles)
    predlabelFiles = [pred_file for pred_file in predlabelFiles if any(label_file.split('/')[-1] == pred_file.split('/')[-1] for label_file in labelFiles)]
    instances_assets = []
    for index in range(len(labelFiles)):
        pcds, intensity = openBinFile(binFiles[index])
        semantics, instances = openLabelFile(labelFiles[index])
        change_semantics = changeSemantics(semantics)
        change_semantics = remap_lut[change_semantics]

        pred_semantics, pred_instances = openLabelFile(predlabelFiles[index])
        instance_poleTosign_dict, flag = poleTosign(pcds, instances, semantics)
        instancesAcc_dict = {}
        instantSet = set()
        for instance in instances:
            instantSet.add(instance)
        for instance in instantSet:
            if instance == 0:
                continue
            mask = (instances == instance)
            orgtype = change_semantics[mask]
            counts = np.bincount(orgtype)
            typeNum = np.argmax(counts)
            predtype = pred_semantics[mask]
            count = 0
            for ele in predtype:
                if ele == typeNum:
                    count = count + 1
            accuracy = float(count/orgtype.size)
            orgtypename = name_label_mapping[learning_map_inv[typeNum]]
            if orgtypename == "person" or orgtypename == "bicycle" or orgtypename == "motorcycle" or orgtypename == "truck" or orgtypename == "bicyclist" or orgtypename == "motorcyclist":
                print(instance, orgtypename, accuracy, orgtype.size)
            if accuracy >= cfg["accuracy_no"][orgtypename]:
                instancesAcc_dict[instance] = accuracy
        #将精度超过阈值的sign和pole进行实例合并
        merge_instances = {}
        copy_instances = np.copy(instances)
        if flag == True:
            for key, values in instance_poleTosign_dict.items():
                all_flag = True
                if key not in instancesAcc_dict:
                    all_flag = False
                else:
                    for value in values:
                        if value not in instancesAcc_dict:
                            all_flag = False

                if all_flag:
                    merge_instances[key] = values
                    for value in values:
                        instances[copy_instances == value] = key

                        instancesAcc_dict[key] += instancesAcc_dict.pop(value)
                    instancesAcc_dict[key] = float(instancesAcc_dict[key]/(len(instance_poleTosign_dict[key]) +1))
        inset = set()
        for ins in instances:
            inset.add(ins)
        # print(inset)
        for accinstance in instancesAcc_dict:
            if accinstance >= 1000 and accinstance <= 3000:
                continue
            valid_flag = assetIsValid(pcds, instances, accinstance, semantics)
            print(accinstance, valid_flag)
            if valid_flag:
                # saveLabelFile(labelFiles[index], semantics, instances)
                asset = {}
                # print(accinstance)
                # 拆分路径
                sequence_path, file_name = os.path.split(labelFiles[index])
                # 再次拆分序列路径
                sequence_dir, sequence_number = os.path.split(sequence_path)
                # 获取上一级目录的名称
                parent_directory = os.path.basename(sequence_dir)

                accinstance_mask = (instances == accinstance)
                accinstance_semantic = change_semantics[accinstance_mask]
                accinstance_count = np.bincount(accinstance_semantic)
                accinstance_typename  = np.argmax(accinstance_count)
                asset["_id"] = parent_directory + "-" + file_name.split('.')[0] + "-" + str(accinstance) + "-" + name_label_mapping[learning_map_inv[accinstance_typename]]
                asset["sequence"] = parent_directory
                asset["scene"] = file_name.split('.')[0]
                asset["type"] = name_label_mapping[learning_map_inv[accinstance_typename]]
                asset["type-semantic"] = int(accinstance_typename)
                asset["type-points"] = int(np.count_nonzero(accinstance_mask))
                # asset["accuracy"] = instancesAcc_dict[accinstance]
                # asset["instances"] = int(accinstance)
                if accinstance in merge_instances:
                    merge_instances[accinstance].append(accinstance)
                    all_instances_list = [int(x) for x in merge_instances[accinstance]]
                    asset["all-instances"] = all_instances_list
                else:
                    asset["all-instances"] = int(accinstance)
                # asset["model"] = model
                asset["dist_index"] = assetPosition(pcds, instances, accinstance, cfg)
                print(asset)
                instances_assets.append(asset)
        if (len(instances_assets) >= 2000):
            print("save batch of 2000 instances of {}".format(parent_directory))
            # Batch insert
            try:
                mdbInstanceAssets.insert_many(instances_assets)
            except pymongo.errors.BulkWriteError as e:
                # https://stackoverflow.com/questions/44838280/how-to-ignore-duplicate-key-errors-safely-using-insert-many
                panic = list(filter(lambda x: x['code'] != 11000, e.details['writeErrors']))
                if len(panic) > 0:
                    raise RuntimeError("Error writing to mongo", e)

            instances_assets = []

    # Batch insert any remaining asset details
    if (len(instances_assets) != 0):
        print('Saving remaining', len(instances_assets), 'instance assets')
        try:
            mdbInstanceAssets.insert_many(instances_assets)
        except pymongo.errors.BulkWriteError as e:
            # https://stackoverflow.com/questions/44838280/how-to-ignore-duplicate-key-errors-safely-using-insert-many
            panic = list(filter(lambda x: x['code'] != 11000, e.details['writeErrors']))
            if len(panic) > 0:
                raise RuntimeError("Error writing to mongo", e)

def instancesSelectNoaccuracy(binpath, labelpath, cfg, mdbInstanceAssets):
    binFiles = np.array(glob.glob(binpath + "*.bin", recursive = True))
    labelFiles = np.array(glob.glob(labelpath + "*.label", recursive = True))

    binFiles = sorted(binFiles)
    labelFiles = sorted(labelFiles)
    instances_assets = []
    for index in range(len(labelFiles)):
        pcds, intensity = openBinFile(binFiles[index])
        semantics, instances = openLabelFile(labelFiles[index])
        change_semantics = changeSemantics(semantics)
        change_semantics = remap_lut[change_semantics]

        instance_poleTosign_dict, flag = poleTosign(pcds, instances, semantics)
        print(instance_poleTosign_dict)
        instancesAcc_dict = {}
        instantSet = set()
        for instance in instances:
            instantSet.add(instance)
        for instance in instantSet:
            if instance == 0:
                continue
            mask = (instances == instance)
            orgtype = change_semantics[mask]
            counts = np.bincount(orgtype)
            typeNum = np.argmax(counts)
            orgtypename = name_label_mapping[learning_map_inv[typeNum]]
            instancesAcc_dict[instance] = 0
        #将精度超过阈值的sign和pole进行实例合并

        merge_instances = {}
        copy_instances = np.copy(instances)
        if flag == True:
            for key, values in instance_poleTosign_dict.items():
                all_flag = True
                if key not in instancesAcc_dict:
                    all_flag = False
                else:
                    for value in values:
                        if value not in instancesAcc_dict:
                            all_flag = False

                if all_flag:
                    merge_instances[key] = values
                    for value in values:
                        instances[copy_instances == value] = key

                        instancesAcc_dict[key] += instancesAcc_dict.pop(value)
                    instancesAcc_dict[key] = float(instancesAcc_dict[key]/(len(instance_poleTosign_dict[key]) +1))
        inset = set()
        for ins in instances:
            inset.add(ins)
        # print(inset)
        print(instancesAcc_dict)
        for accinstance in instancesAcc_dict:
            if accinstance >= 1000 and accinstance <= 3000:
                continue
            valid_flag = assetIsValid(pcds, instances, accinstance, semantics)
            print(accinstance, valid_flag)
            if valid_flag:
                # saveLabelFile(labelFiles[index], semantics, instances)
                asset = {}
                # print(accinstance)
                # 拆分路径
                sequence_path, file_name = os.path.split(labelFiles[index])
                # 再次拆分序列路径
                sequence_dir, sequence_number = os.path.split(sequence_path)
                # 获取上一级目录的名称
                parent_directory = os.path.basename(sequence_dir)

                accinstance_mask = (instances == accinstance)
                accinstance_semantic = change_semantics[accinstance_mask]
                accinstance_count = np.bincount(accinstance_semantic)
                accinstance_typename  = np.argmax(accinstance_count)
                asset["_id"] = parent_directory + "-" + file_name.split('.')[0] + "-" + str(accinstance) + "-" + name_label_mapping[learning_map_inv[accinstance_typename]]
                asset["sequence"] = parent_directory
                asset["scene"] = file_name.split('.')[0]
                asset["type"] = name_label_mapping[learning_map_inv[accinstance_typename]]
                asset["type-semantic"] = int(accinstance_typename)
                asset["type-points"] = int(np.count_nonzero(accinstance_mask))
                # asset["instances"] = int(accinstance)
                if accinstance in merge_instances:
                    merge_instances[accinstance].append(accinstance)
                    all_instances_list = [int(x) for x in merge_instances[accinstance]]
                    asset["all-instances"] = all_instances_list
                else:
                    asset["all-instances"] = int(accinstance)
                asset["dist_index"] = assetPosition(pcds, instances, accinstance, cfg)
                print(asset)
                instances_assets.append(asset)
        if (len(instances_assets) >= 2000):
            print("save batch of 2000 instances of {}".format(parent_directory))
            # Batch insert
            try:
                mdbInstanceAssets.insert_many(instances_assets)
            except pymongo.errors.BulkWriteError as e:
                # https://stackoverflow.com/questions/44838280/how-to-ignore-duplicate-key-errors-safely-using-insert-many
                panic = list(filter(lambda x: x['code'] != 11000, e.details['writeErrors']))
                if len(panic) > 0:
                    raise RuntimeError("Error writing to mongo", e)

            instances_assets = []

    # Batch insert any remaining asset details
    if (len(instances_assets) != 0):
        print('Saving remaining', len(instances_assets), 'instance assets')
        try:
            mdbInstanceAssets.insert_many(instances_assets)
        except pymongo.errors.BulkWriteError as e:
            # https://stackoverflow.com/questions/44838280/how-to-ignore-duplicate-key-errors-safely-using-insert-many
            panic = list(filter(lambda x: x['code'] != 11000, e.details['writeErrors']))
            if len(panic) > 0:
                raise RuntimeError("Error writing to mongo", e)

def parse_args():
    p = argparse.ArgumentParser(
        description='instances extract')
    # p.add_argument("-orginalbinPath", 
    #     help="Path to the ariginal semanticKITTI sequences bins", 
    #     nargs='?', const="",
    #     default="")
    # p.add_argument("-orginallabelPath", 
    #     help="Path to the ariginal semanticKITTI sequences labels", 
    #     nargs='?', const="",
    #     default="")
    p.add_argument("-binPath", 
        help="Path to the compress semanticKITTI sequences bins", 
        nargs='?', const="",
        default="")
    p.add_argument("-labelPath", 
        help="Path to the compress semanticKITTI sequences lables", 
        nargs='?', const="",
        default="")
    # p.add_argument("-predlabelPath",
    #     help="Path to the predicted semanticKITTI labels",
    #     nargs='?', const="",
    #     default="")
    # p.add_argument("-model",
    #     help="model to predict semanticKITTI bins",
    #     nargs='?', const="",
    #     default="")
    # p.add_argument(
    #   '-mdb', 
    #   type=str,
    #   required=False,
    #   default="mongodb.txt",
    #   help='mongodb config file. Defaults to %(default)s')
    # p.add_argument(
    #   '-config',
    #   type=str,
    #   required=False,
    #   default="config.yaml",
    #   help='Dataset config file. Defaults to %(default)s')

    return p.parse_args()

def main():
    args = parse_args()
    # print("Connecting to Mongo")
    # mdb = mongoConnect(args.mdb)
    # model = args.model
    # mdbColAssets = mdb["assets4"]
    # mdbColAssetMetadata = mdb["asset_metadata4"]
    # print("Connected")
    # if mdb is not None:
    #     print("Connected to MongoDB")
    #     mdb_name = "instanceAssets_five"
    #     mdbInstanceAssets = mdb[mdb_name]
    # else:
    #     print("Failed to connect to MongoDB")

    # orginal_binpath = os.path.normpath(args.orginalbinPath) +  "/"
    # orginal_labelpath = os.path.normpath(args.orginallabelPath) +  "/"

    binpath = os.path.normpath(args.binPath) +  "/"
    labelpath = os.path.normpath(args.labelPath) +  "/"
    binpathchange = args.binPath + "instance/"
    labelpathchange = args.labelPath + "instance/"
    # predlabelpath = os.path.normpath(args.predlabelPath) +  "/"
    if not os.path.exists(binpathchange):
        os.makedirs(binpathchange)
        copy_files_to_another_directory(binpath,binpathchange)
    if not os.path.exists(labelpathchange):
        os.makedirs(labelpathchange)
        labelInatanceModify(binpath, labelpath, labelpathchange)
    # # open config file
    # try:
    #     print("Opening config file %s" % args.config)
    #     cfg = yaml.safe_load(open(args.config, 'r'))
    # except Exception as e:
    #     print(e)
    #     print("Error opening yaml file.")
    #     quit()

    # predict(model, orginal_binpath, orginal_labelpath, predlabelpath)    
    # # instancesSelect(binpathchange, labelpathchange, predlabelpath,cfg, mdbInstanceAssets,model)
    # instancesSelectNoaccuracy(binpathchange, labelpathchange, cfg, mdbInstanceAssets)


if __name__ == '__main__':
    main()
    # create_semantickitti_info_file(2)