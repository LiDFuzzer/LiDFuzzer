import numpy as np
import glob
import pickle
import copy
import json
import os
from pathlib import Path
import shutil
import pandas as pd

# --------------------------------------------------------------------------
# OPEN

"""
openLabel
For a specific sequence and scene
Opens a label file splitting between semantics, instances 
"""
def openLabelFile(labelFile):
    # Label
    label_arr = np.fromfile(labelFile, dtype=np.int32)
    semantics = label_arr & 0xFFFF
    instances = label_arr >> 16 

    return semantics, instances

"""
openBinFiles
For a specific sequence and scene
Opens a bin file splitting between xyz, intensity
"""
def openBinFile(binFile):
    # Bin File
    pcdArr = np.fromfile(binFile, dtype=np.float32)
    pcdArr = pcdArr.reshape((int(np.shape(pcdArr)[0]) // 4, 4))
    
    intensity = pcdArr[:, 3]
    pcdArr = np.delete(pcdArr, 3, 1)

    return pcdArr, intensity

"""
Setup step to get all scenes scan / label paths
"""
def getBinsLabels(binPath, labelPath):

    binFiles = []
    labelFiles = []
    binFilesSequence = np.array(glob.glob(binPath + "*.bin", recursive = True))
    labelFilesSequence = np.array(glob.glob(labelPath + "*.label", recursive = True))
    
    # Sort
    labelFilesSequence = sorted(labelFilesSequence)
    binFilesSequence = sorted(binFilesSequence)
    
    for labelFile in labelFilesSequence:
        labelFiles.append(labelFile)
    
    for binFile in binFilesSequence:
        binFiles.append(binFile)

    return binFiles, labelFiles

def getBins(binPath):
    binFiles = []
    binFilesSequence = np.array(glob.glob(binPath + "*.bin", recursive = True))
    binFilesSequence = sorted(binFilesSequence)
    for binFile in binFilesSequence:
        binFiles.append(binFile)
    
    return binFiles

def getLabels(labelPath):
    labelFiles = []
    labelFilesSequence = np.array(glob.glob(labelPath + "*.label", recursive = True))
    labelFilesSequence = sorted(labelFilesSequence)
    for labelFile in labelFilesSequence:
        labelFiles.append(labelFile)
    return labelFiles
# --------------------------------------------------------------------------
# SAVE

def saveLabelFile(labelFile, semantics, instances):
    labelsCombined = (instances << 16) | (semantics & 0xFFFF)
    labelsCombined = labelsCombined.astype(np.int32)
    labelsCombined.tofile(labelFile)

def write_record(text, path):
    with open(path, 'a') as file:
        file.write(text)

def record_dict_to_file(data, path):
    data = copy.deepcopy(data)
    with open(path, 'a') as file:
        # Write 'mod' data
        file.write("mod:\n")
        for key, value in data['mod'].items():
            file.write(f"{key}: {value}\n")
        # Write 'new' data
        file.write("\nnew:\n")
        for key, value in data['new'].items():
            file.write(f"{key}: {value}\n")
        # Write the rest of the data
        file.write("\n")
        del data['mod']
        del data['new']
        for key, value in data.items():
            if isinstance(value, dict):
                file.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    file.write(f"  {sub_key}: {sub_value}\n")
            else:
                file.write(f"{key}: {value}\n")


def writepopulation(filepath, context):
    with open(filepath, 'ab') as pkl_file:
        pickle.dump(context, pkl_file)

def readpopulation(filepath):
    with open(filepath, 'rb') as pkl_file:
        while True:
            try:
                data = pickle.load(pkl_file)
            except EOFError:
                break
    return data

def changeJsonLabel(filedict):
    labelFilesSequence = np.array(glob.glob(filedict + "*.json", recursive = True))
    labelFilesSequence = sorted(labelFilesSequence)
    parent_dir = Path(filedict).parent
    
    for labelFile in labelFilesSequence:
        with open(labelFile, 'r') as file:
            pts_semantic_mask = json.load(file)['pts_semantic_mask']
            pts_semantic_mask = np.array(pts_semantic_mask, dtype=np.int32)
            instances = np.zeros(np.shape(pts_semantic_mask)[0], np.int32)
            labelsCombined = (instances << 16) | (pts_semantic_mask & 0xFFFF)
        new_filename = os.path.join(parent_dir, os.path.basename(labelFile).replace('.json', '.label'))
        labelsCombined.tofile(new_filename)
    shutil.rmtree(filedict)


def save_individual_data(instances, save_path):
    """
    Creates a DataFrame based on the instances of individual data and saves it to a CSV file.
    
    Args:
        instances (list): List of instance data, typically individual.individual["instance"]
        save_path (str): Path to save the output CSV file
    """
    num_instances = len(instances)
    columns = []
    for i in range(num_instances):
        columns.append(f"ID_{i}")
        columns.append(f"Angle_{i}")

    df = pd.DataFrame(columns=columns)
    
    data = {}
    for index, instance_asset in enumerate(instances):
        asset = instance_asset.get_asset()
        
        id_key = f"ID_{index}"
        angle_key = f"Angle_{index}"
        
        data[id_key] = asset["_id"] if asset is not None else None
        data[angle_key] = instance_asset.get_angle()
    
    df = df.append(data, ignore_index=True)
    
    file_exists = os.path.exists(save_path)

    df.to_csv(save_path, mode='a', index=False, header=not file_exists)
    print(f"Data saved to {save_path}")