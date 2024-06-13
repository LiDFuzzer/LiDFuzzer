"""
eval 
Handles evaluation of predictions files, runs models, updates the final analytics 

@Date 6/23/22
"""


import glob, os
import numpy as np
from os.path import basename
import shutil
from operator import itemgetter
import time
import subprocess
import sys
sys.path.append("..")
sys.path.append("../..")
# sys.path.append("/.")
import service.eval.ioueval as ioueval

import data.baseAccuracyRepository as baseAccuracyRepository
import data.fileIoUtil as fileIoUtil
import data.fileIO as fileIO
from domain.semanticMapping import learning_map_inv
from domain.semanticMapping import learning_map_inv
from domain.semanticMapping import learning_map
from domain.semanticMapping import learning_ignore
from domain.semanticMapping import name_label_mapping
from domain.modelConstants import Models

# from service.models.cylRunner import CylRunner
# from service.models.js3cCPURunner import JS3CCPURunner
# from service.models.js3cGPURunner import JS3CGPURunner
# from service.models.spvRunner import SpvRunner
# from service.models.salRunner import SalRunner
# from service.models.sq3Runner import Sq3Runner
# from service.models.polRunner import PolRunner
# from service.models.ranRunner import RanRunner

import domain.config as config

# --------------------------------------------------------------------------
# Constants for the ioueval evaluator

numClasses = len(learning_map_inv)

# make lookup table for mapping
maxkey = max(learning_map.keys())

# +100 hack making lut bigger just in case there are unknown labels
remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
remap_lut[list(learning_map.keys())] = list(learning_map.values())

# create evaluator
ignore = []
for cl, ign in learning_ignore.items():
    if ign:
        x_cl = int(cl)
        ignore.append(x_cl)

evaluator = ioueval.iouEval(numClasses, ignore)


# --------------------------------------------------------------------------
# Evaluators

"""
evalLabels

Evaluates the modified prediction and new prediction against our mutation ground truth semantic label
"""
def evalLabels(modifiedLabel, modifiedPrediction, newPrediction, individual, model):
    # Set up results
    results = {}
    results["mod"] = {}
    results["mod"]["classes"] = {}
    results["new"] = {}
    results["new"]["classes"] = {}
    accMod, jaccMod, classJaccMod = None, None, None
    accNew, jaccNew, classJaccNew = None, None, None
    # if model == "Cylinder3D":
    #     accMod, jaccMod, classJaccMod = evaluateLabelPred(modifiedLabel, modifiedPrediction)
    #     accNew, jaccNew, classJaccNew = evaluateLabelPred(modifiedLabel, newPrediction)
    # # accNew, jaccNew, classJaccNew = evaluateNewLabelPred(modifiedLabel, newPrediction, individual)
    # elif model == "SPVCNN" or model == "MinkuNet" or model == "FRNet":
    accMod, jaccMod, classJaccMod = evaluateLabelPredNoMap(modifiedLabel, modifiedPrediction, individual)
    accNew, jaccNew, classJaccNew = evaluateLabelPredNoMap(modifiedLabel, newPrediction, individual)
    # save classwise jaccard
    for i, jacc in enumerate(classJaccMod):
        if i not in ignore:
            results["mod"][name_label_mapping[learning_map_inv[i]]] = jacc
    for i, jacc in enumerate(classJaccNew):
        if i not in ignore:
            results["new"][name_label_mapping[learning_map_inv[i]]] = jacc
    # Save acc
    results["mod"]["accuracy"] = accMod
    results["new"]["accuracy"] = accNew
    # Save jacc
    results["mod"]["jaccard"] = jaccMod
    results["new"]["jaccard"] = jaccNew

    # Get percent loss
    results["accuracyChange"] = accNew - accMod
    results["jaccardChange"] = jaccNew - jaccMod
    results["percentLossAcc"] = results["accuracyChange"] * 100
    results["percentLossJac"] = results["jaccardChange"] * 100
    if individual.IfInstance == True:
        for index, instance_asset in enumerate(individual.individual["instance"]):
            asset = instance_asset.get_asset()
            results[index] = asset
        # fileIO.save_individual_data(individual.individual["instance"],"output_data.csv")
    if individual.IfWeather == True: 
        weather_type = individual.individual["weather"][0].get_weather_type()
        results["weather"] = weather_type
    return results

"""
evaluateLabelPred

Perfoms evaluation of one label file
Against one prediction File
Returns accuracy, jaccard, and class jaccard
"""
def evaluateLabelPred(labelFile, predictionFile):
    global evaluator

    groundTruth, _ = fileIoUtil.openLabelFile(labelFile)
    prediction, _ = fileIoUtil.openLabelFile(predictionFile)

    # Map to correct classes for evaluation,
    # Example classification "moving-car" -> "car"

    groundTruthMapped = remap_lut[groundTruth] # remap to xentropy format
    predictionMapped = remap_lut[prediction] # remap to xentropy format

    # Prepare evaluator
    evaluator.reset()
    evaluator.addBatch(predictionMapped, groundTruthMapped)
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    return m_accuracy, m_jaccard, class_jaccard

def evaluateLabelPredNoMap(labelFile, predictionFile, individual):
    global evaluator

    groundTruth, _ = fileIoUtil.openLabelFile(labelFile)
    prediction, _ = fileIoUtil.openLabelFile(predictionFile)
    if individual.IfWeather == True: 
        weather_type = individual.individual["weather"][0].get_weather_type()
        indice = individual.individual["weather"][0].get_indice()
        if weather_type == "snow" or weather_type == "fog":
            groundTruth = groundTruth[:-indice]
            prediction = prediction[:-indice]
    groundTruthMapped = remap_lut[groundTruth]
    # Map to correct classes for evaluation,
    # Example classification "moving-car" -> "car"
    # Prepare evaluator
    evaluator.reset()
    evaluator.addBatch(prediction, groundTruthMapped)
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()
    
    return m_accuracy, m_jaccard, class_jaccard

def evaluateNewLabelPred(modifiedLabel, newPrediction, individual):
    global evaluator

    if individual.IfWeather == True:
        indice = individual.individual["weather"][0].get_indice()
        weather_type = individual.individual["weather"][0].get_weather_type()

        if weather_type == "rain":
            groundTruth, _ = fileIoUtil.openPreLabelFile(modifiedLabel, indice, weather_type)
            prediction, _ = fileIoUtil.openLabelFile(newPrediction)
        elif weather_type == "snow":
            groundTruth, _ = fileIoUtil.openLabelFile(modifiedLabel)
            prediction, _ = fileIoUtil.openPreLabelFile(newPrediction, indice, weather_type)
        elif weather_type == "fog":
            groundTruth, _ = fileIoUtil.openLabelFile(modifiedLabel)
            prediction, _ = fileIoUtil.openPreLabelFile(newPrediction, indice, weather_type) 
        elif weather_type == None:
            groundTruth, _ = fileIoUtil.openLabelFile(modifiedLabel)
            prediction, _ = fileIoUtil.openLabelFile(newPrediction)
    else:
        groundTruth, _ = fileIoUtil.openLabelFile(modifiedLabel)
        prediction, _ = fileIoUtil.openLabelFile(newPrediction)
    # Map to correct classes for evaluation,
    # Example classification "moving-car" -> "car"
    print(groundTruth.shape, prediction.shape)
    groundTruthMapped = remap_lut[groundTruth] # remap to xentropy format
    predictionMapped = remap_lut[prediction] # remap to xentropy format

    # Prepare evaluator
    evaluator.reset()
    evaluator.addBatch(predictionMapped, groundTruthMapped)
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    return m_accuracy, m_jaccard, class_jaccard


# --------------------------------------------------------------------------

"""
Evaluates a given set of mutations

Note 
all bins must be in:
staging
all labels must be in:
done/labels
all modified predictions must be in:
done/modifiedPred/model

"""
# def evalBatch(details, sessionManager,  complete, total):

#     print("\n\nBegin Evaluation:")

#     # ------------------------------------
#     # Get the new predictions

#     # Move the bins that are in this collection of details to the current velodyne folder to run the models on them
#     print("Move bins to evaluate to the current folder")
#     for detail in details:
#         shutil.move(sessionManager.stageDir + "/" + detail["_id"] + ".bin", sessionManager.currentVelDir + "/" + detail["_id"] + ".bin")

#     # run all models on bin files
#     print("Run models:", sessionManager.models)
#     modelTimes = {}
#     for model in sessionManager.models:
#         print("Running ", model)
#         # Get the Model Runner
#         if model == Models.CYL.value:
#             modelRunner = CylRunner(sessionManager.modelDir)
#         elif model == Models.SPV.value:
#             modelRunner = SpvRunner(sessionManager.modelDir)
#         elif model == Models.SAL.value:
#             modelRunner = SalRunner(sessionManager.modelDir)
#         elif model == Models.SQ3.value:
#             modelRunner = Sq3Runner(sessionManager.modelDir)
#         elif model == Models.POL.value:
#             modelRunner = PolRunner(sessionManager.modelDir)
#         elif model == Models.RAN.value:
#             modelRunner = RanRunner(sessionManager.modelDir)
#         elif model == Models.JS3CGPU.value:
#             modelRunner = JS3CGPURunner(sessionManager.modelDir)
#         elif model == Models.JS3CCPU.value:
#             modelRunner = JS3CCPURunner(sessionManager.modelDir)
#         else:
#             raise ValueError("Model {} not supported!".format(model))

#         # Run the docker image
#         tic = time.perf_counter()
#         modelRunner.run(sessionManager.currentDatasetDir, sessionManager.resultDir + "/" + model)
#         toc = time.perf_counter()
#         timeSecondsModel = toc - tic
#         modelTimes[model] = timeSecondsModel


#     # Move bins to done from the current folder
#     print("Move bins to done")
#     allfiles = os.listdir(sessionManager.currentVelDir + "/")
#     for f in allfiles:
#         shutil.move(sessionManager.currentVelDir + "/" + f, sessionManager.doneVelDir + "/" + f)



    # # Evaluate
    # print("Eval")

    # # Get the labels (modified ground truth)
    # labelFiles = []
    # for detail in details:
    #     labelFiles.append(sessionManager.doneLabelDir + "/" + detail["_id"] + ".label")


    # # Get the modified predictions for the models
    # # Set up modified pred dict
    # modifiedPred = {}
    # for model in sessionManager.models:
    #     modifiedPred[model] = []
    # # Get the modified pred file paths
    # for detail in details:
    #     for model in sessionManager.models:
    #         modifiedPred[model].append(sessionManager.doneMutatedPredDir + "/" + model + "/" + detail["_id"] + ".label")


    # # Get the predictions made for the mutated scans
    # predFiles = {}
    # for model in sessionManager.models:
    #     if model == Models.CYL.value or model == Models.SPV.value or model == Models.RAN.value \
    #        or model == Models.JS3CCPU.value or model == Models.JS3CGPU.value:
    #         predFiles[model] = glob.glob(sessionManager.resultDir + "/" + model + "/*.label")
    #     else:
    #         predFiles[model] = glob.glob(sessionManager.resultDir + "/" + model + "/sequences/00/predictions/*.label")

    #     # Sort the new predictions
    #     predFiles[model] = sorted(predFiles[model])


    # # Sort the labels, modified predictions
    # labelFiles = sorted(labelFiles)
    # for model in sessionManager.models:
    #     modifiedPred[model] = sorted(modifiedPred[model])
    # details = sorted(details, key=itemgetter('_id'))


    # # Assert that we have predictions and labels for all models
    # totalFiles = len(labelFiles)
    # for model in sessionManager.models:
    #     totalFiles += len(predFiles[model])
    # if (totalFiles / (1 + len(sessionManager.models)) != len(details)):
    #     errorString = "ERROR: preds do not match labels,"
    #     for model in sessionManager.models:
    #         errorString += " {} {} ".format(model, len(predFiles[model]))

    #     raise ValueError(errorString + ", labels {}, details {} at {}/{}".format(len(labelFiles), len(details), complete, total))



    # # Evaluate the labels, modified predictions, and new predictions
    # for index in range(0, len(labelFiles)):
    #     print("{}/{}, {} | {}/{}".format(index, len(labelFiles), details[index]["_id"],  complete, total))

    #     for model in sessionManager.models:
    #         # Get the accuracy differentials
    #         modelResults = evalLabels(labelFiles[index], modifiedPred[model][index], predFiles[model][index])
    #         # Update the details
    #         details[index][model] = modelResults

    #         # Move model prediction to the done folder
    #         shutil.move(predFiles[model][index], sessionManager.donePredDir + "/" + model + "/" + details[index]["_id"] + ".label")


    # return details, modelTimes

def evaluateSingelLabelPred(labelFile, predictionFile):
    global evaluator
    results = {}
    groundTruth, _ = fileIoUtil.openLabelFile(labelFile)
    prediction, _ = fileIoUtil.openLabelFile(predictionFile)
    # Map to correct classes for evaluation,
    # Example classification "moving-car" -> "car"
    groundTruthMapped = remap_lut[groundTruth] # remap to xentropy format
    predictionMapped = prediction # remap to xentropy format

    # Prepare evaluator
    evaluator.reset()
    evaluator.addBatch(predictionMapped, groundTruthMapped)
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            results[name_label_mapping[learning_map_inv[i]]] = jacc

        # Save acc
    results["accuracy"] = m_accuracy
    results["jaccard"] = m_jaccard
    return results

import csv
def append_to_csv(data, file_path):
    """
    将数据追加到指定的CSV文件中。如果文件不存在，则创建文件并写入表头。

    参数:
    data (dict): 要写入CSV文件的数据。
    file_path (str): CSV文件的路径。
    """
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        
        # 如果文件不存在，则写入表头
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)

if __name__ == '__main__':
    # modifiedLabels = np.array(glob.glob("/home/semantic-kitti-api-master/4/" + "*.label", recursive = True))
    # modifiedPredictions = np.array(glob.glob("/home/semantic-kitti-api-master/5/" + "*.label", recursive = True))
    # newPredictions = np.array(glob.glob("/home/semantic-kitti-api-master/3_predict/" + "*.label", recursive = True))
    # result = evalLabels(modifiedLabels[0], modifiedPredictions[0], newPredictions[0])
    # print(type(result['percentLossAcc']))
    models = ["Cylinder3D", "SPVCNN","FRNet","MinkuNet"]
    sequences = ["03", "04", "06"]
    for model in models:
        for sequence in sequences:
            Labels = sorted(glob.glob(os.path.join("/home/semantickitti/sequences/{}/labels".format(sequence), "*.label")))
            Predictions = sorted(glob.glob(os.path.join("/home/semantickitti/sequences/{}/{}/compress_predlabel".format(sequence,model), "*.label"))) 
            for label, predict in zip(Labels,Predictions):
                result = evaluateSingelLabelPred(label, predict)
                append_to_csv(result, '{}_output_no_traffic_sign{}.csv'.format(sequence,model) )
    