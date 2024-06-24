import sys
sys.path.append("..")
from algorithm.instanceGene import InstanceGene
import random
from data.assetRepository import AssetRepository
from data import fileIoUtil
import numpy as np
import open3d as o3d
import service.pcd.pcdRotate as pcdRotate
import service.pcd.pcdAdd as pcdAdd
import service.pcd.pcdCommon as pcdCommon
import service.pcd.pcdWeather as pcdWeather
import service.pcd.pcdDeform as pcdDeform
import service.pcd.pcdIntensity as pcdIntensity
import service.pcd.pcdRemove as pcdRemove
import service.pcd.pcdRotate as pcdRotate
import service.pcd.pcdScale as pcdScale
import service.pcd.pcdSignReplace as pcdSignReplace
import domain.semanticMapping as semanticMapping
import domain.mutationsEnum as mutationsEnum
from  algorithm.weatherGene import WeatherGene
import copy
import time
import domain.config as config
from domain.semanticMapping import instances as instance_map
import pandas as pd
class Individual():
    def __init__(self, IfInstance, IfWeather, max_radius, genenums, instances_range, weather_range, scales, angles, intensitys_range, pk, model, baseLabelPath, baseBinPath, basePredictionPath, baseSequence, baseScene):
        self.max_radius = max_radius
        self.genenums = genenums
        self.instances_range = copy.deepcopy(instances_range)
        self.scales = scales
        self.angles = angles
        self.intensitys_range = copy.deepcopy(intensitys_range)
        self.weather_range = copy.deepcopy(weather_range)
        self.pk = pk
        self.model = model
        self.visual = False
        self.baseLabelPath = baseLabelPath
        self.baseBinPath = baseBinPath
        self.basePredictionPath = basePredictionPath
        self.baseSequence = baseSequence
        self.baseScene = baseScene
        self.pcdArr = None
        self.intensity = None
        self.semantics = None
        self.instances = None
        self.modelPredictsScene = None
        self.fitness = None
        self.compare_result = None
        self.compare_result_instance = None
        self.IfWeather = IfWeather
        self.IfInstance = IfInstance
        self.individual = {}
        self.pcdArr_after_instance = None
        self.intensity_after_instance = None



    """
    visualize
    Uses open3d to visualize the specific mutation

    Colors based on semantics
    Or by intensity if INTENSITY was the mutation
    """
    def visAllInstances(self, pcdArrAssetList, pcdArr, semantics):

        obbs = []

        # Get scene
        pcdScene = o3d.geometry.PointCloud()
        pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

        # Color either with intensity or the semantic label
        colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
        for semIdx in range(0, len(semantics)):
            colors[semIdx][0] = (semanticMapping.color_map_alt_rgb[semantics[semIdx]][0] / 255)
            colors[semIdx][1] = (semanticMapping.color_map_alt_rgb[semantics[semIdx]][1] / 255)
            colors[semIdx][2] = (semanticMapping.color_map_alt_rgb[semantics[semIdx]][2] / 255)
        pcdScene.colors = o3d.utility.Vector3dVector(colors)

            # Create point clouds for each asset and their oriented bounding boxes
        for pcdArrAsset in pcdArrAssetList:
            pcdAsset = o3d.geometry.PointCloud()
            pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAsset)
            obb = pcdAsset.get_oriented_bounding_box()
            # obb.line_width = 5  # Adjust the thickness (in pixels)
            obb.color = (1, 0, 0)  # Adjust the color (RGB)
            # obb.color = (0.7, 0.7, 0.7)
            obbs.append(obb)
        # Visualize all the oriented bounding boxes and the scene
        o3d.visualization.draw_geometries(obbs + [pcdScene])



    def vis_allPcdPoints(self, points, visual):
        if visual:
            pcd_point = o3d.geometry.PointCloud()
            pcd_point.points = o3d.utility.Vector3dVector(points)
            viewer = o3d.visualization.Visualizer()
            viewer.create_window(window_name='可视化', width=1920, height=1080)
            opt = viewer.get_render_option()
            opt.background_color = np.asarray([1, 1, 1])
            opt.point_size = 1
            opt.show_coordinate_frame = True
            pcd_point.paint_uniform_color([0, 0, 0])
            viewer.add_geometry(pcd_point)
            viewer.run()

    def set(self, attr, value):
        setattr(self, attr, value)

    def changeInstancesRange(self, baseSequence):
        instanceRange = []
        instance_dict = config.rule[baseSequence]
        for key, value in instance_dict.items():
            instanceType = instance_map[key]
            instanceRange.append(instanceType)
        return instanceRange

    def initial(self, model, assetRepository,genenums, instances_range_old, visual, baseBinPath, baseLabelPath, baseSequence, baseScene, basePredictionPath):
        instances_range_og = instances_range_old.copy()
        # instances_range_og = self.changeInstancesRange(baseSequence)
        instances_range_og.append("NULL")
        pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBin(baseBinPath, baseLabelPath, baseSequence, baseScene)
        if self.IfInstance:
            individualInstance = []
            modelPredictionsScene = {}
            modelPredictionsScene[model] = fileIoUtil.openModelPredictions(basePredictionPath, model, baseSequence, baseScene)
            details = {}
            pcdAssetList = []
            #Initialize the gene sequence in order from outer ring to inner ring
            for distIndex in range(genenums - 1, -1, -1):
                successNum = 0
                instances_range = copy.deepcopy(instances_range_og)
                instancegene = InstanceGene()
                modelPredictionAsset = {}
                while(successNum < 10):
                    instances_range = copy.deepcopy(instances_range_og)
                    pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = None, None, None, None, None
                    instancetype = random.choice(instances_range)
                    #When there are no instances in the ring
                    if instancetype == "NULL":
                        successNum += 1
                        individualInstance.append(instancegene)
                        break
                    #When there are instances in the ring
                    else:
                        while(True):
                            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepository.getAssetWithinModelDistTypeNum(model, distIndex, instancetype)
                            if assetRecord != None:
                                print(assetRecord)
                                break
                            if not instances_range: 
                                break
                            instances_range.remove(instancetype)
                            if instances_range[0] == "NULL":  
                                break
                            elif instances_range:
                                other_instances = [inst for inst in instances_range if inst != "NULL"]
                                instancetype = random.choice(other_instances)
                            else:
                                break
                        if assetRecord == None:
                            continue
                        details["asset"] = assetRecord["_id"]
                        details["assetSequence"] = assetRecord["sequence"]
                        details["assetScene"] = assetRecord["scene"]
                        details["assetType"] = assetRecord["type"]
                        details["typeNum"] = assetRecord["type-semantic"]
                        details["assetPoints"] = assetRecord["type-points"]
                        if (assetRecord["type-semantic"] == 81):
                            details["assetPoints"] = int(np.shape(pcdArrAsset[semanticsAsset == 81])[0])
                        # _, instanceAssetScene = fileIoUtil.openLabel(baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                        _, instanceAssetScene = fileIoUtil.openInstanceLabel(baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                        modelAssetScene = fileIoUtil.openModelPredictions(basePredictionPath, model, assetRecord["sequence"], assetRecord["scene"])
                        if isinstance(assetRecord["all-instances"], int):
                            assetRecord["all-instances"] = [assetRecord["all-instances"]] 
                        maskOnlyInst = np.zeros_like(instanceAssetScene, dtype=bool)
                        for allInstance in assetRecord["all-instances"]:
                            maskOnlyInst |= (instanceAssetScene == allInstance)
                        modelPredictionAsset[model] = modelAssetScene[maskOnlyInst]
                        success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdRotate.rotate(pcdArr, intensity, semantics, instances,
                                                                                                                pcdArrAsset,
                                                                                                                details, None,
                                                                                                                modelPredictionsScene)
                        # success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdAdd.addInstance(pcdArr, intensity, semantics, instances,
                        #                                                                                        pcdArrAsset, details, None,modelPredictionsScene, baseSequence)
                        pcdArr, intensity, semantics, instances = sceneResult
                        # Combine the final results
                        if success:
                            instancegene.set_asset(assetRecord)
                            instancegene.set_angle(details['rotate'])
                            # print(details['rotate'])
                            pcdArr, intensity, semantics, instances = pcdCommon.combine(pcdArr, intensity, semantics, instances,
                                                                                            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset)
                            # Combine model results
                            modelPredictionsScene[model] = np.hstack((modelPredictionsScene[model], modelPredictionAsset[model]))
                            pcdAssetList.append(pcdArrAsset)
                            individualInstance.append(instancegene)
                            break
                        else:
                            successNum += 1

                if successNum >= 10:
                    individualInstance.append(instancegene)


            self.modelPredictsScene = modelPredictionsScene[model]
            self.individual.update({"instance":individualInstance})
            self.pcdArr_after_instance = pcdArr
            self.intensity_after_instance = intensity
            # Visualize the mutation if enabled
            if len(individualInstance) == genenums and visual:
                self.visAllInstances(pcdAssetList, pcdArr, semantics)

        if self.IfWeather:
            individualWeather = []
            weathertype = random.choice(self.weather_range)
            weathergene = WeatherGene()
            if weathertype == "NULL":
                pass
            elif weathertype == "rain":
                pcdArr, intensity, indice, ran = pcdWeather.augment_rain(pcdArr, intensity, None, None)
                weathergene.set_weather_type(weathertype)
                weathergene.set_weather_para(ran)
                weathergene.set_indice(indice)
            elif weathertype == "snow":
                pcdArr, intensity, add_points, ran = pcdWeather.augment_snow(pcdArr, intensity, None, None)
                weathergene.set_weather_type(weathertype)
                weathergene.set_weather_para(ran)
                weathergene.set_add_points(add_points)
                weathergene.set_indice(len(add_points))
            elif weathertype == "fog":
                pcdArr, intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(pcdArr, intensity, None, None, None)
                weathergene.set_weather_type(weathertype)
                weathergene.set_weather_para(ran)
                weathergene.set_add_points(add_points)
                weathergene.set_add_intensity(add_intensity)
                weathergene.set_indice(len(add_points))
            individualWeather.append(weathergene)
            self.individual.update({"weather":individualWeather})
            print(weathertype)

        # # Combine the xyz, intensity and semantics, instance labels and bins
        # if len(individual) == genenums:
        self.pcdArr = pcdArr
        self.intensity = intensity
        self.semantics = semantics
        self.instances = instances
        self.vis_allPcdPoints(self.pcdArr, self.visual)

    def initial_more_instances(self, model, assetRepository,genenums, instances_range_old, visual, baseBinPath, baseLabelPath, baseSequence, baseScene, basePredictionPath):
        instances_range_og = instances_range_old.copy()
        # instances_range_og = self.changeInstancesRange(baseSequence)
        instances_range_og.append("NULL")
        pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBin(baseBinPath, baseLabelPath, baseSequence, baseScene)
        if self.IfInstance:
            individualInstance = []
            modelPredictionsScene = {}
            modelPredictionsScene[model] = fileIoUtil.openModelPredictions(basePredictionPath, model, baseSequence, baseScene)
            details = {}
            pcdAssetList = []
            #Initialize the gene sequence in order from outer ring to inner ring
            for distIndex in range(genenums - 1, -1, -1):
                successNum = 0
                instances_range = copy.deepcopy(instances_range_og)
                instancegene = InstanceGene()
                modelPredictionAsset = {}
                while(successNum < 10):
                    instances_range = copy.deepcopy(instances_range_og)
                    pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = None, None, None, None, None
                    instancetype = random.choice(instances_range)
                    #When there are no instances in the ring
                    if instancetype == "NULL":
                        successNum += 1
                        individualInstance.append(instancegene)
                        break
                    #When there are instances in the ring
                    else:
                        while(True):
                            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepository.getAssetWithinDistTypeNumRandom(distIndex, instancetype)
                            if assetRecord != None:
                                print(assetRecord)
                                break
                            if not instances_range: 
                                break
                            instances_range.remove(instancetype)
                            if instances_range[0] == "NULL":  
                                break
                            elif instances_range:
                                other_instances = [inst for inst in instances_range if inst != "NULL"]
                                instancetype = random.choice(other_instances)
                            else:
                                break
                        if assetRecord == None:
                            continue
                        details["asset"] = assetRecord["_id"]
                        details["assetSequence"] = assetRecord["sequence"]
                        details["assetScene"] = assetRecord["scene"]
                        details["assetType"] = assetRecord["type"]
                        details["typeNum"] = assetRecord["type-semantic"]
                        details["assetPoints"] = assetRecord["type-points"]
                        if (assetRecord["type-semantic"] == 81):
                            details["assetPoints"] = int(np.shape(pcdArrAsset[semanticsAsset == 81])[0])
                        # _, instanceAssetScene = fileIoUtil.openLabel(baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                        _, instanceAssetScene = fileIoUtil.openInstanceLabel(baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                        modelAssetScene = fileIoUtil.openModelPredictions(basePredictionPath, model, assetRecord["sequence"], assetRecord["scene"])
                        if isinstance(assetRecord["all-instances"], int):
                            assetRecord["all-instances"] = [assetRecord["all-instances"]] 
                        maskOnlyInst = np.zeros_like(instanceAssetScene, dtype=bool)
                        for allInstance in assetRecord["all-instances"]:
                            maskOnlyInst |= (instanceAssetScene == allInstance)
                        modelPredictionAsset[model] = modelAssetScene[maskOnlyInst]
                        # success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdRotate.rotate(pcdArr, intensity, semantics, instances,
                        #                                                                                         pcdArrAsset,
                        #                                                                                         details, None,
                        #                                                                                         modelPredictionsScene)
                        success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdAdd.addInstance_no_rule(pcdArr, intensity, semantics, instances,
                                                                                                               pcdArrAsset, details, None,modelPredictionsScene)
                        pcdArr, intensity, semantics, instances = sceneResult
                        # Combine the final results
                        if success:
                            instancegene.set_asset(assetRecord)
                            instancegene.set_angle(details['rotate'])
                            # print(details['rotate'])
                            pcdArr, intensity, semantics, instances = pcdCommon.combine(pcdArr, intensity, semantics, instances,
                                                                                            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset)
                            # Combine model results
                            modelPredictionsScene[model] = np.hstack((modelPredictionsScene[model], modelPredictionAsset[model]))
                            pcdAssetList.append(pcdArrAsset)
                            individualInstance.append(instancegene)
                            break
                        else:
                            successNum += 1

                if successNum >= 10:
                    individualInstance.append(instancegene)


            self.modelPredictsScene = modelPredictionsScene[model]
            self.individual.update({"instance":individualInstance})
            self.pcdArr_after_instance = pcdArr
            self.intensity_after_instance = intensity
            # Visualize the mutation if enabled
            if len(individualInstance) == genenums and visual:
                self.visAllInstances(pcdAssetList, pcdArr, semantics)

        if self.IfWeather:
            individualWeather = []
            weathertype = random.choice(self.weather_range)
            weathergene = WeatherGene()
            if weathertype == "NULL":
                pass
            elif weathertype == "rain":
                pcdArr, intensity, indice, ran = pcdWeather.augment_rain(pcdArr, intensity, None, None)
                weathergene.set_weather_type(weathertype)
                weathergene.set_weather_para(ran)
                weathergene.set_indice(indice)
            elif weathertype == "snow":
                pcdArr, intensity, add_points, ran = pcdWeather.augment_snow(pcdArr, intensity, None, None)
                weathergene.set_weather_type(weathertype)
                weathergene.set_weather_para(ran)
                weathergene.set_add_points(add_points)
                weathergene.set_indice(len(add_points))
            elif weathertype == "fog":
                pcdArr, intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(pcdArr, intensity, None, None, None)
                weathergene.set_weather_type(weathertype)
                weathergene.set_weather_para(ran)
                weathergene.set_add_points(add_points)
                weathergene.set_add_intensity(add_intensity)
                weathergene.set_indice(len(add_points))
            individualWeather.append(weathergene)
            self.individual.update({"weather":individualWeather})
            print(weathertype)

        # # Combine the xyz, intensity and semantics, instance labels and bins
        # if len(individual) == genenums:
        self.pcdArr = pcdArr
        self.intensity = intensity
        self.semantics = semantics
        self.instances = instances
        self.vis_allPcdPoints(self.pcdArr, self.visual)


    def initial_random(self, model, assetRepository,genenums, instances_range_old, visual, baseBinPath, baseLabelPath, baseSequence, baseScene, basePredictionPath):
        # instances_range_og = self.changeInstancesRange(baseSequence)
        instances_range_og = instances_range_old.copy()
        instances_range_og.append("NULL")
        pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBin(baseBinPath, baseLabelPath, baseSequence, baseScene)
        if self.IfInstance:
            individualInstance = []
            modelPredictionsScene = {}
            modelPredictionsScene[model] = fileIoUtil.openModelPredictions(basePredictionPath, model, baseSequence, baseScene)
            details = {}
            pcdAssetList = []
            #Initialize the gene sequence in order from outer ring to inner ring
            successNum = 0
            instances_range = copy.deepcopy(instances_range_og)
            instancegene = InstanceGene()
            modelPredictionAsset = {}
            while(successNum < 10):
                instances_range = copy.deepcopy(instances_range_og)
                pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = None, None, None, None, None
                instancetype = random.choice(instances_range)
                instancetype = "car"
                #When there are no instances in the ring
                if instancetype == "NULL":
                    successNum += 1
                    individualInstance.append(instancegene)
                    break
                #When there are instances in the ring
                else:
                    while(True):
                        pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepository.getAssetWithinDistTypeNumRandomSequence(2, instancetype,"03")
                        if assetRecord != None:
                            print(assetRecord)
                            break
                        if not instances_range: 
                            break
                        instances_range.remove(instancetype)
                        if instances_range[0] == "NULL":  
                            break
                        elif instances_range:
                            other_instances = [inst for inst in instances_range if inst != "NULL"]
                            instancetype = random.choice(other_instances)
                        else:
                            break
                    if assetRecord == None:
                        continue
                    details["asset"] = assetRecord["_id"]
                    details["assetSequence"] = assetRecord["sequence"]
                    details["assetScene"] = assetRecord["scene"]
                    details["assetType"] = assetRecord["type"]
                    details["typeNum"] = assetRecord["type-semantic"]
                    details["assetPoints"] = assetRecord["type-points"]
                    if (assetRecord["type-semantic"] == 81):
                        details["assetPoints"] = int(np.shape(pcdArrAsset[semanticsAsset == 81])[0])
                    # _, instanceAssetScene = fileIoUtil.openLabel(baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                    _, instanceAssetScene = fileIoUtil.openInstanceLabel(baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                    modelAssetScene = fileIoUtil.openModelPredictions(basePredictionPath, model, assetRecord["sequence"], assetRecord["scene"])
                    if isinstance(assetRecord["all-instances"], int):
                        assetRecord["all-instances"] = [assetRecord["all-instances"]] 
                    maskOnlyInst = np.zeros_like(instanceAssetScene, dtype=bool)
                    for allInstance in assetRecord["all-instances"]:
                        maskOnlyInst |= (instanceAssetScene == allInstance)
                    modelPredictionAsset[model] = modelAssetScene[maskOnlyInst]
                    # success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdAdd.addInstance_no_render(pcdArr, intensity, semantics, instances,
                    #                                                                                         pcdArrAsset,
                    #                                                                                         details, None,
                    #                                                                                         modelPredictionsScene, baseSequence)
                    success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdRotate.rotate(pcdArr, intensity, semantics, instances,
                                                                                                            pcdArrAsset,
                                                                                                            details, None,
                                                                                                            modelPredictionsScene)
                    pcdArr, intensity, semantics, instances = sceneResult
                    # Combine the final results
                    if success:
                        instancegene.set_asset(assetRecord)
                        instancegene.set_angle(details['rotate'])
                        pcdArr, intensity, semantics, instances = pcdCommon.combine(pcdArr, intensity, semantics, instances,
                                                                                        pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset)
                        # Combine model results
                        modelPredictionsScene[model] = np.hstack((modelPredictionsScene[model], modelPredictionAsset[model]))
                        pcdAssetList.append(pcdArrAsset)
                        individualInstance.append(instancegene)
                        break
                    else:
                        successNum += 1

            if successNum >= 10:
                individualInstance.append(instancegene)


            self.modelPredictsScene = modelPredictionsScene[model]
            self.individual.update({"instance":individualInstance})
            self.pcdArr_after_instance = pcdArr
            self.intensity_after_instance = intensity
            # Visualize the mutation if enabled
            if visual:
                self.visAllInstances(pcdAssetList, pcdArr, semantics)

        # if self.IfWeather:
        #     individualWeather = []
        #     weathertype = random.choice(self.weather_range)
        #     weathergene = WeatherGene()
        #     if weathertype == "NULL":
        #         pass
        #     elif weathertype == "rain":
        #         pcdArr, intensity, indice, ran = pcdWeather.augment_rain(pcdArr, intensity, None, None)
        #         weathergene.set_weather_type(weathertype)
        #         weathergene.set_weather_para(ran)
        #         weathergene.set_indice(indice)
        #     elif weathertype == "snow":
        #         pcdArr, intensity, add_points, ran = pcdWeather.augment_snow(pcdArr, intensity, None, None)
        #         weathergene.set_weather_type(weathertype)
        #         weathergene.set_weather_para(ran)
        #         weathergene.set_add_points(add_points)
        #         weathergene.set_indice(len(add_points))
        #     elif weathertype == "fog":
        #         pcdArr, intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(pcdArr, intensity, None, None, None)
        #         weathergene.set_weather_type(weathertype)
        #         weathergene.set_weather_para(ran)
        #         weathergene.set_add_points(add_points)
        #         weathergene.set_add_intensity(add_intensity)
        #         weathergene.set_indice(len(add_points))
        #     individualWeather.append(weathergene)
        #     self.individual.update({"weather":individualWeather})
        #     print(weathertype)

        # # Combine the xyz, intensity and semantics, instance labels and bins
        # if len(individual) == genenums:
        self.pcdArr = pcdArr
        self.intensity = intensity
        self.semantics = semantics
        self.instances = instances
        # self.vis_allPcdPoints(self.pcdArr, self.visual)
    
    def initial_from_record(self, genenums, recordpath, assetRepository, row_index):
        df = pd.read_csv(recordpath)
        
        new_column_name = self.model
        
        if new_column_name not in df.columns:
            df[new_column_name] = None  

        row = df.iloc[row_index]
        print(row)
        instance_gene_list = []
        for i in range(0, genenums * 2, 2):
            asset_id = row[i]
            angle = row[i + 1]
            gene = InstanceGene()
            if not pd.isna(asset_id):
                gene.set_asset(asset_id)
                gene.set_angle(angle)
            instance_gene_list.append(gene)

        self.individual.update({"instance": instance_gene_list})
        flag = self.crossover_generate_pcd_instance_from_record(assetRepository)

        if not flag:
            df.at[row_index, new_column_name] = self.model
            df.to_csv(recordpath, index=False)

        return flag
    def initial_by_id(self, model, assetRepository,genenums, instances_range_old, visual, baseBinPath, baseLabelPath, baseSequence, baseScene, basePredictionPath):
        instances_range_og = ["car", "person", "bicyclist","motorcycle"]
        rotate_number = None
        pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBin(baseBinPath, baseLabelPath, baseSequence, baseScene)
        if self.IfInstance:
            individualInstance = []
            modelPredictionsScene = {}
            modelPredictionsScene[model] = fileIoUtil.openModelPredictions(basePredictionPath, model, baseSequence, baseScene)
            details = {}
            pcdAssetList = []
            #Initialize the gene sequence in order from outer ring to inner ring
            for distIndex in range(1):
                successNum = 0
                instancegene = InstanceGene()
                modelPredictionAsset = {}
                while(successNum < 10):
                    pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = None, None, None, None, None
                    instancetype = instances_range_og[distIndex]
                    print(instancetype)
                    #When there are no instances in the ring
                    if instancetype == "NULL":
                        successNum += 1
                        individualInstance.append(instancegene)
                        break
                    #When there are instances in the ring
                    else:
                        while(True):
                            if instancetype ==  "car" or instancetype ==  "bicyclist":
                                ins_number = 3
                            else:
                                ins_number =2
                            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepository.getAssetWithinDistTypeNumRandom(2, instancetype)
                            if assetRecord != None:
                                print(assetRecord)
                                break
                        if assetRecord == None:
                            continue
                        details["asset"] = assetRecord["_id"]
                        details["assetSequence"] = assetRecord["sequence"]
                        details["assetScene"] = assetRecord["scene"]
                        details["assetType"] = assetRecord["type"]
                        details["typeNum"] = assetRecord["type-semantic"]
                        details["assetPoints"] = assetRecord["type-points"]
                        if (assetRecord["type-semantic"] == 81):
                            details["assetPoints"] = int(np.shape(pcdArrAsset[semanticsAsset == 81])[0])
                        # _, instanceAssetScene = fileIoUtil.openLabel(baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                        _, instanceAssetScene = fileIoUtil.openInstanceLabel(baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                        modelAssetScene = fileIoUtil.openModelPredictions(basePredictionPath, model, assetRecord["sequence"], assetRecord["scene"])
                        if isinstance(assetRecord["all-instances"], int):
                            assetRecord["all-instances"] = [assetRecord["all-instances"]] 
                        maskOnlyInst = np.zeros_like(instanceAssetScene, dtype=bool)
                        for allInstance in assetRecord["all-instances"]:
                            maskOnlyInst |= (instanceAssetScene == allInstance)
                        modelPredictionAsset[model] = modelAssetScene[maskOnlyInst]
                        if instancetype =="car":
                            rotate_number = None
                        success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdRotate.rotate(pcdArr, intensity, semantics, instances,
                                                                                                                pcdArrAsset,
                                                                                                                details, rotate_number,
                                                                                                                modelPredictionsScene)
                        # success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdAdd.addInstance(pcdArr, intensity, semantics, instances,
                        #                                                                                         pcdArrAsset, details, None,modelPredictionsScene, baseSequence)
                        pcdArr, intensity, semantics, instances = sceneResult
                        # Combine the final results
                        if success:
                            instancegene.set_asset(assetRecord)
                            instancegene.set_angle(details['rotate'])
                            print(details['rotate'])
                            # print(details['rotate'])
                            pcdArr, intensity, semantics, instances = pcdCommon.combine(pcdArr, intensity, semantics, instances,
                                                                                            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset)
                            # Combine model results
                            modelPredictionsScene[model] = np.hstack((modelPredictionsScene[model], modelPredictionAsset[model]))
                            pcdAssetList.append(pcdArrAsset)
                            individualInstance.append(instancegene)
                            break
                        else:
                            successNum += 1

                if successNum >= 10:
                    individualInstance.append(instancegene)


            self.modelPredictsScene = modelPredictionsScene[model]
            self.individual.update({"instance":individualInstance})
            self.pcdArr_after_instance = pcdArr
            self.intensity_after_instance = intensity
            # Visualize the mutation if enabled
            self.visAllInstances(pcdAssetList, pcdArr, semantics)
        self.pcdArr = pcdArr
        self.intensity = intensity
        self.semantics = semantics
        self.instances = instances

       
    def crossover_generate_pcd_instance(self, assetRepository):
        pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBin(self.baseBinPath, self.baseLabelPath, self.baseSequence, self.baseScene)
        details = {}
        modelPredictionsScene = {}
        modelPredictionsScene[self.model] = fileIoUtil.openModelPredictions(self.basePredictionPath, self.model, self.baseSequence, self.baseScene)
        successNum = 0 
        instanceNum = 0
        for indiInstance in self.individual["instance"]:
            modelPredictionAsset = {}
            if indiInstance.get_asset() == None:
                pass
            else:
                instanceNum += 1
                print(indiInstance.get_asset()["_id"])
                pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepository.getAssetById(indiInstance.get_asset()["_id"])
                details["asset"] = assetRecord["_id"]
                details["assetSequence"] = assetRecord["sequence"]
                details["assetScene"] = assetRecord["scene"]
                details["assetType"] = assetRecord["type"]
                details["typeNum"] = assetRecord["type-semantic"]
                details["assetPoints"] = assetRecord["type-points"]
                if (assetRecord["type-semantic"] == 81):
                    details["assetPoints"] = int(np.shape(pcdArrAsset[semanticsAsset == 81])[0])
                #_, instanceAssetScene = fileIoUtil.openLabel(self.baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                _, instanceAssetScene = fileIoUtil.openInstanceLabel(self.baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                modelAssetScene = fileIoUtil.openModelPredictions(self.basePredictionPath, self.model, assetRecord["sequence"], assetRecord["scene"])
                if isinstance(assetRecord["all-instances"], int):
                    assetRecord["all-instances"] = [assetRecord["all-instances"]] 
                maskOnlyInst = np.zeros_like(instanceAssetScene, dtype=bool)
                for allInstance in assetRecord["all-instances"]:
                    maskOnlyInst |= (instanceAssetScene == allInstance)
                modelPredictionAsset[self.model] = modelAssetScene[maskOnlyInst]
                success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdRotate.rotate(pcdArr, intensity, semantics, instances,
                                                                                                                    pcdArrAsset,
                                                                                                                    details, indiInstance.get_angle(),
                                                                                                                    modelPredictionsScene)
                # success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdAdd.addInstance(pcdArr, intensity, semantics, instances,
                #                  pcdArrAsset, details, None,modelPredictionsScene, self.baseSequence)
                pcdArr, intensity, semantics, instances = sceneResult
                if success == False:
                    return False
                if success:
                    successNum += 1
                    indiInstance.set_angle(details['rotate'])
                    if indiInstance.get_intensity() != None:
                        intensityAsset, details = pcdIntensity.intensityChange(intensityAsset, details, indiInstance.get_intensity(), details["assetType"])
                    pcdArr, intensity, semantics, instances = pcdCommon.combine(pcdArr, intensity, semantics, instances,
                                                                                    pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset)
                    # Combine model results
                    modelPredictionsScene[self.model] = np.hstack((modelPredictionsScene[self.model], modelPredictionAsset[self.model]))

        if successNum != instanceNum:
            return False
        else:
            self.pcdArr_after_instance = pcdArr
            self.intensity_after_instance = intensity
            self.pcdArr = pcdArr
            self.intensity = intensity
            self.semantics = semantics
            self.instances = instances
            self.modelPredictsScene = modelPredictionsScene[self.model]
            self.fitness = None
            self.compare_result = None
            self.compare_result_instance = None
            return True
        

    def crossover_generate_pcd_instance_from_record(self, assetRepository):
        pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBin(self.baseBinPath, self.baseLabelPath, self.baseSequence, self.baseScene)
        details = {}
        modelPredictionsScene = {}
        modelPredictionsScene[self.model] = fileIoUtil.openModelPredictions(self.basePredictionPath, self.model, self.baseSequence, self.baseScene)
        successNum = 0 
        instanceNum = 0
        for indiInstance in self.individual["instance"]:
            modelPredictionAsset = {}
            if indiInstance.get_asset() == None:
                pass
            else:
                instanceNum += 1
                pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepository.getAssetById(indiInstance.get_asset())
                details["asset"] = assetRecord["_id"]
                details["assetSequence"] = assetRecord["sequence"]
                details["assetScene"] = assetRecord["scene"]
                details["assetType"] = assetRecord["type"]
                details["typeNum"] = assetRecord["type-semantic"]
                details["assetPoints"] = assetRecord["type-points"]
                if (assetRecord["type-semantic"] == 81):
                    details["assetPoints"] = int(np.shape(pcdArrAsset[semanticsAsset == 81])[0])
                #_, instanceAssetScene = fileIoUtil.openLabel(self.baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                _, instanceAssetScene = fileIoUtil.openInstanceLabel(self.baseLabelPath, assetRecord["sequence"], assetRecord["scene"])
                modelAssetScene = fileIoUtil.openModelPredictions(self.basePredictionPath, self.model, assetRecord["sequence"], assetRecord["scene"])
                if isinstance(assetRecord["all-instances"], int):
                    assetRecord["all-instances"] = [assetRecord["all-instances"]] 
                maskOnlyInst = np.zeros_like(instanceAssetScene, dtype=bool)
                for allInstance in assetRecord["all-instances"]:
                    maskOnlyInst |= (instanceAssetScene == allInstance)
                modelPredictionAsset[self.model] = modelAssetScene[maskOnlyInst]
                # success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdRotate.rotate(pcdArr, intensity, semantics, instances,
                #                                                                                                     pcdArrAsset,
                #                                                                                                     details, indiInstance.get_angle(),
                #                                                                                                     modelPredictionsScene)
                success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdAdd.addInstance_no_rule(pcdArr, intensity, semantics, instances,
                                                                                                        pcdArrAsset, details, None,modelPredictionsScene)
                pcdArr, intensity, semantics, instances = sceneResult
                if success == False:
                    return False
                if success:
                    successNum += 1
                    indiInstance.set_angle(details['rotate'])
                    if indiInstance.get_intensity() != None:
                        intensityAsset, details = pcdIntensity.intensityChange(intensityAsset, details, indiInstance.get_intensity(), details["assetType"])
                    pcdArr, intensity, semantics, instances = pcdCommon.combine(pcdArr, intensity, semantics, instances,
                                                                                    pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset)
                    # Combine model results
                    modelPredictionsScene[self.model] = np.hstack((modelPredictionsScene[self.model], modelPredictionAsset[self.model]))

        if successNum != instanceNum:
            return False
        else:
            self.pcdArr_after_instance = pcdArr
            self.intensity_after_instance = intensity
            self.pcdArr = pcdArr
            self.intensity = intensity
            self.semantics = semantics
            self.instances = instances
            self.modelPredictsScene = modelPredictionsScene[self.model]
            self.fitness = None
            self.compare_result = None
            self.compare_result_instance = None
            return True
        
    def crossover_generate_pcd_weather(self):
        if self.IfInstance == True:
            pcdArr, intensity = copy.deepcopy(self.pcdArr), copy.deepcopy(self.intensity)
        else:
            pcdArr, intensity, _, _ = fileIoUtil.openLabelBin(self.baseBinPath, self.baseLabelPath, self.baseSequence, self.baseScene)

        weathergene = self.individual["weather"][0]
        weathertype = weathergene.get_weather_type()
        weatherpara = weathergene.get_weather_para()
        weather_indice = weathergene.get_indice()
        add_points = weathergene.get_add_points()
        add_intensity = weathergene.get_add_intensity()

        if weathertype == "rain":
            # print(pcdArr.shape, intensity.shape, weather_indice.shape)
            pcdArr, intensity, indice, ran = pcdWeather.augment_rain(pcdArr, intensity, weather_indice, weatherpara)
            self.individual["weather"][0].set_indice(indice)

        elif weathertype == "snow":
            pcdArr, intensity, add_points, ran = pcdWeather.augment_snow(pcdArr, intensity, add_points, weatherpara)

        elif weathertype == "fog":
            pcdArr, intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(pcdArr, intensity, add_points, add_intensity, weatherpara)

        elif weathertype == None:
            pass
        self.pcdArr = pcdArr
        self.intensity = intensity


    def mutation_generate_pcd_instanceandweather(self, index, assetRepository):
        random.seed(time.time())
        if index < self.genenums:
            #change asset type
            mutatetype = random.randint(0, 2)
            details = {}
            if mutatetype == 0:
                instancegeneoriginal = copy.deepcopy(self.individual["instance"][index])
                number = 0
                flag = True
                while True and number < 20:
                    instances_range = self.changeInstancesRange(self.baseSequence)
                    instances_range.append("NULL")
                    instancetype = random.choice(instances_range)
                    if instancetype == "NULL":
                        self.individual["instance"][index] = InstanceGene()
                    else:
                        pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = None, None, None, None, None
                        while True:
                            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepository.getAssetWithinModelDistTypeNum(self.model, index, instancetype)
                            if assetRecord is not None:
                                break
                            if not instances_range: 
                                break
                            instances_range.remove(instancetype)
                            if instances_range[0] == "NULL":  
                                break
                            elif instances_range:
                                other_instances = [inst for inst in instances_range if inst != "NULL"]
                                instancetype = random.choice(other_instances)
                            else:
                                break
                        if assetRecord == None:
                            continue
                        instancegene = InstanceGene()
                        instancegene.set_asset(assetRecord)
                        self.individual["instance"][index] = instancegene
                    flag = self.crossover_generate_pcd_instance(assetRepository)
                    number += 1
                    if flag:
                        break

                if number >= 20:
                    self.individual["instance"][index] = instancegeneoriginal
                    return False
                else:
                    weathergene = self.individual["weather"][0]
                    weathertype = weathergene.get_weather_type()
                    weatherpara = weathergene.get_weather_para()
                    weather_indice = weathergene.get_indice()
                    add_points = weathergene.get_add_points()
                    add_intensity = weathergene.get_add_intensity()
                    if weathertype == "rain":
                        # print(pcdArr.shape, intensity.shape, weather_indice.shape)
                        self.pcdArr, self.intensity, indice, ran = pcdWeather.augment_rain(self.pcdArr, self.intensity, weather_indice, weatherpara)
                        self.individual["weather"][0].set_indice(indice)

                    elif weathertype == "snow":
                        self.pcdArr, self.intensity, add_points, ran = pcdWeather.augment_snow(self.pcdArr, self.intensity, add_points, weatherpara)

                    elif weathertype == "fog":
                        self.pcdArr, self.intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(self.pcdArr, self.intensity, add_points, add_intensity, weatherpara)

                    elif weathertype == None:
                        pass
                    return True
            #change asset intensity
            elif mutatetype == 1:
                random_intensity = random.uniform(self.intensitys_range[0], self.intensitys_range[1])
                asset = self.individual["instance"][index].get_asset()
                if asset == None:
                    return True
                self.individual["instance"][index].set_intensity(random_intensity)
                pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepository.getAssetById(asset["_id"])
                intensityAsset, details = pcdIntensity.intensityChange(intensityAsset, details, random_intensity, asset["type"])
                self.intensity = pcdCommon.changeIntensity(pcdArrAsset, self.pcdArr, intensityAsset, self.intensity)
                self.intensity_after_instance = pcdCommon.changeIntensity(pcdArrAsset, self.pcdArr_after_instance, intensityAsset, self.intensity_after_instance)
                return True
            #change asset angle
            elif mutatetype == 2:
                number = 0
                angle = self.individual["instance"][index].get_angle()
                while True and number < 20:
                    self.individual["instance"][index].set_angle(None)
                    flag = self.crossover_generate_pcd_instance(assetRepository)
                    number += 1
                    if flag:
                        new_angle = self.individual["instance"][index].get_angle()
                        if new_angle == angle: continue
                        else: break

                if number >= 20:
                    self.individual["instance"][index].set_angle(angle)
                    return False
                else:
                    weathergene = self.individual["weather"][0]
                    weathertype = weathergene.get_weather_type()
                    weatherpara = weathergene.get_weather_para()
                    weather_indice = weathergene.get_indice()
                    add_points = weathergene.get_add_points()
                    add_intensity = weathergene.get_add_intensity()

                    if weathertype == "rain":
                        # print(pcdArr.shape, intensity.shape, weather_indice.shape)
                        self.pcdArr, self.intensity, indice, ran = pcdWeather.augment_rain(self.pcdArr, self.intensity, weather_indice, weatherpara)
                        self.individual["weather"][0].set_indice(indice)

                    elif weathertype == "snow":
                        self.pcdArr, self.intensity, add_points, ran = pcdWeather.augment_snow(self.pcdArr, self.intensity, add_points, weatherpara)

                    elif weathertype == "fog":
                        self.pcdArr, self.intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(self.pcdArr, self.intensity, add_points, add_intensity, weatherpara)

                    elif weathertype == None:
                        pass
                    return True
        else:
            mutatetype = random.randint(0, 1)
            #change weather type
            if mutatetype == 0:
                weather_type = self.individual["weather"][0].get_weather_type()
                weather_range = [x for x in self.weather_range if x != weather_type]
                new_weathertype = random.choice(weather_range)
                weathergene = WeatherGene()
                if new_weathertype == "NULL":
                    self.pcdArr, self.intensity = self.pcdArr_after_instance, self.intensity_after_instance
                    self.individual["weather"][0] = weathergene
                elif new_weathertype == "rain":
                    self.pcdArr, self.intensity, indice, ran = pcdWeather.augment_rain(self.pcdArr_after_instance.copy(), self.intensity_after_instance.copy(), None, None)
                    weathergene.set_weather_type(new_weathertype)
                    weathergene.set_weather_para(ran)
                    weathergene.set_indice(indice)
                elif new_weathertype == "snow":
                    self.pcdArr, self.intensity, add_points, ran = pcdWeather.augment_snow(self.pcdArr_after_instance.copy(), self.intensity_after_instance.copy(), None, None)
                    weathergene.set_weather_type(new_weathertype)
                    weathergene.set_weather_para(ran)
                    weathergene.set_add_points(add_points)
                    weathergene.set_indice(len(add_points))
                elif new_weathertype == "fog":
                    self.pcdArr, self.intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(self.pcdArr_after_instance.copy(), self.intensity_after_instance.copy(), None, None, None)
                    weathergene.set_weather_type(new_weathertype)
                    weathergene.set_weather_para(ran)
                    weathergene.set_add_points(add_points)
                    weathergene.set_add_intensity(add_intensity)
                    weathergene.set_indice(len(add_points))
                self.individual["weather"][0] = weathergene
            #change weather intensity
            elif mutatetype == 1:
                weathergene = self.individual["weather"][0]
                weathertype = weathergene.get_weather_type()
                if weathertype == "NULL":
                    self.pcdArr, self.intensity = self.pcdArr_after_instance, self.intensity_after_instance
                    self.individual["weather"][0] = WeatherGene()
                elif weathertype == "rain":
                    self.pcdArr, self.intensity, indice, ran = pcdWeather.augment_rain(self.pcdArr_after_instance.copy(), self.intensity_after_instance.copy(), None, None)
                    self.individual["weather"][0].set_weather_para(ran)
                    self.individual["weather"][0].set_indice(indice)
                    self.individual["weather"][0].set_add_points = None
                    self.individual["weather"][0].set_add_intensity = None
                elif weathertype == "snow":
                    self.pcdArr, self.intensity, add_points, ran = pcdWeather.augment_snow(self.pcdArr_after_instance.copy(), self.intensity_after_instance.copy(), None, None)
                    self.individual["weather"][0].set_weather_para(ran)
                    self.individual["weather"][0].set_add_points(add_points)
                    self.individual["weather"][0].set_indice(len(add_points))
                    self.individual["weather"][0].set_add_intensity = None
                elif weathertype == "fog":
                    self.pcdArr, self.intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(self.pcdArr_after_instance.copy(), self.intensity_after_instance.copy(), None, None, None)
                    self.individual["weather"][0].set_weather_para(ran)
                    self.individual["weather"][0].set_add_points(add_points)
                    self.individual["weather"][0].set_add_intensity(add_intensity)
                    self.individual["weather"][0].set_indice(len(add_points))
            return True
            # mutatetype = random.randint(0, 1)
            # mutatetype = 0
            # instance_pcd_flag = self.crossover_generate_pcd_instance(assetRepository)
            # if instance_pcd_flag == False:
            #     return False
            # #change weather type
            # if mutatetype == 0:
            #     weather_type = self.individual["weather"][0].get_weather_type()
            #     weather_range = [x for x in self.weather_range if x != weather_type]
            #     new_weathertype = random.choice(weather_range)
            #     weathergene = WeatherGene()
            #     if new_weathertype == "NULL":
            #         self.individual["weather"][0] = weathergene
            #     elif new_weathertype == "rain":
            #         self.pcdArr, self.intensity, indice, ran = pcdWeather.augment_rain(self.pcdArr, self.intensity, None, None)
            #         weathergene.set_weather_type(new_weathertype)
            #         weathergene.set_weather_para(ran)
            #         weathergene.set_indice(indice)
            #     elif new_weathertype == "snow":
            #         self.pcdArr, self.intensity, add_points, ran = pcdWeather.augment_snow(self.pcdArr, self.intensity, None, None)
            #         weathergene.set_weather_type(new_weathertype)
            #         weathergene.set_weather_para(ran)
            #         weathergene.set_add_points(add_points)
            #         weathergene.set_indice(len(add_points))
            #     elif new_weathertype == "fog":
            #         self.pcdArr, self.intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(self.pcdArr, self.intensity, None, None, None)
            #         weathergene.set_weather_type(new_weathertype)
            #         weathergene.set_weather_para(ran)
            #         weathergene.set_add_points(add_points)
            #         weathergene.set_add_intensity(add_intensity)
            #         weathergene.set_indice(len(add_points))
            #     self.individual["weather"][0] = weathergene
            # #change weather intensity
            # elif mutatetype == 1:
            #     weathergene = self.individual["weather"][0]
            #     weathertype = weathergene.get_weather_type()
            #     if weathertype == "NULL":
            #         self.individual["weather"][0] = WeatherGene()
            #     elif weathertype == "rain":
            #         self.pcdArr, self.intensity, indice, ran = pcdWeather.augment_rain(self.pcdArr, self.intensity, None, None)
            #         self.individual["weather"][0].set_weather_para(ran)
            #         self.individual["weather"][0].set_indice(indice)
            #         self.individual["weather"][0].set_add_points = None
            #         self.individual["weather"][0].set_add_intensity = None
            #     elif weathertype == "snow":
            #         self.pcdArr, self.intensity, add_points, ran = pcdWeather.augment_snow(self.pcdArr, self.intensity, None, None)
            #         self.individual["weather"][0].set_weather_para(ran)
            #         self.individual["weather"][0].set_add_points(add_points)
            #         self.individual["weather"][0].set_indice(len(add_points))
            #         self.individual["weather"][0].set_add_intensity = None
            #     elif weathertype == "fog":
            #         self.pcdArr, self.intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(self.pcdArr, self.intensity, None, None, None)
            #         self.individual["weather"][0].set_weather_para(ran)
            #         self.individual["weather"][0].set_add_points(add_points)
            #         self.individual["weather"][0].set_add_intensity(add_intensity)
            #         self.individual["weather"][0].set_indice(len(add_points))
            # return True
    def mutation_generate_pcd_instance(self, index, assetRepository):
            random.seed(time.time())
            if index < self.genenums:
                #change asset type
                mutatetype = random.randint(0, 2)
                details = {}
                if mutatetype == 0:
                    instancegeneoriginal = copy.deepcopy(self.individual["instance"][index])
                    number = 0
                    flag = True
                    while True and number < 20:
                        # instances_range = copy.deepcopy(self.instances_range)
                        instances_range = self.changeInstancesRange(self.baseSequence)
                        instances_range.append("NULL")
                        instancetype = random.choice(instances_range)
                        if instancetype == "NULL":
                            self.individual["instance"][index] = InstanceGene()
                        else:
                            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = None, None, None, None, None
                            while True:
                                pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepository.getAssetWithinModelDistTypeNum(self.model, index, instancetype)
                                if assetRecord is not None:
                                    break
                                if not instances_range: 
                                    break
                                instances_range.remove(instancetype)
                                if instances_range[0] == "NULL":  
                                    break
                                elif instances_range:
                                    other_instances = [inst for inst in instances_range if inst != "NULL"]
                                    instancetype = random.choice(other_instances)
                                else:
                                    break
                            if assetRecord == None:
                                continue
                            instancegene = InstanceGene()
                            instancegene.set_asset(assetRecord)
                            self.individual["instance"][index] = instancegene
                        flag = self.crossover_generate_pcd_instance(assetRepository)
                        number += 1
                        if flag:
                            break

                    if number >= 20:
                        self.individual["instance"][index] = instancegeneoriginal
                        return False
                    else:
                        return True
                #change asset intensity
                elif mutatetype == 1:
                    random_intensity = random.uniform(self.intensitys_range[0], self.intensitys_range[1])
                    asset = self.individual["instance"][index].get_asset()
                    if asset == None:
                        return True
                    self.individual["instance"][index].set_intensity(random_intensity)
                    pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepository.getAssetById(asset["_id"])
                    intensityAsset, details = pcdIntensity.intensityChange(intensityAsset, details, random_intensity, asset["type"])
                    self.intensity = pcdCommon.changeIntensity(pcdArrAsset, self.pcdArr, intensityAsset, self.intensity)
                    return True
                #change asset angle
                elif mutatetype == 2:
                    number = 0
                    angle = self.individual["instance"][index].get_angle()
                    while True and number < 20:
                        self.individual["instance"][index].set_angle(None)
                        flag = self.crossover_generate_pcd_instance(assetRepository)
                        number += 1
                        if flag:
                            new_angle = self.individual["instance"][index].get_angle()
                            if new_angle == angle: continue
                            else: break

                    if number >= 20:
                        self.individual["instance"][index].set_angle(angle)
                        return False
                    else:
                        return True
                    
                    
    def mutation_generate_pcd_weather(self, mutatetype):
        random.seed(time.time())
        if self.IfInstance == True:
            pcdArr, intensity = copy.deepcopy(self.pcdArr), copy.deepcopy(self.intensity)
        else:
            pcdArr, intensity, _, _ = fileIoUtil.openLabelBin(self.baseBinPath, self.baseLabelPath, self.baseSequence, self.baseScene)
        #change weather type
        if mutatetype == 0:
            weather_type = self.individual["weather"][0].get_weather_type()
            weather_range = [x for x in self.weather_range if x != weather_type]
            new_weathertype = random.choice(weather_range)
            if new_weathertype == "NULL":
                self.individual["weather"][0] = WeatherGene()
            elif new_weathertype == "rain":
                pcdArr, intensity, indice, ran = pcdWeather.augment_rain(pcdArr, intensity, None, None)
                self.individual["weather"][0].set_weather_type(new_weathertype)
                self.individual["weather"][0].set_weather_para(ran)
                self.individual["weather"][0].set_indice(indice)
                self.individual["weather"][0].set_add_points = None
                self.individual["weather"][0].set_add_intensity = None
            elif new_weathertype == "snow":
                pcdArr, intensity, add_points, ran = pcdWeather.augment_snow(pcdArr, intensity, None, None)
                self.individual["weather"][0].set_weather_type(new_weathertype)
                self.individual["weather"][0].set_weather_para(ran)
                self.individual["weather"][0].set_add_points(add_points)
                self.individual["weather"][0].set_add_intensity = None
                self.individual["weather"][0].set_indice(len(add_points))
            elif new_weathertype == "fog":
                pcdArr, intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(pcdArr, intensity, None, None, None)
                self.individual["weather"][0].set_weather_type(new_weathertype)
                self.individual["weather"][0].set_weather_para(ran)
                self.individual["weather"][0].set_add_points(add_points)
                self.individual["weather"][0].set_add_intensity(add_intensity)
                self.individual["weather"][0].set_indice(len(add_points))

            self.pcdArr = pcdArr
            self.intensity = intensity

        #change weather intensity
        elif mutatetype == 1:
            weathergene = self.individual["weather"][0]
            weathertype = weathergene.get_weather_type()
            if weathertype == "NULL":
                self.individual["weather"][0] = WeatherGene()
            elif weathertype == "rain":
                pcdArr, intensity, indice, ran = pcdWeather.augment_rain(pcdArr, intensity, None, None)

                self.individual["weather"][0].set_weather_para(ran)
                self.individual["weather"][0].set_indice(indice)
                self.individual["weather"][0].set_add_points = None
                self.individual["weather"][0].set_add_intensity = None
            elif weathertype == "snow":
                pcdArr, intensity, add_points, ran = pcdWeather.augment_snow(pcdArr, intensity, None, None)
                self.individual["weather"][0].set_weather_para(ran)
                self.individual["weather"][0].set_add_points(add_points)
                self.individual["weather"][0].set_indice(len(add_points))
                self.individual["weather"][0].set_add_intensity = None
            elif weathertype == "fog":

                pcdArr, intensity, add_points, add_intensity, ran = pcdWeather.augment_fog(pcdArr, intensity, None, None, None)
                self.individual["weather"][0].set_weather_para(ran)
                self.individual["weather"][0].set_add_points(add_points)
                self.individual["weather"][0].set_add_intensity(add_intensity)
                self.individual["weather"][0].set_indice(len(add_points))
                
            self.pcdArr = pcdArr
            self.intensity = intensity


if __name__ == '__main__':
    assetRepository = AssetRepository("/home/LiDFuzzer/tool/selected_data/pcs/sequences","/home/LiDFuzzer/tool/selected_data/labels/sequences","/home/LiDFuzzer/tool/mongoconnect.txt", "instanceAssetsMinkuNet")
    assetRepository.removeinstance("04")
    # assetRepository = AssetRepository("/home/LiDFuzzer/tool/selected_data/semantic_kitti_pcs/dataset/sequences","/home/LiDFuzzer/tool/selected_data/semantic_kitti_labels/dataset/sequences","    /home/LiDFuzzer/mongodb.txt")
    # max_radius = None
    # genenums = 5 
    # instances_list = ["car", "traffic-sign", "NULL"]
    # weather_range = ["fog"] 
    # scales = None 
    # angles = None 
    # intensitys = None 
    # pk = False
    # model = 'Cylinder3D'
    # baseLabelPath = "/home/LiDFuzzer/tool/selected_data/labels/sequences"
    # baseBinPath = "/home/LiDFuzzer/tool/selected_data/pcs/sequences"
    # basePredictionPath = "/home/LiDFuzzer/tool/pred_data"
    # baseSequence = "04"
    # baseScene = "000000"
    # IfInstance = True
    # IfWeather = False
    # ind = Individual(IfInstance, IfWeather, max_radius, genenums, instances_list, weather_range, scales, angles, intensitys, pk, model, assetRepository, baseLabelPath, baseBinPath, basePredictionPath, baseSequence, baseScene)
    # ind.initial(ind.model, ind.assetRepository, ind.pk, ind.genenums, ind.instances_range, ind.visual, ind.baseBinPath, ind.baseLabelPath, ind.baseSequence, ind.baseScene, ind.basePredictionPath)