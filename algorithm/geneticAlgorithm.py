import yaml
import os
import sys
import re
sys.path.append("..")
from data.fileIO import getBins,getLabels
from data.assetRepository import AssetRepository
from algorithm.individual import Individual
from data import fileIoUtil, fileIO
from InstanceFilter import runModel
from service.eval.eval import evalLabels
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
import numpy as np
import copy
import random
import time
import shutil
from service.models.mmdetection3d import run_inference

class Genetic():
    def __init__(self, args):
        self.mongoConnect = args.mdb
        self.pk = args.pk
            # open config file
        try:
            print("Opening config file %s" % args.config)
            self.cfg = yaml.safe_load(open(args.config, 'r'))
        except Exception as e:
            print(e)
            print("Error opening yaml file.")
            quit()
        # Required paths
        #e.x. /home/semanticKITTI/dataset/sequences/02/velodyne/
        self.binPath = args.binPath
        # self.binPath = os.path.normpath(args.binPath)
        #e.x. /home/semanticKITTI/dataset/sequences/02/labels/
        self.labelPath = args.labelPath
        # self.labelPath = os.path.normpath(args.labelPath)
        #e.x. /home/semanticKITTI/dataset/sequences/02/pre_labels/
        self.basePredictionPath = args.predPath
        # self.basePredictionPath = os.path.normpath(args.predPath)
        #e.x. /home/semanticKITTI/dataset/sequences/02/modifiedPre_labels/
        self.modifidePrediction = args.modifidePrediction
        # self.modifidePrediction = os.path.normpath(args.modifidePrediction)
        self.modelDir = os.path.normpath(args.modelDir)
        self.model = args.models
        self.baseSequence = args.baseSequence
        self.IfInstance = args.IfInstance
        # self.IfWeather = args.IfWeather
        self.IfWeather = False

        print("\nProvided Paths:")
        print("MongoConnect: {}".format(self.mongoConnect))
        print("ModelDirectory: {}".format(self.modelDir))
        print("Scan Bin Path: {}".format(self.binPath))
        print("Label Path: {}".format(self.labelPath))
        print("Original Prediction Path: {}".format(self.basePredictionPath))
        print("Modified Prediction Path: {}".format(self.basePredictionPath))

        self.binFiles = getBins(self.binPath + "/")
        self.labelFiles = getLabels(self.labelPath+ "/")
        self.predictFiles = getLabels(self.basePredictionPath+ "/")
        self.termination = self.cfg["GAconfig"]["termination"]
        self.individualnums = self.cfg["GAconfig"]["population_numbers"]
        self.gene_numbers = self.cfg["GAconfig"]["gene_numbers"]
        self.instances_range = self.cfg["GAconfig"]["instances"]
        self.weather_range = self.cfg["GAconfig"]["weathers"]
        self.scale = self.cfg["GAconfig"]["scale"]
        self.crossover_para = self.cfg["GAconfig"]["crossover"]
        self.mutation_para = self.cfg["GAconfig"]["mutation"]
        self.angle = self.cfg["GAconfig"]["angle"]
        self.intensity_range = self.cfg["GAconfig"]["intensitys"]
        self.max_radius = self.cfg["GAconfig"]["max_radius"]
        self.baseBinPath = os.path.dirname(os.path.dirname(self.binPath)) + "/"
        self.baseLabelPath = os.path.dirname(os.path.dirname(self.labelPath)) + "/"
        # self.assetRepository = AssetRepository(self.baseBinPath, self.baseLabelPath, self.mongoConnect, "instance_assets")
        # self.assetAllInstances = AssetRepository(self.baseBinPath, self.baseLabelPath, self.mongoConnect, "sceneallinstance")
        self.assetRepository = AssetRepository(self.baseBinPath, self.baseLabelPath, self.mongoConnect, "instanceAssets"+ self.model)
        self.assetAllInstances = AssetRepository(self.baseBinPath, self.baseLabelPath, self.mongoConnect, "sceneallinstance")
        # self.populations = []

    def run(self):
        # for index in range(len(self.labelFiles)): 
        for index in range(len(self.labelFiles)):
            lablefile = self.labelFiles[index]
            print(lablefile)
            # last_accuracy,last_jaccard,now_accuracy, now_jaccard = 0,0,0,0
            # fail_num = 0
            result = re.search(r'/(\d+)\.label$', lablefile)
            baseScene = result.group(1)
            population = self.initPopulations(baseScene)
            for generation in range(self.termination):
                if generation == 0:
                    self.saveMutatedFiles(self.binPath, self.labelPath, self.modifidePrediction, population, baseScene, generation)
                    self.evaluate(self.binPath, self.basePredictionPath, self.labelPath, self.modifidePrediction, population, self.baseSequence, baseScene, generation)
                    self.record(population, baseScene, generation)
                    population = self.rankandcrowdingsurvival(population, self.individualnums)
                else:
                    child_population = []
                    crossovernum = 0
                    while len(child_population) < self.individualnums:
                        parent1 = self.tournament_selection(population, 3)
                        parent2 = self.tournament_selection(population, 3)
                        child1, child2, crossoverflag1,  crossoverflag2= self.crossover(parent1, parent2, self.crossover_para)

                        # if crossoverflag1:
                        #     mutateflag1 = self.mutate(child1, self.mutation_para)
                        #     if len(child_population) < self.individualnums and mutateflag1:  # check any space could to insert individual
                        #             child_population.append(child1)
                        #             crossovernum = 0
                        #     if len(child_population) == self.individualnums:
                        #         break

                        # if crossoverflag2:
                        #     mutateflag2 =self.mutate(child2, self.mutation_para)
                        #     if len(child_population) < self.individualnums and mutateflag2:  # double-check any space could to insert individual
                        #             child_population.append(child2)
                        #             crossovernum = 0
                        #     if len(child_population) == self.individualnums:
                        #         break
                        if crossoverflag1:
                            mutate_success = False  
                            attempts = 0  
                            
                            while attempts < 3 and not mutate_success:
                                mutateflag1 = self.mutate(child1, self.mutation_para)
                                if mutateflag1:
                                    mutate_success = True  
                                attempts += 1  
                            
                            if mutate_success and len(child_population) < self.individualnums:
                                child_population.append(child1)
                                crossovernum = 0  

                            if len(child_population) == self.individualnums:
                                break

                        if crossoverflag2:
                            mutate_success = False  
                            attempts = 0  
                            
                            while attempts < 3 and not mutate_success:
                                mutateflag1 = self.mutate(child1, self.mutation_para)
                                if mutateflag1:
                                    mutate_success = True  
                                attempts += 1  
                            
                            if mutate_success and len(child_population) < self.individualnums:
                                child_population.append(child1)
                                crossovernum = 0  

                            if len(child_population) == self.individualnums:
                                break

                        if crossoverflag1 == False and crossoverflag2 == False:
                            crossovernum += 1
                            if crossovernum > 8:
                                    if len(child_population) < self.individualnums:
                                        child_population.append(child1)
                                    if len(child_population) < self.individualnums:
                                        child_population.append(child2)
                                    crossovernum = 0
                        
                    self.saveMutatedFiles(self.binPath, self.labelPath, self.modifidePrediction, child_population, baseScene, generation)
                    self.evaluate(self.binPath, self.basePredictionPath, self.labelPath, self.modifidePrediction, child_population, self.baseSequence, baseScene, generation)
                    total_population = population + child_population
                    population = self.rankandcrowdingsurvival(total_population, self.individualnums)
                    self.saveMutatedFiles(self.binPath, self.labelPath, self.modifidePrediction, population, baseScene, generation)
                    self.record(population, baseScene, generation)

    def random_run(self):
        for index in range(1): 
            lablefile = self.labelFiles[index]
            print(lablefile)
            result = re.search(r'/(\d+)\.label$', lablefile)
            baseScene = result.group(1)
            for generation in range(self.termination):
                population = self.initPopulations_random(baseScene)
                self.saveMutatedFiles(self.binPath, self.labelPath, self.modifidePrediction, population, baseScene, generation)
                self.evaluate(self.binPath, self.basePredictionPath, self.labelPath, self.modifidePrediction, population, self.baseSequence, baseScene, generation)
                self.record(population, baseScene, generation)

    def random_no_evaluate(self):
        for index in range(len(self.labelFiles)): 
            lablefile = self.labelFiles[index]
            print(lablefile)
            result = re.search(r'/(\d+)\.label$', lablefile)
            baseScene = result.group(1)
            population = self.initPopulations(baseScene)
            self.saveMutatedFiles(self.binPath, self.labelPath, self.modifidePrediction, population, baseScene, 0)

    
    def random_all(self):
        total_index = 0 
        for index in range(len(self.labelFiles)): 
            lablefile = self.labelFiles[index]
            print(lablefile)
            result = re.search(r'/(\d+)\.label$', lablefile)
            baseScene = result.group(1)
            # population, total_index = self.initPopulations_from_record(baseScene, total_index)
            population = self.initPopulations_random_all(baseScene)
            self.saveMutatedFiles_random_all(self.binPath, self.labelPath, self.modifidePrediction, population, baseScene)
            self.evaluate_random_all(self.binPath, self.basePredictionPath, self.labelPath, self.modifidePrediction, population, self.baseSequence, baseScene)
            self.record_random_all(population, baseScene)

    def record(self, population, baseScene, generation):
        recore_title = baseScene + "-------" + str(generation) + '\n'
        recordpath = "/home/LiDFuzzer/record/" + "scene{}-{}-weather.txt".format(self.baseSequence,self.model)
        fileIO.write_record(recore_title, recordpath)
        for ind in population:
            fileIO.record_dict_to_file(ind.compare_result, recordpath)

    def record_random_all(self, population, baseScene):
        recordpath = "/home/LiDFuzzer/record/" + "scene{}-{}-Random.txt".format(self.baseSequence, self.model)
        generation = -1
        for index, ind in enumerate(population):
            if index%20 == 0:
                generation += 1 
                recore_title = baseScene + "-------" + str(generation) + '\n'
                fileIO.write_record(recore_title, recordpath)  
            fileIO.record_dict_to_file(ind.compare_result, recordpath) 

    def initPopulations(self, baseScene):
        population = []
        if self.pk == False:
            for i in range(self.individualnums):
                ind = Individual(self.IfInstance, self.IfWeather, self.max_radius, self.gene_numbers, self.instances_range, self.weather_range, self.scale, self.angle, self.intensity_range, self.pk, self.model, \
                                self.baseLabelPath, self.baseBinPath, self.basePredictionPath, self.baseSequence, baseScene)
                ind.initial(ind.model, self.assetRepository, ind.genenums, ind.instances_range, ind.visual, ind.baseBinPath, ind.baseLabelPath, ind.baseSequence, ind.baseScene, ind.basePredictionPath)
                population.append(ind)
            # poprecord = "/home/LiDFuzzer/seed/" + baseScene + "population.pkl"
            # fileIO.writepopulation(poprecord, population)
        else:
            poprecord = "/home/LiDFuzzer/seed/" + baseScene + "population.pkl"
            history_population = fileIO.readpopulation(poprecord)
            for i in range(self.individualnums):
                ind = Individual(self.IfInstance, self.IfWeather, self.max_radius, self.gene_numbers, self.instances_range, self.weather_range, self.scale, self.angle, self.intensity_range, self.pk, self.model, \
                                    self.baseLabelPath, self.baseBinPath, self.basePredictionPath, self.baseSequence, baseScene)
                ind.initial_pk(ind.model, self.assetRepository, self.assetAllInstances, ind.genenums, ind.instances_range, ind.visual, ind.baseBinPath, ind.baseLabelPath, ind.baseSequence, ind.baseScene, ind.basePredictionPath, history_population[i])
                population.append(ind)
        return population
    
    def initPopulations_random(self, baseScene):
        population = []
        for i in range(self.individualnums):
            ind = Individual(self.IfInstance, self.IfWeather, self.max_radius, self.gene_numbers, self.instances_range, self.weather_range, self.scale, self.angle, self.intensity_range, self.pk, self.model, \
                            self.baseLabelPath, self.baseBinPath, self.basePredictionPath, self.baseSequence, baseScene)
            ind.initial_random(ind.model, self.assetRepository, ind.genenums, ind.instances_range, ind.visual, ind.baseBinPath, ind.baseLabelPath, ind.baseSequence, ind.baseScene, ind.basePredictionPath)
            population.append(ind)
        return population
    
    def initPopulations_random_all(self, baseScene):
        population = []
        for i in range(self.individualnums*self.termination):
            ind = Individual(self.IfInstance, self.IfWeather, self.max_radius, self.gene_numbers, self.instances_range, self.weather_range, self.scale, self.angle, self.intensity_range, self.pk, self.model, \
                            self.baseLabelPath, self.baseBinPath, self.basePredictionPath, self.baseSequence, baseScene)
            # ind.initial_more_instances(ind.model, self.assetRepository, ind.genenums, ind.instances_range, ind.visual, ind.baseBinPath, ind.baseLabelPath, ind.baseSequence, ind.baseScene, ind.basePredictionPath)
            ind.initial(ind.model, self.assetRepository, ind.genenums, ind.instances_range, ind.visual, ind.baseBinPath, ind.baseLabelPath, ind.baseSequence, ind.baseScene, ind.basePredictionPath)
            population.append(ind)
        return population
    
    def initPopulations_from_record(self, baseScene, total_index):
        population = []
        for i in range(self.individualnums*self.termination):
            ind = Individual(self.IfInstance, self.IfWeather, self.max_radius, self.gene_numbers, self.instances_range, self.weather_range, self.scale, self.angle, self.intensity_range, self.pk, self.model, \
                            self.baseLabelPath, self.baseBinPath, self.basePredictionPath, self.baseSequence, baseScene)
            # ind.initial_more_instances(ind.model, self.assetRepository, ind.genenums, ind.instances_range, ind.visual, ind.baseBinPath, ind.baseLabelPath, ind.baseSequence, ind.baseScene, ind.basePredictionPath)
            flag = ind.initial_from_record(ind.genenums, "/home/LiDFuzzer/output_data copy.csv",self.assetRepository, total_index)
            total_index += 1
            if flag:
                population.append(ind)
        return population,total_index

    def evaluate(self, binPath, basePredictionPath, labelPath, modifidePrediction, population, baseSequence, baseScene, generation):
        binpath = binPath + "/" + self.model + "/" + baseScene + "/" + str(generation) + "/"
        saveLabel = labelPath + "/" + self.model + "/" + baseScene + "/" + str(generation) + "/"
        saveModel = modifidePrediction + "/" + baseScene + "/" + str(generation) + "/"
        predlabelpath = basePredictionPath + "/" + baseSequence + "/" + self.model + "/" + "compress_predlabel" + "/" + baseScene + "/" + str(generation) + "/"
        if os.path.exists(predlabelpath):
            shutil.rmtree(predlabelpath)
            os.makedirs(predlabelpath)
            print("Save mutated prediction folder alrealdy have done!")
        else:
            os.makedirs(predlabelpath)
            print("Save mutated prediction folder alrealdy have done!")

        # if self.model == "Cylinder3D" or self.model == "SPVCNN" or self.model == "MinkuNet":
        #     run_inference(model = self.model, pcd_data_list = binpath, out_dir = predlabelpath)
        #     fileIO.changeJsonLabel(predlabelpath)
        if self.model == "Cylinder3D":
                # runCommand = "python3 demo_folder.py"
                # runCommand += " --demo-folder {}".format(binpath)
                # runCommand += " --save-folder {}".format(predlabelpath)
                # modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel)
                runCommand = "python test.py"
                runCommand += " /home/LiDFuzzer/suts/mmdetection3d/configs/cylinder3d/cylinder3d_4xb4-3x_semantickitti.py"
                runCommand += " /home/LiDFuzzer/suts/mmdetection3d/checkpoints/cylinder3d_4xb4_3x_semantickitti_20230318_191107-822a8c31.pth"
                modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel, predlabelpath)
            elif self.model == "SPVNAS":
                runCommand = "python test.py"
                runCommand += " /home/LiDFuzzer/suts/mmdetection3d/configs/spvcnn/spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti.py"
                runCommand += " /home/LiDFuzzer/suts/mmdetection3d/checkpoints/spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_125908-d68a68b7.pth"
                modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel, predlabelpath)
            elif self.model == "JS3C-Net":
                runCommand = "python test_kitti_segment.py"
                runCommand += " --log_dir JS3C-Net-kitti --gpu 0"
                modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel, predlabelpath)
            elif self.model == "FRNet":
                runCommand = "python test.py"
                runCommand += " /home/LiDFuzzer/suts/FRNet/configs/frnet/frnet-semantickitti_seg.py"
                runCommand += " /home/LiDFuzzer/suts/FRNet/frnet-semantickitti_seg.pth"
                modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel, predlabelpath)
            elif self.model == "SphereFormer":
                runCommand = "python train.py"
                runCommand += " --config /home/LiDFuzzer/suts/SphereFormer/config/semantic_kitti/semantic_kitti_unet32_spherical_transformer.yaml"
                modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel, predlabelpath)
            if modelstatu == 0:
                print("Model prediction completed")
            else:
                print("An error occurred during the prediction process")
                return
        modifiedLabels = getLabels(saveLabel)
        modifiedPredictions = getLabels(saveModel)
        newPredictions = getLabels(predlabelpath)

        # for i in range(self.individualnums):
        #     compare_result = evalLabels(modifiedLabels[i], modifiedPredictions[i], newPredictions[i], population[i], self.model)
        #     print(compare_result)
        #     population[i].compare_result = compare_result
        #     population[i].fitness = [compare_result["percentLossAcc"], compare_result["percentLossJac"]]
        for i in range(self.individualnums):
            compare_result = evalLabels(modifiedLabels[i], modifiedPredictions[i], newPredictions[i], population[i], self.model)
            print(compare_result)
            population[i].compare_result = compare_result
            population[i].fitness = [compare_result["percentLossAcc"], compare_result["percentLossJac"]]


    def evaluate_random_all(self, binPath, basePredictionPath, labelPath, modifidePrediction, population, baseSequence, baseScene):
            binpath = binPath + "/" + self.model + "/" + baseScene + "/" 
            saveLabel = labelPath + "/" + self.model + "/" + baseScene + "/" 
            saveModel = modifidePrediction + "/" + baseScene + "/" 
            predlabelpath = basePredictionPath + "/" + baseSequence + "/" + self.model + "/" + "compress_predlabel" + "/" + baseScene + "/"
            if os.path.exists(predlabelpath):
                shutil.rmtree(predlabelpath)
                os.makedirs(predlabelpath)
                print("Save mutated prediction folder alrealdy have done!")
            else:
                os.makedirs(predlabelpath)
                print("Save mutated prediction folder alrealdy have done!")

            # if self.model == "Cylinder3D" or self.model == "SPVCNN" or self.model == "MinkuNet":
            #     run_inference(model = self.model, pcd_data_list = binpath, out_dir = predlabelpath)
            #     fileIO.changeJsonLabel(predlabelpath)
            if self.model == "Cylinder3D":
                # runCommand = "python3 demo_folder.py"
                # runCommand += " --demo-folder {}".format(binpath)
                # runCommand += " --save-folder {}".format(predlabelpath)
                # modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel)
                runCommand = "python test.py"
                runCommand += " /home/LiDFuzzer/suts/mmdetection3d/configs/cylinder3d/cylinder3d_4xb4-3x_semantickitti.py"
                runCommand += " /home/LiDFuzzer/suts/mmdetection3d/checkpoints/cylinder3d_4xb4_3x_semantickitti_20230318_191107-822a8c31.pth"
                modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel, predlabelpath)
            elif self.model == "SPVNAS":
                runCommand = "python test.py"
                runCommand += " /home/LiDFuzzer/suts/mmdetection3d/configs/spvcnn/spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti.py"
                runCommand += " /home/LiDFuzzer/suts/mmdetection3d/checkpoints/spvcnn_w32_8xb2-amp-laser-polar-mix-3x_semantickitti_20230425_125908-d68a68b7.pth"
                modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel, predlabelpath)
            elif self.model == "JS3C-Net":
                runCommand = "python test_kitti_segment.py"
                runCommand += " --log_dir JS3C-Net-kitti --gpu 0"
                modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel, predlabelpath)
            elif self.model == "FRNet":
                runCommand = "python test.py"
                runCommand += " /home/LiDFuzzer/suts/FRNet/configs/frnet/frnet-semantickitti_seg.py"
                runCommand += " /home/LiDFuzzer/suts/FRNet/frnet-semantickitti_seg.pth"
                modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel, predlabelpath)
            elif self.model == "SphereFormer":
                runCommand = "python train.py"
                runCommand += " --config /home/LiDFuzzer/suts/SphereFormer/config/semantic_kitti/semantic_kitti_unet32_spherical_transformer.yaml"
                modelstatu = runModel(None, runCommand, self.model, binpath, saveLabel, predlabelpath)
            if modelstatu == 0:
                print("Model prediction completed")
            else:
                print("An error occurred during the prediction process")
                return
            modifiedLabels = getLabels(saveLabel)
            modifiedPredictions = getLabels(saveModel)
            newPredictions = getLabels(predlabelpath)

            # for i in range(self.individualnums):
            #     compare_result = evalLabels(modifiedLabels[i], modifiedPredictions[i], newPredictions[i], population[i], self.model)
            #     print(compare_result)
            #     population[i].compare_result = compare_result
            #     population[i].fitness = [compare_result["percentLossAcc"], compare_result["percentLossJac"]]
            for i in range(len(newPredictions)):
                compare_result = evalLabels(modifiedLabels[i], modifiedPredictions[i], newPredictions[i], population[i], self.model)
                compare_result
                print(compare_result)
                population[i].compare_result = compare_result
                population[i].fitness = [compare_result["percentLossAcc"], compare_result["percentLossJac"]]


    def getFitnessList(self, population):
        all_fitness_list = []
        for pop in population:
            all_fitness_list.append(pop.fitness)
        return np.array(all_fitness_list)
    
    def rankandcrowdingsurvival(self, population, n_survive=None, nds=None):
        F = self.getFitnessList(population).astype(float, copy=False)
        survivors = []
        nds = nds if nds is not None else NonDominatedSorting()
        fronts = nds.do(F, n_stop_if_ranked=n_survive)
        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                population[i].set("rank", k)
                population[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        # select_populations = copy.deepcopy([population[i] for i in survivors])
        return [population[i] for i in survivors]

    def tournament_selection(self, population, tournament_size):
        random.seed(time.time())
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda x: x.rank)
        winners = [tournament[0]]
        for competitor in tournament[1:]:
            if competitor.rank == winners[0].rank:
                winners.append(competitor)
            else:
                break
        winner = max(winners, key=lambda x: x.crowding)
        winner_copy = copy.deepcopy(winner)
        return winner_copy

    def saveMutatedFiles(self, binPath, labelPath, modifidePrediction, population, baseScene, generation):
        saveVelodyne = binPath + "/" + self.model + "/" + baseScene + "/" + str(generation) + "/"
        if os.path.exists(saveVelodyne):
            shutil.rmtree(saveVelodyne)
            os.makedirs(saveVelodyne)
            print("Save bin folder alrealdy have done!")
        else:
            os.makedirs(saveVelodyne)
            print("Save bin folder alrealdy have done!")

        saveLabel = labelPath+ "/" + self.model + "/" + baseScene + "/" + str(generation) + "/"
        if os.path.exists(saveLabel):
            shutil.rmtree(saveLabel)
            os.makedirs(saveLabel)
            print("Save label folder alrealdy have done!")
        else:
            os.makedirs(saveLabel)
            print("Save label folder alrealdy have done!")

        saveModifiedPre = modifidePrediction+ "/" + baseScene + "/" + str(generation) + "/"
        if os.path.exists(saveModifiedPre):
            shutil.rmtree(saveModifiedPre)
            os.makedirs(saveModifiedPre)
            print("Save mutated prediction folder alrealdy have done!")
        else:
            os.makedirs(saveModifiedPre)
            print("Save mutated prediction folder alrealdy have done!")
        # #save instance 
        # for index, ind in enumerate(population):
        #     filename = str(index).rjust(6, '0')
        #     modify_instances = np.zeros(np.shape(ind.modelPredictsScene)[0], np.int32)
        #     fileIoUtil.saveBinLabelPair(ind.pcdArr_after_instance, ind.intensity_after_instance, ind.semantics, ind.instances, saveVelodyne, saveLabel, filename)
        #     fileIoUtil.savemodifySemantics(ind.modelPredictsScene, modify_instances, saveModifiedPre, filename)
        #     print(ind.pcdArr_after_instance.shape, ind.intensity_after_instance.shape, ind.semantics.shape, ind.instances.shape, ind.modelPredictsScene.shape)
        #save instance and weather
        for index, ind in enumerate(population):
            filename = str(index).rjust(6, '0')
            if self.IfWeather == True:
                weather_gene = ind.individual["weather"][0]
                weather_indice = weather_gene.get_indice()
                weather_type = weather_gene.get_weather_type()
                if weather_type == "rain":
                    semantics = ind.semantics[weather_indice]
                    instances = ind.instances[weather_indice]
                    modelPredictsScene = ind.modelPredictsScene[weather_indice]
                    # print(ind.modelPredictsScene.shape, ind.instances.shape)
                    modify_instances = np.zeros(np.shape(modelPredictsScene)[0], np.int32)
                if weather_type == "snow" or  weather_type == "fog":
                    add_semantics = np.full((weather_indice,) + ind.semantics.shape[1:], 200)
                    add_instances = np.full((weather_indice,) + ind.instances.shape[1:], 0) 
                    semantics = np.hstack([ind.semantics, add_semantics])
                    instances = np.hstack([ind.instances, add_instances])
                    modelPredictsScene = np.hstack([ind.modelPredictsScene, add_semantics])
                    # print(ind.modelPredictsScene.shape, ind.instances.shape)
                    modify_instances = np.zeros(np.shape(modelPredictsScene)[0], np.int32)
                if weather_type == None:
                    semantics = ind.semantics
                    instances = ind.instances
                    modelPredictsScene = ind.modelPredictsScene
                    modify_instances = np.zeros(np.shape(ind.modelPredictsScene)[0], np.int32)
                fileIoUtil.saveBinLabelPair(ind.pcdArr, ind.intensity, semantics, instances, saveVelodyne, saveLabel, filename)
                fileIoUtil.savemodifySemantics(modelPredictsScene, modify_instances, saveModifiedPre, filename)
                print(ind.pcdArr.shape, ind.intensity.shape, semantics.shape, instances.shape, modelPredictsScene.shape, weather_type)
            else:
                fileIoUtil.saveBinLabelPair(ind.pcdArr, ind.intensity, ind.semantics, ind.instances, saveVelodyne, saveLabel, filename)
                fileIoUtil.saveLabelSemantics(ind.modelPredictsScene, saveModifiedPre, filename)
        # for index, ind in enumerate(population):
        #     filename = str(index).rjust(6, '0')
        #     weather_gene = ind.individual["weather"][0]
        #     weather_indice = weather_gene.get_indice()
        #     weather_type = weather_gene.get_weather_type()
        #     if weather_type == "rain":
        #         ind.semantics = ind.semantics[weather_indice]
        #         ind.instances = ind.instances[weather_indice]
        #         ind.modelPredictsScene = ind.modelPredictsScene[weather_indice]
        #         # print(ind.modelPredictsScene.shape, ind.instances.shape)
        #         modify_instances = np.zeros(np.shape(ind.modelPredictsScene)[0], np.int32)
        #     if weather_type == "snow" or  weather_type == "fog":
        #         add_semantics = np.full((weather_indice,) + ind.semantics.shape[1:], 300)
        #         add_instances = np.full((weather_indice,) + ind.instances.shape[1:], 0) 
        #         ind.semantics = np.hstack([ind.semantics, add_semantics])
        #         ind.instances = np.hstack([ind.instances, add_instances])
        #         ind.modelPredictsScene = np.hstack([ind.modelPredictsScene, add_semantics])
        #         # print(ind.modelPredictsScene.shape, ind.instances.shape)
        #         modify_instances = np.zeros(np.shape(ind.modelPredictsScene)[0], np.int32)
        #     if weather_type == None:
        #         modify_instances = np.zeros(np.shape(ind.modelPredictsScene)[0], np.int32)
        #     fileIoUtil.saveBinLabelPair(ind.pcdArr, ind.intensity, ind.semantics, ind.instances, saveVelodyne, saveLabel, filename)
        #     fileIoUtil.savemodifySemantics(ind.modelPredictsScene, modify_instances, saveModifiedPre, filename)
        #     print(ind.pcdArr.shape, ind.intensity.shape, ind.semantics.shape, ind.instances.shape, ind.modelPredictsScene.shape, weather_type)

    def saveMutatedFiles_random_all(self, binPath, labelPath, modifidePrediction, population, baseScene):
        saveVelodyne = binPath + "/" + self.model + "/" + baseScene + "/"
        if os.path.exists(saveVelodyne):
            shutil.rmtree(saveVelodyne)
            os.makedirs(saveVelodyne)
            print("Save bin folder alrealdy have done!")
        else:
            os.makedirs(saveVelodyne)
            print("Save bin folder alrealdy have done!")

        saveLabel = labelPath+ "/" + self.model + "/" + baseScene + "/"
        if os.path.exists(saveLabel):
            shutil.rmtree(saveLabel)
            os.makedirs(saveLabel)
            print("Save label folder alrealdy have done!")
        else:
            os.makedirs(saveLabel)
            print("Save label folder alrealdy have done!")

        saveModifiedPre = modifidePrediction+ "/" + baseScene + "/"
        if os.path.exists(saveModifiedPre):
            shutil.rmtree(saveModifiedPre)
            os.makedirs(saveModifiedPre)
            print("Save mutated prediction folder alrealdy have done!")
        else:
            os.makedirs(saveModifiedPre)
            print("Save mutated prediction folder alrealdy have done!")

        for index, ind in enumerate(population):
            filename = str(index).rjust(6, '0')
            if self.IfWeather == True:
                weather_gene = ind.individual["weather"][0]
                weather_indice = weather_gene.get_indice()
                weather_type = weather_gene.get_weather_type()
                if weather_type == "rain":
                    semantics = ind.semantics[weather_indice]
                    instances = ind.instances[weather_indice]
                    modelPredictsScene = ind.modelPredictsScene[weather_indice]
                    # print(ind.modelPredictsScene.shape, ind.instances.shape)
                    modify_instances = np.zeros(np.shape(modelPredictsScene)[0], np.int32)
                if weather_type == "snow" or  weather_type == "fog":
                    add_semantics = np.full((weather_indice,) + ind.semantics.shape[1:], 200)
                    add_instances = np.full((weather_indice,) + ind.instances.shape[1:], 0) 
                    semantics = np.hstack([ind.semantics, add_semantics])
                    instances = np.hstack([ind.instances, add_instances])
                    modelPredictsScene = np.hstack([ind.modelPredictsScene, add_semantics])
                    # print(ind.modelPredictsScene.shape, ind.instances.shape)
                    modify_instances = np.zeros(np.shape(modelPredictsScene)[0], np.int32)
                if weather_type == None:
                    semantics = ind.semantics
                    instances = ind.instances
                    modelPredictsScene = ind.modelPredictsScene
                    modify_instances = np.zeros(np.shape(ind.modelPredictsScene)[0], np.int32)
                fileIoUtil.saveBinLabelPair(ind.pcdArr, ind.intensity, semantics, instances, saveVelodyne, saveLabel, filename)
                fileIoUtil.savemodifySemantics(modelPredictsScene, modify_instances, saveModifiedPre, filename)
                print(ind.pcdArr.shape, ind.intensity.shape, semantics.shape, instances.shape, modelPredictsScene.shape, weather_type)
            else:
                fileIoUtil.saveBinLabelPair(ind.pcdArr, ind.intensity, ind.semantics, ind.instances, saveVelodyne, saveLabel, filename)
                fileIoUtil.saveLabelSemantics(ind.modelPredictsScene, saveModifiedPre, filename)


    # def crossover(self, parent1, parent2, crossover):
    #         if crossover >= random.random():
    #             if self.IfInstance:
    #                 start_index = random.randint(0, self.gene_numbers - 2)
    #                 end_index = random.randint(start_index + 1, self.gene_numbers - 1)
                    
    #                 # 获取父代的实例信息
    #                 individualInstance1 = parent1.individual["instance"]
    #                 individualInstance2 = parent2.individual["instance"]
                    
    #                 # 深拷贝原始实例以便在需要时恢复
    #                 originalIndividualInstance1 = copy.deepcopy(individualInstance1)
    #                 originalIndividualInstance2 = copy.deepcopy(individualInstance2)

    #                 # 交换两个父代在选定区间的元素
    #                 individualInstance1[start_index:end_index+1], individualInstance2[start_index:end_index+1] = \
    #                 individualInstance2[start_index:end_index+1], individualInstance1[start_index:end_index+1]

    #                 # 更新父代实例
    #                 parent1.individual["instance"] = individualInstance1
    #                 parent2.individual["instance"] = individualInstance2

    #                 # 尝试生成新的PCD实例，检查是否有效
    #                 crossoverFlag1 = parent1.crossover_generate_pcd_instance(self.assetRepository)
    #                 crossoverFlag2 = parent2.crossover_generate_pcd_instance(self.assetRepository)

    #                 if not crossoverFlag1:
    #                     parent1.individual["instance"] = originalIndividualInstance1
                    
    #                 if not crossoverFlag2:
    #                     parent2.individual["instance"] = originalIndividualInstance2

    #                 return parent1, parent2, crossoverFlag1, crossoverFlag2
    #         else:
    #             return parent1, parent2, True, True
        
    def crossover(self, parent1, parent2, crossover):
            if crossover >= random.random():
                if self.IfInstance:
                    start_index = random.randint(0, self.gene_numbers - 2)
                    end_index = random.randint(start_index + 1, self.gene_numbers - 1)
                    
                    individualInstance1 = parent1.individual["instance"]
                    individualInstance2 = parent2.individual["instance"]
                    
                    originalIndividualInstance1 = copy.deepcopy(individualInstance1)
                    originalIndividualInstance2 = copy.deepcopy(individualInstance2)

                    individualInstance1[start_index:end_index+1], individualInstance2[start_index:end_index+1] = \
                    individualInstance2[start_index:end_index+1], individualInstance1[start_index:end_index+1]

                    parent1.individual["instance"] = individualInstance1
                    parent2.individual["instance"] = individualInstance2

                    crossoverFlag1 = parent1.crossover_generate_pcd_instance(self.assetRepository)
                    crossoverFlag2 = parent2.crossover_generate_pcd_instance(self.assetRepository)

                    if not crossoverFlag1:
                        parent1.individual["instance"] = originalIndividualInstance1
                    
                    if not crossoverFlag2:
                        parent2.individual["instance"] = originalIndividualInstance2

                    return parent1, parent2, crossoverFlag1, crossoverFlag2
            else:
                return parent1, parent2, True, True
        
    def mutate(self, child, mutation):
        if mutation >= random.random():
            index = random.randint(0, self.gene_numbers)
            mutationflag = child.mutation_generate_pcd_instance(index, self.assetRepository)
            if mutationflag == False:
                return False
            else:
                return True
        return True

    # def mutate(self, child, mutation):
    #     if mutation >= random.random():
    #         index = random.randint(0, self.gene_numbers)
    #         mutationflag = child.mutation_generate_pcd_instance(index, self.assetRepository)
    #         if mutationflag == False:
    #             return False
    #         else:
    #             return True
    #     return True

    def getbestvalue(self, population):
        best_accuracy, best_jaccard = [],[]
        for ind in population:
            best_accuracy.append(ind.fitness[0])
            best_jaccard.append(ind.fitness[1]) 
        
        return copy.deepcopy(min(best_accuracy)),copy.deepcopy(min(best_jaccard))


