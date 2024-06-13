"""
AssetRepository 
Handles all database interaction for assets

@Date 6/23/22
"""


import pymongo
import numpy as np

import data.fileIoUtil as fileIoUtil
import data.mongoRepository as mongoRepository

# --------------------------------------------------------------------------

class AssetRepository(mongoRepository.MongoRepository):
    def __init__(self, binPath, labelPath, mongoConnectPath, mdbtype):
        super(AssetRepository, self).__init__(mongoConnectPath)
        self.binPath = binPath
        self.labelPath = labelPath       
        self.assetCollection = self.db[mdbtype]



    """
    Gets an asset by the id
    """
    def getAssetById(self, id):

        asset = self.assetCollection.find_one({ "_id" : id })

        return self.getInstanceFromAssetRecord(asset)



    def getAllInstancesofScene(self, sequence, scene):

        asset = self.assetCollection.aggregate([
            { "$match": { "sequence" : sequence, "scene" : scene} },
            { "$sample": { "size": 1 } }
        ])

        assetRecord = None
        try:
            assetRecord = asset.next()
        except:
            # print("Get assetRecord failed")
            return None, None, None, None, None

        return assetRecord
    

    """
    Gets a random asset from a specific type , model and distance 
    """
    def getRandomAssetWithinModelDistType(self, model, distIndex, type):

        asset = self.assetCollection.aggregate([
            { "$match": {"model" : model, "dist_index" : distIndex, "type" : type} },
            { "$sample": { "size": 1 } }
        ])

        assetRecord = None
        try:
            assetRecord = asset.next()
        except:
            # print("Get assetRecord failed")
            return None, None, None, None, None

        return self.getInstanceFromAssetRecord(assetRecord)

    
    """
    Gets a random asset from a specific type , model and distance 
    """
    def getAssetWithinModelDistTypeNum(self, model, distIndex, type):

        asset = self.assetCollection.aggregate([
            { "$match": {"model" : model, "dist_index" : distIndex, "type" : type} },
            {"$sort": {"type-points": pymongo.ASCENDING}}
        ])

        assetRecord = None
        try:
            assetRecord = asset.next()
        except:
            # print("Get assetRecord failed")
            return None, None, None, None, None

        return self.getInstanceFromAssetRecord(assetRecord)
    

    """
    Gets a random asset from a specific type , model and distance 
    """
    def getAssetWithinDistTypeNumRandom(self, distIndex, type):

        asset = self.assetCollection.aggregate([
            { "$match": {"dist_index" : distIndex, "type" : type} },
              { "$sample": { "size": 1 } }
        ])

        assetRecord = None
        try:
            assetRecord = asset.next()
        except:
            # print("Get assetRecord failed")
            return None, None, None, None, None

        return self.getInstanceFromAssetRecord(assetRecord)
    
        """
    Gets a random asset from a specific type , model and distance 
    """
    def getAssetWithinDistTypeNumRandomSequence(self, distIndex, type, sequence):

        asset = self.assetCollection.aggregate([
            { "$match": {"dist_index" : distIndex, "type" : type, "sequence": sequence} },
            { "$sample": { "size": 1 } }
        ])

        assetRecord = None
        try:
            assetRecord = asset.next()
        except:
            # print("Get assetRecord failed")
            return None, None, None, None, None

        return self.getInstanceFromAssetRecord(assetRecord)

    def getAssetWithinModelDistTypeNumMore(self, model, distIndex, type, successnum):

        asset = self.assetCollection.aggregate([
            { "$match": {"model" : model, "dist_index" : distIndex, "type" : type} },
            {"$sort": {"type-points": pymongo.ASCENDING}}
        ])

        assetRecord = None
        try:
            for _ in range(successnum + 1):
                assetRecord = asset.next()
            if assetRecord == None:
                return None, None, None, None, None
        except:
            # print("Get assetRecord failed")
            return None, None, None, None, None

        return self.getInstanceFromAssetRecord(assetRecord)

    """
    Gets a random asset from a specific type , model and distance 
    """
    def getRandomAssetWithinModelDist(self, model, distIndex):

        asset = self.assetCollection.aggregate([
            { "$match": {"model" : model, "dist_index" : distIndex} },
            { "$sample": { "size": 1 } }
        ])

        assetRecord = None
        try:
            assetRecord = asset.next()
        except:
            print("Get assetRecord failed")
            return None, None, None, None, None

        return self.getInstanceFromAssetRecord(assetRecord)

    """
    Gets a random asset of type
    """
    def getRandomAssetOfType(self, type):

        asset = self.assetCollection.aggregate([
            { "$match": { "type" : type } },
            { "$sample": { "size": 1 } }
        ])

        assetRecord = None
        try:
            assetRecord = asset.next()
        except:
            print("Get assetRecord failed")
            return None, None, None, None, None

        return self.getInstanceFromAssetRecord(assetRecord)



    """
    Gets a random asset
    """
    def getRandomAsset(self):

        asset = self.assetCollection.aggregate([
            { "$sample": { "size": 1 } }
        ])

        assetRecord = None
        try:
            assetRecord = asset.next()
        except:
            print("Get assetRecord failed")
            return None, None, None, None, None

        return self.getInstanceFromAssetRecord(assetRecord)



    """
    Gets a random asset of specified types
    """
    def getRandomAssetOfTypes(self, typeNums):

        typeQuery = []
        for type in typeNums:
            typeQuery.append({"typeNum": type})

        asset = self.assetCollection.aggregate([
            { "$match": {  
                "$or":  typeQuery
            }},
            { "$sample": { "size": 1 } }
        ])

        assetRecord = None
        try:
            assetRecord = asset.next()
        except:
            print("Get assetRecord failed")
            return None, None, None, None, None

        return self.getInstanceFromAssetRecord(assetRecord)
    


    """
    Gets a random asset of specified types
    """
    def getRandomAssetOfTypesWithinScene(self, typeNums, sequence, scene):

        typeQuery = []
        for type in typeNums:
            typeQuery.append({"typeNum": type})

        asset = self.assetCollection.aggregate([
            { "$match": {  
                "sequence" : sequence, 
                "scene" : scene,
                "$or":  typeQuery
            }},
            { "$sample": { "size": 1 } }
        ])

        assetRecord = None
        try:
            assetRecord = asset.next()
        except:
            print("Get assetRecord failed")
            return None, None, None, None, None

        return self.getInstanceFromAssetRecord(assetRecord)



    """
    Gets the data from a given asset Record 
    """
    # def getInstanceFromAssetRecord(self, assetRecord):

    #     allinstances = assetRecord["all-instances"]
    #     sequence = assetRecord["sequence"]
    #     scene = assetRecord["scene"]
    #     pcdArr, intensity, semantics, labelInstance = fileIoUtil.openLabelBin(self.binPath, self.labelPath, sequence, scene)
    #     maskOnlyInst = np.zeros_like(labelInstance, dtype=bool)
    #     if isinstance(allinstances, int):
    #         maskOnlyInst = (labelInstance == allinstances)

    #     elif isinstance(allinstances, list):
    #         for ins in allinstances:
    #             maskOnlyInst |= (labelInstance == ins)
    #     pcdArr = pcdArr[maskOnlyInst, :]
    #     intensity = intensity[maskOnlyInst]
    #     semantics = semantics[maskOnlyInst]
    #     labelInstance = labelInstance[maskOnlyInst]

    #     return pcdArr, intensity, semantics, labelInstance, assetRecord

    def getInstanceFromAssetRecord(self, assetRecord):
        allinstances = assetRecord["all-instances"]
        sequence = assetRecord["sequence"]
        scene = assetRecord["scene"]
        # pcdArr, intensity, semantics, labelInstance = fileIoUtil.openLabelBin(self.binPath, self.labelPath, sequence, scene)
        pcdArr, intensity, semantics, labelInstance = fileIoUtil.openInstancesLabelBin(self.binPath, self.labelPath, sequence, scene)
        # Ensure allinstances is a list
        if isinstance(allinstances, int):
            allinstances = [allinstances]  # Make a single-element list if it's an int

        # Initialize the mask as False for all elements
        maskOnlyInst = np.zeros_like(labelInstance, dtype=bool)

        # Update the mask for each instance
        for ins in allinstances:
            maskOnlyInst |= (labelInstance == ins)
        # Apply the mask to filter the arrays
        pcdArr = pcdArr[maskOnlyInst, :]
        intensity = intensity[maskOnlyInst]
        semantics = semantics[maskOnlyInst]
        labelInstance = labelInstance[maskOnlyInst]
        return pcdArr, intensity, semantics, labelInstance, assetRecord




    """
    Page Request for asset record
    Sorted by _id
    """
    def getAssetsPaged(self, page, pageLimit):
        return self.assetCollection.find({}).sort([("_id", pymongo.ASCENDING)]).skip((page - 1) * pageLimit).limit(pageLimit)


    def removeinstance(self, sequence):
        query = {"sequence": sequence}  # 查询条件
        self.assetCollection.delete_many(query)








