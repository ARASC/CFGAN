# -*- coding: utf-8 -*-
"""
Author:
    Xuxin Zhang,xuxinz@qq.com
Reference: Chae D K , Kang J S , Kim S W , et al.
CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks[C]// the 27th ACM International Conference. ACM, 2018.

"""

from collections import defaultdict
import torch
import pandas as pd


def loadUseInfo(trainFile,splitMark):
    print(trainFile)
    UseInfo = pd.DataFrame(columns=["userId","useAge","useGender","useOccupation","useZipcode"])
    index = 0
    for line in open(trainFile):
        userId, useAge, useGender, useOccupation, useZipcode = line.strip().split(splitMark)
        UseInfo.loc['%d'%index]=[userId, useAge, useGender, useOccupation, useZipcode]
        index = index + 1
    UseInfo.to_csv("data/ml-100k/userInfo.csv",index=False)
    print("Demographic information about the users loading done: %d users" % (index))
    return UseInfo 

def loadItemInfo(trainFile,splitMark):
#    trainFile = "data/ml-100k/u.item"
#    splitMark =  "|"
    print(trainFile)
    ItemInfo = pd.DataFrame(columns=["movie_id","movie_title","release_date","video_release_date",
                                     "IMDb_URL","unknowngenres","Action","Adventure",
                                     "Animation","Childrens","Comedy","Crime",
                                     "Documentary","Drama","Fantasy","Film_Noir",
                                     "Horror","Musical","Mystery","Romance",
                                     "Sci_Fi","Thriller","War","Western"])
    index = 0
    for line in open(trainFile,encoding = 'ISO-8859-1'):
        ItemInfo.loc['%d'%index]=line.strip().split(splitMark)
        index = index + 1
    ItemInfo.to_csv("data/ml-100k/itemInfo.csv")
    print("Information about the items loading done: %d users" % (index))
    return ItemInfo  


def loadTrainingData(trainFile,splitMark):
    trainSet = defaultdict(list) 
    max_u_id = -1
    max_i_id = -1
    for line in open(trainFile):
        userId, itemId, rating,_ = line.strip().split(splitMark)
        userId = int(userId)
        itemId = int(itemId)
        trainSet[userId].append(itemId)
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id + 1
    itemCount = max_i_id + 1
    print("Training data loading done" )
    return trainSet,userCount,itemCount  


def loadTestData(testFile,splitMark):
    testSet = defaultdict(list)
    max_u_id = -1
    max_i_id = -1
    for line in open(testFile):
        userId, itemId, rating,_ = line.strip().split(splitMark)
        userId = int(userId)
        itemId = int(itemId)
        testSet[userId].append(itemId)
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id + 1
    itemCount = max_i_id + 1
    print("Test data loading done")
    return testSet,userCount,itemCount


def to_Vectors(trainSet, userCount, itemCount, userList_test, mode):
    
    testMaskDict = defaultdict(lambda: [0] * itemCount)
    batchCount = userCount #改动  直接写成userCount
    if mode == "itemBased":#改动  itemCount userCount互换   batchCount是物品数
        userCount = itemCount
        itemCount = batchCount
        batchCount = userCount
    trainDict = defaultdict(lambda: [0] * itemCount)
    for userId, i_list in trainSet.items():
        for itemId in i_list:
            testMaskDict[userId][itemId] = -99999
            if mode == "userBased":
                trainDict[userId][itemId] = 1.0
            else:
                trainDict[itemId][userId] = 1.0

    trainVector = []
    for batchId in range(batchCount):
        trainVector.append(trainDict[batchId])

    testMaskVector = []
    for userId in userList_test:
        testMaskVector.append(testMaskDict[userId])
#    print("Converting to vectors done....")
    return (torch.Tensor(trainVector)), torch.Tensor(testMaskVector), batchCount

