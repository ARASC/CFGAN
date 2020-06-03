# -*- coding: utf-8 -*-
"""
Author:
    Xuxin Zhang,xuxinz@qq.com
Reference: Chae D K , Kang J S , Kim S W , et al.
CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks[C]// the 27th ACM International Conference. ACM, 2018.

Baseline: https://github.com/1051003502/CFGAN
"""

import random
import re
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import preprocessing

import data
import cfgan

import warnings
warnings.filterwarnings("ignore")


def UseInfoPreprocessing(UseInfo):
    
    useGender_dummies = pd.get_dummies(UseInfo['useGender'])
    UseInfo = UseInfo.join(useGender_dummies)
    UseInfo.drop(['useGender'], axis=1, inplace=True)

    # create feature for the alphabetical part of the Occupation
    UseInfo['useOccupationLetter'] = UseInfo['useOccupation'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    # convert the distinct Occupation letters with incremental integer values
    UseInfo['useOccupationLetter'] = pd.factorize(UseInfo['useOccupationLetter'])[0]
    UseInfo[['useOccupation','useOccupationLetter']].head()
    UseInfo.drop(['useOccupation'], axis=1, inplace=True)
    
    UseInfo['useZipcodeLetter'] = UseInfo['useZipcode'].str.split().str[0]
    UseInfo['useZipcodeLetter'] = UseInfo['useZipcodeLetter'].apply(lambda x: "99999" if not(x.isnumeric()) else x)
    UseInfo['useZipcodeLetter'] = pd.factorize(UseInfo['useZipcodeLetter'])[0]
    UseInfo[['useZipcode','useZipcodeLetter']].head()
    UseInfo.drop(['useZipcode'], axis=1, inplace=True)

    # StandardScaler will subtract the mean from each value then scale to the unit varience
    scaler = preprocessing.StandardScaler()
    UseInfo['useAge_scaled'] = scaler.fit_transform(UseInfo['useAge'].values.reshape(-1,1))
    UseInfo.drop(['useAge'], axis=1, inplace=True)
    
    return UseInfo


def select_negative_items(realData, num_pm, num_zr):
    '''
    realData : n-dimensional indicator vector specifying whether u has purchased each item i
    num_pm : num of negative items (partial-masking) sampled on the t-th iteration
    num_zr : num of negative items (zeroreconstruction regularization) sampled on the t-th iteration
    '''
    data = np.array(realData)
    n_items_pm = np.zeros_like(data)
    n_items_zr = np.zeros_like(data)
    for i in range(data.shape[0]):
        p_items = np.where(data[i] != 0)[0]
        all_item_index = random.sample(range(data.shape[1]), 1683)
        for j in range(p_items.shape[0]):   
            all_item_index.remove(list(p_items)[j])
        random.shuffle(all_item_index)
        n_item_index_pm = all_item_index[0 : num_pm]
        n_item_index_zr = all_item_index[num_pm : (num_pm+num_zr)]
        n_items_pm[i][n_item_index_pm] = 1
        n_items_zr[i][n_item_index_zr] = 1
    
    return n_items_pm, n_items_zr


def computeTopN(groundTruth,result,topN):
    result=result.tolist()
    for i in range(len(result)):
        result[i]=(result[i],i)
    result.sort(key=lambda x:x[0],reverse=True)
    hit=0
    for i in range(topN):
        if(result[i][1] in groundTruth):
            hit=hit+1    
    return hit/topN


def main(userCount,itemCount,testSet,trainVector,trainMaskVector,\
         UseInfo_pre,topN,epochCount,pro_ZR,pro_PM,alpha):

    info_shape = UseInfo_pre.shape[1]
    UseInfo_pre = UseInfo_pre.values
    UseInfo_pre = np.insert(UseInfo_pre,0,[0,0,0,0,0],axis=0)
    UseInfo_pre = torch.tensor(UseInfo_pre.astype(np.float32))
    result_precision=np.zeros((1,2))
    
    # Build the generator and discriminator
    G=cfgan.generator(itemCount, info_shape)
    D=cfgan.discriminator(itemCount, info_shape)
    regularization = nn.MSELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
    G_step=5
    D_step=2
    batchSize_G = 32
    batchSize_D = 32
    
    for epoch in range(epochCount): 
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        for step in range(G_step):
            
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, userCount - batchSize_G - 1)
            realData = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            eu = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            useInfo_batch = Variable(copy.deepcopy(UseInfo_pre[leftIndex:leftIndex + batchSize_G]))
            
            # Select a random batch of negative items for every user
            n_items_pm,n_items_zr = select_negative_items(realData,pro_PM,pro_ZR)           
            ku_zp = Variable(torch.tensor(n_items_pm + n_items_zr))
            realData_zp = Variable(torch.ones_like(realData)) * eu + Variable(torch.zeros_like(realData)) * ku_zp
            
            # Generate a batch of new purchased vector
            fakeData=G(realData,useInfo_batch) 
            fakeData_ZP = fakeData * (eu + ku_zp)  
            fakeData_result=D(fakeData_ZP,useInfo_batch) 
            
            # Train the discriminator
            g_loss = np.mean(np.log(1.-fakeData_result.detach().numpy()+10e-5))  + alpha*regularization(fakeData_ZP,realData_zp)
            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()
            
        # ---------------------
        #  Train Discriminator
        # ---------------------
        for step in range(D_step):
            # Select a random batch of purchased vector
            leftIndex=random.randint(1,userCount-batchSize_D-1)
            realData=Variable(copy.deepcopy(trainVector[leftIndex:leftIndex+batchSize_D])) 
            eu = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            useInfo_batch = Variable(copy.deepcopy(UseInfo_pre[leftIndex:leftIndex + batchSize_G]))
            
            # Select a random batch of negative items for every user
            n_items_pm, _ = select_negative_items(realData,pro_PM,pro_ZR)
            ku = Variable(torch.tensor(n_items_pm))
            
            # Generate a batch of new purchased vector
            fakeData=G(realData,useInfo_batch) 
            fakeData_ZP = fakeData * (eu + ku)
            
            # Train the discriminator
            fakeData_result=D(fakeData_ZP,useInfo_batch) 
            realData_result=D(realData,useInfo_batch) 
            d_loss = -np.mean(np.log(realData_result.detach().numpy()+10e-5) + 
                              np.log(1. - fakeData_result.detach().numpy()+10e-5)) + 0*regularization(fakeData_ZP,realData_zp)
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()
            
        if( epoch%1==0):
            n_user=len(testSet)
            
            index=0
            precisions=0
            for testUser in testSet.keys():

                data = Variable(copy.deepcopy(trainVector[testUser]))
                useInfo_index = Variable(copy.deepcopy(torch.tensor(np.expand_dims(UseInfo_pre[index], axis=0))))
                #  Exclude the purchased vector that have occurred in the training set
                result = G(data.reshape(1,1683),useInfo_index) + Variable(copy.deepcopy(trainMaskVector[index]))
                result = result.reshape(1683)
                
                precision = computeTopN(testSet[testUser], result, topN)
                precisions+=precision
                index+=1

            precisions = precisions/n_user
            result_precision=np.concatenate((result_precision,np.array([[epoch,precisions]])),axis = 0)

            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},precision:{}'.format(epoch, epochCount,
            d_loss.item(),
            g_loss.item(),
            precisions))

    return result_precision


def result_plt(result_precision):
    plt.figure()
    plt.title("the precision of CFGAN")
    plt.xlabel('epoch')
    plt.plot(result_precision[:,0], result_precision[:,1], "r-*",label='precision' )
    plt.ylim([0, 0.6])
    plt.legend()  
    plt.show()
    
if __name__ == '__main__':
    
    topN=5
    epochs = 1000
    pro_ZR = 50
    pro_PM = 50
    alpha = 0.1
    UseInfo = data.loadUseInfo("data/ml-100k/u.user"  , "|")
#    ItemInfo = data.loadItemInfo("data/ml-100k/u.item"  , "|")
    UseInfo_pre = UseInfoPreprocessing(UseInfo)
    UseInfo_pre.drop(['userId'], axis=1, inplace=True)
    trainSet,train_use,train_item= data.loadTrainingData("data/ml-100k/u1.base", "\t")
    testSet,test_use,test_item = data.loadTestData("data/ml-100k/u1.test", "\t")
    userCount = max(train_use,test_use)
    itemCount = max(train_item,test_item)
    userList_test = list(testSet.keys())
    trainVector, trainMaskVector, batchCount = data.to_Vectors(trainSet, userCount, \
                                                              itemCount, userList_test, "userBased")

    result_precision = main(userCount,itemCount,testSet,\
                            trainVector,trainMaskVector,UseInfo_pre,topN,epochs,pro_ZR,pro_PM,alpha)
    result_precision = result_precision[1:,]
    result_plt(result_precision)


    
    