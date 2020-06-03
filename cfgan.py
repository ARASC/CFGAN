# -*- coding: utf-8 -*-
"""
Author:
    Xuxin Zhang,xuxinz@qq.com
Reference: Chae D K , Kang J S , Kim S W , et al.
CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks[C]// the 27th ACM International Conference. ACM, 2018.

"""
import torch
import torch.nn as nn
 
class discriminator(nn.Module):
    def __init__(self,itemCount,info_shape):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(itemCount+info_shape,1024),
            nn.ReLU(True),
            nn.Linear(1024,128),
            nn.ReLU(True),
            nn.Linear(128,16),
            nn.ReLU(True),
            nn.Linear(16,1),
            nn.Sigmoid()
        )
    def forward(self,data,condition):
        data_c  = torch.cat((data,condition),1)
        result=self.dis( data_c )
        return result  
    
class generator(nn.Module):
    def __init__(self,itemCount,info_shape):
        self.itemCount = itemCount
        super(generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(self.itemCount+info_shape, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512,1024),
            nn.ReLU(True),
            nn.Linear(1024, itemCount),
            nn.Sigmoid()
        )
    def forward(self,noise,useInfo):
        G_input = torch.cat([noise, useInfo], 1)
        result=self.gen(G_input)
        return result   