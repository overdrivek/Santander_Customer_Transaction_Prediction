# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:58:08 2018

@author: naraya01
"""

import pandas as pd
import os

train_file = '/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/train.csv'

#pd_train = pd.read_csv(train_file,sep=',',header=None)
pd_train = pd.read_csv(train_file,sep=',')

#input_train = pd_train.iloc[:,0:pd_train.shape[1]-1]
input_train = pd_train.drop('target',axis=1)
print(input_train.shape)
label_train = pd_train['target']

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(input_train,label_train,test_size=0.1,shuffle=True,stratify=label_train)

num_classes = 2
import numpy as np

readme_folder = os.path.split(train_file)[0]
training_folder = 'training_files'
train_files_folder= os.path.join(os.path.split(train_file)[0],training_folder)
if os.path.isdir(train_files_folder) is False:
    os.mkdir(train_files_folder)

with open(os.path.join(readme_folder,training_folder,'readme.txt'),'w') as f_readme:
    for i in range(num_classes):
        f_readme.writelines("number of class {} in training = {}\n".format(i,len(np.where(Y_train==i)[0])))
        f_readme.writelines("number of class {} in test = {}\n".format(i,len(np.where(Y_test==i)[0])))
        print("number of class {} in training = {}\n".format(i,len(np.where(Y_train==i)[0])))
        print("number of class {} in test = {}\n".format(i,len(np.where(Y_test==i)[0])))


df_full = pd.concat([X_train,Y_train],axis=1)

train_filepath = os.path.join(train_files_folder,'train_set.csv')
test_filepath = os.path.join(train_files_folder,'validation_set.csv')

df_full.to_csv(train_filepath,header=True,index=False,sep=',',mode='w')


df_test = pd.concat([X_test,Y_test],axis=1)
df_test.to_csv(test_filepath,header=True,index=False,sep=',',mode='w')


