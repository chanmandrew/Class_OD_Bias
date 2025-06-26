# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:24:50 2022

@author: Chandrew
"""

import fastai
from fastai.data.all import *
from fastai.vision.all import *
import pandas as pd
import numpy as np

#%%
train_path = '/home/andrewchan/Desktop/classification/data_csv/train_a1.csv' #'C:/Users/13019/OneDrive/Desktop/Classification/data_csv/train_a1.csv'
val_path = '/home/andrewchan/Desktop/classification/data_csv/val_a1.csv'
test_path = '/home/andrewchan/Desktop/classification/data_csv/test_a1.csv'
#takes the df from the object detection and makes it into the classification format

df_train = pd.read_csv(train_path,header = None,usecols=[0,5],names = ['fname','label']).drop_duplicates(subset='fname').fillna('no pneumonia').reset_index(drop = True)
df_val = pd.read_csv(val_path,header = None,usecols=[0,5],names = ['fname','label']).drop_duplicates(subset='fname').reset_index(drop = True).fillna('no pneumonia')
df_test = pd.read_csv(test_path,header = None,usecols=[0,5],names = ['fname','label']).drop_duplicates(subset='fname').reset_index(drop = True).fillna('no pneumonia')
df_total =pd.concat([df_train,df_val],names = ['fname','label'],axis = 0,ignore_index=True)
val_index = np.arange(len(df_train),len(df_train)+len(df_val))
splitter = IndexSplitter(val_index)
#test_splitter = RandomSplitter(valid_pct = 0.2)
#%% images into the dataloader

img_path= '/home/andrewchan/Desktop/rsna-pneumonia-master/data/processed/train_jpg/'

pneumonia = DataBlock(blocks = (ImageBlock, CategoryBlock),
                   get_x = ColReader(0, pref=img_path),
                   get_y = ColReader(1),
                   batch_tfms = [*aug_transforms(size=224),Normalize.from_stats(*imagenet_stats)],
                   splitter = splitter)
pneumonia_test = DataBlock(blocks = (ImageBlock, CategoryBlock),
                   get_x = ColReader(0, pref=img_path),
                   get_y = ColReader(1),
                   batch_tfms = [*aug_transforms(size=224),Normalize.from_stats(*imagenet_stats)])

dls = pneumonia.dataloaders(df_total,num_workers = 0)
dls.show_batch(max_n = 9)
#dls_train = pneumonia.dataloaders(df_train,num_workers = 0)
#dls_val = pneumonia.dataloaders(df_val)
#dls_train.show_batch(max_n=9)
#dls_val.show_batch(max_n=9)
#dls = DataLoaders(dls_train,dls_val)

#%%  run the model

learn = vision_learner(dls, resnet50, metrics=accuracy)
#learn.loss_func
#learn.opt_func
print(learn.lr_find()) #this is not workin on my cpu. Might have to run this on the big computer

#%% Test out
#learn.fit_one_cycle(1)
lr = 0.002511886414140463
save_cbs = SaveModelCallback(monitor = 'accuracy', min_delta = 0.1, fname = 'pneumonia_classification')
stop_cbs = EarlyStoppingCallback(monitor = 'accuracy', min_delta = 0.01, patience = 5)
learn.fine_tune(40,base_lr = lr,cbs = [save_cbs,stop_cbs])

#%%
import datetime
today = datetime.datetime.now()
today = today.strftime("%m"+"_"+ "%d"+"_"+"%y")
learn.save('/home/andrewchan/Desktop/classification/snapshots/'+today)

#%%
#tta = learn.tta(use_max=True)
learn.show_results(max_n = 16)


#%% make a test Dataloader
dls_test = pneumonia_test.dataloaders(df_test,num_workers = 0) #load the data
dls_test.show_batch(max_n = 9)

#%% load the model adn make some inferences. 
checkpoint_path = '/home/andrewchan/Desktop/classification/snapshots/07_19_22'
learn_new = vision_learner(dls, resnet50, metrics=accuracy)
learn_new.load(checkpoint_path)

fpaths = np.array([img_path+img_id for img_id in df_test.fname.values])


#test_dl = learn_new.dls.test_dl(dls_test, with_label=True)
test_dl = learn_new.dls.test_dl(fpaths, with_label=True)
preds, *_,decoded = learn_new.get_preds(dl=test_dl)

#%% Sifting throught the predictions:

df_all_data = pd.read_csv('/home/andrewchan/Desktop/compiled_data.csv')    

def get_sex_and_truth(imgId,df = df_all_data):
    row = df_all_data.loc[df_all_data.patientId ==imgId[:-4]].iloc[0]
    ground_truth = True
    gender = ''
    if np.isnan(row.x):
        ground_truth = False
    gender = row['Patient Gender']
    #need to return the gender and the the ground truth
    return gender, ground_truth
#assuming that the class has to be greater than 0.5
#male_stats = [list(),list(),list(),list()] #[tp,tn,fn,fp]
male_stats = [0,0,0,0]
female_stats = [0,0,0,0]


for prd, fname in zip(preds,df_test.fname.values): #we are iterating through fnames
    
    if prd[0].item() > prd[1].item(): 
        dis = False # this means yes or no pneumonia
    else: 
        dis = True
        
    gen, gt = get_sex_and_truth(fname)
    
    if gen == 'F':
        if dis and gt:           female_stats[0]+=1
        if not(dis) and not(gt): female_stats[1]+=1
        if not(dis) and gt:      female_stats[2]+=1
        if dis and not(gt):      female_stats[3]+=1
    else:
        if dis and gt:           male_stats[0]+=1
        if not(dis) and not(gt): male_stats[1]+=1
        if not(dis) and gt:      male_stats[2]+=1
        if dis and not(gt):      male_stats[3]+=1
        

#write for loop to get some shits. 
'''for each thing in test_df, get the gender from the og df...make a function you will be using this more than once. 
as well as the actual prediction and compare it to the normal. Then you can calculate the FP, FN, TN, and TP. You can
also make a PR and a ROC curve from the data shits reeeEEEEEEEEE. That may be a later problem though,. '''
