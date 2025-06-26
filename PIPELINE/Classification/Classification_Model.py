#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:20:12 2022

@author: andrewchan
"""
import fastai
from fastai.data.all import *
from fastai.vision.all import *
import pandas as pd
import numpy as np

#%% takes data from aim 1 splits
def fastai_classification(parent_loc, aim_num, trial_num, percent):
    
    if aim_num == 1:
        train_path =parent_loc+'/splits/Aim_1/trial_'+str(trial_num) +'/train_a1_.csv'
        val_path =parent_loc+'/splits/Aim_1/trial_'+str(trial_num) +'/val_a1_.csv'
        test_path =parent_loc+'/splits/Aim_1/trial_'+str(trial_num) +'/test_a1_.csv'
    if aim_num == 2:
        train_path =parent_loc+'/splits/Aim_2/trial_'+str(trial_num) +'/' + str(percent)+'%F/train_a2_'+str(percent)+'%F_.csv'
        val_path =parent_loc+'/splits/Aim_2/trial_'+str(trial_num) +'/' + str(percent)+'%F/val_a2_'+str(percent)+'%F_.csv'
        test_path =parent_loc+'/splits/Aim_2/trial_'+str(trial_num) +'/' + str(percent)+'%F/test_a2_'+str(percent)+'%F_.csv'
    
    df_train = pd.read_csv(train_path,header = None,usecols=[0,5],names = ['fname','label']).drop_duplicates(subset='fname').fillna('no pneumonia').reset_index(drop = True)
    df_val = pd.read_csv(val_path,header = None,usecols=[0,5],names = ['fname','label']).drop_duplicates(subset='fname').reset_index(drop = True).fillna('no pneumonia')
    df_test = pd.read_csv(test_path,header = None,usecols=[0,5],names = ['fname','label']).drop_duplicates(subset='fname').reset_index(drop = True).fillna('no pneumonia')
    df_total =pd.concat([df_train,df_val],names = ['fname','label'],axis = 0,ignore_index=True)
    val_index = np.arange(len(df_train),len(df_train)+len(df_val))
    splitter = IndexSplitter(val_index)
    
    #%% images into the dataloader
    img_path= '/home/andrewchan/Desktop/rsna-pneumonia-master/data/processed/train_jpg/'
    if(not(os.listdir(img_path)) and not(os.listdir(parent_loc + '/images'))):
        #print('Please put the .jpg images in the images folder in the pipeline. If you don''t know where it is...ask andrew chan')
        return 'Please put the .jpg images in the images folder in the pipeline. If you don''t know where it is...ask andrew chan'
   
    if os.listdir(parent_loc + '/images'):
        img_path = parent_loc + '/images'
    
    pneumonia = DataBlock(blocks = (ImageBlock(cls=PILImageBW ), CategoryBlock),
                       get_x = ColReader(0, pref=img_path),
                       get_y = ColReader(1),
                       batch_tfms = [*aug_transforms(size=224, do_flip = True, max_rotate = 10.0, min_zoom = 0.5, max_zoom = 1.5, max_lighting = 0.3, p_lighting = 0.8, pad_mode = 'zeros', max_warp = 0.2, mult = 2),Normalize.from_stats(*imagenet_stats)],
                       splitter = splitter)
    pneumonia_test = DataBlock(blocks = (ImageBlock(cls=PILImageBW ), CategoryBlock),
                       get_x = ColReader(0, pref=img_path),
                       get_y = ColReader(1),
                       batch_tfms = [*aug_transforms(size=224),Normalize.from_stats(*imagenet_stats)])
    
    dls = pneumonia.dataloaders(df_total,num_workers = 0)
    dls.show_batch(max_n = 9)
    
    #%%  Create Learning and find learning rate
    f1score = F1Score()
    learn = vision_learner(dls, resnet50, metrics=[f1score], pretrained = True)
    # I haven'ty fiddled with optmizers yet #opt = SGD(learn.parameters(), lr = lr)
    print(learn.lr_find()) 
    
    #%% Test out
    lr = 0.00363078061491251 #use the value that was printed from learn.lr_find()
    save_cbs = SaveModelCallback(monitor = 'f1_score')#, fname = 'pneumonia_classification')
    stop_cbs = EarlyStoppingCallback(monitor = 'f1_score', patience = 5)
    call_backs = [save_cbs,stop_cbs]
    learn.fine_tune(40,lr,cbs = call_backs)
    
    #%% Confusion Matrix
    
    # shows confusion matrix (tp,tn,fp,fn) of the VALIDATION set
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    
    # shows examples of missed predictions
    interp.plot_top_losses(4)
    
    #%% save stuff
    import datetime
    today = datetime.datetime.now()
    today = today.strftime("%m"+"_"+ "%d"+"_"+"%y")
    name = 'Classification_A1_'
    if not(os.path.exists(parent_loc+'/Classification/Snapshots/Aim_'+str(aim_num) +'/trial_'+str(trial_num))):
        os.makedirs(parent_loc+'/Classification/Snapshots/Aim_'+str(aim_num) +'/trial_'+str(trial_num))
    
    if aim_num ==1:
        save_path =parent_loc+'/Classification/Snapshots/Aim_'+str(aim_num) +'/trial_'+str(trial_num)+'/classification_a1'
        learn.save(save_path)
    if aim_num ==2:
        save_path =parent_loc+'/Classification/Snapshots/Aim_'+str(aim_num) +'/trial_'+str(trial_num)+'/classification_a2_'+str(percent)+'%F'
        learn.save(save_path)
    #%% make a test Dataloader
    dls_test = pneumonia_test.dataloaders(df_test,num_workers = 0) #load the data
    dls_test.show_batch(max_n = 9)
    
    #%% load the model adn make some inferences. 
    #checkpoint_path = '/home/andrewchan/Desktop/classification/snapshots/'+name +today #07_22_22'
    learn_new = vision_learner(dls, resnet50, metrics=[])
    learn_new.load(save_path)
    
    fpaths = np.array([img_path+img_id for img_id in df_test.fname.values])
    
    test_dl = learn_new.dls.test_dl(fpaths, with_label=True)
    preds, *_,decoded = learn_new.get_preds(dl=test_dl)
    
    #%% Sifting throught the predictions:
    
    df_all_data = pd.read_csv(parent_loc+'/splits/compiled_data.csv')    
    df_preds = pd.DataFrame(columns = ['fname', 'sex', 'pos_pred','neg_pred','ground_truth'])
    
    def get_sex_and_truth(imgId,df = df_all_data):
        row = df_all_data.loc[df_all_data.patientId ==imgId[:-4]].iloc[0]
        ground_truth = 1
        gender = ''
        if np.isnan(row.x):
            ground_truth = 0
        gender = row['Patient Gender']
        return gender, ground_truth
    
    #assuming that the class has to be greater than 0.5
    male_stats = [0,0,0,0]
    female_stats = [0,0,0,0]
    male_gt = list()
    male_pred = list()
    female_gt = list()
    female_pred = list()
    
    
    for prd, fname in zip(preds,df_test.fname.values):
        
        if prd[0].item() > prd[1].item(): 
            dis = 0 # this means yes or no pneumonia
        else: 
            dis = 1
            
        gen, gt = get_sex_and_truth(fname)
        
        if gen == 'F':
            if dis and gt:           female_stats[0]+=1
            if not(dis) and not(gt): female_stats[1]+=1
            if not(dis) and gt:      female_stats[2]+=1
            if dis and not(gt):      female_stats[3]+=1
            female_gt.append(gt)
            female_pred.append(dis)
        else:
            if dis and gt:           male_stats[0]+=1
            if not(dis) and not(gt): male_stats[1]+=1
            if not(dis) and gt:      male_stats[2]+=1
            if dis and not(gt):      male_stats[3]+=1
            male_gt.append(gt)
            male_pred.append(dis)
    
        df_preds.loc[len(df_preds.index)] = [fname,gen,prd[1].item(),prd[0].item(),gt]
    
    #%% Code for 
    from sklearn import metrics
    import matplotlib.pyplot as plt
    
    confusion_matrix_m = metrics.confusion_matrix(male_gt, male_pred)
    cm_display_m = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_m, display_labels = [False, True])
    cm_display_m.plot(cmap = 'GnBu')
    plt.show()
    
    confusion_matrix_f = metrics.confusion_matrix(female_gt, female_pred)
    cm_display_f = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_f, display_labels = [False, True])
    cm_display_f.plot(cmap = 'GnBu')
    plt.show()
    
    #%% save the thing 
    if not(os.path.exists(parent_loc+'/Classification/Analysis/Aim_'+str(aim_num) +'/trial_'+str(trial_num))):
        os.makedirs(parent_loc+'/Classification/Analysis/Aim_'+str(aim_num) +'/trial_'+str(trial_num))
    if aim_num ==1:
        csv_path =parent_loc+'/Classification/Analysis/Aim_'+str(aim_num) +'/trial_'+str(trial_num)+'/TEST_classification_a1.csv'
        df_preds.to_csv(csv_path,index = False)
    if aim_num ==2:
        csv_path =parent_loc+'/Classification/Analysis/Aim_'+str(aim_num) +'/trial_'+str(trial_num)+'/TEST_classification_a2_'+str(percent)+'%F.csv'    
        df_preds.to_csv(csv_path,index = False)









