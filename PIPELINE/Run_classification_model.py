#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:05:28 2022

@author: andrewchan

Note, the only thing you should change is the Aim_num and the trail_num
Aim_num refers to 1: models on balanced dataset, 2: imbalanced dataset
Trail_num: we are running this experiment like 10-20 times for each...we need data!!

Ask Andrew For access to the images. they are in .jpg format!

"""
import sys
import os
path_dir = '/home/andrewchan/Desktop/PIPELINE' #this is the path to the pipeline directory CHANGE ME
data_dir = path_dir + '/splits'
model_dir = path_dir + '/Classification'
aim_num = 1 # CHANGE ME
trial_num = 0 #CHANGE ME
percent = 0 # CHANGE ME FOR AIM 2. If running AIM 1 don't worry about it. This s percent female in the dataset.  

#%% Running the model
sys.path.append(model_dir)
from Classification.Classification_Model import fastai_classification

'''if you want to run one at a time uncomment this line'''
# fastai_classification(path_dir, aim_num, trial_num, percent)

'''if you want to run all of the aim_2 percents from one trial at once, uncomment the for loop''' 
# percents = [0,25,50,75,100]
# for per in percents:
#     fastai_classification(path_dir, aim_num, trial_num, per)


