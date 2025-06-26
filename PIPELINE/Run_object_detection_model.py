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
path_dir = '/home/andrewchan/Desktop/PIPELINE' #this is the path to the pipeline directory
data_dir = path_dir + '/splits'
model_dir = path_dir + '/Object_detection'
aim_num = 1
trial_num = 0
percent = 75 #percent female
#%% Running the model
sys.path.append(model_dir)
from Object_detection.Object_detection_model import icevision_object_detection

'''if you want to run one at a time uncomment this line'''
# icevision_object_detection(path_dir, aim_num, trial_num, percent)

'''if you want to run all of the aim_2 percents from one trial at once, uncomment the for loop. This will take a while!'''
# percents = [0,25,50,75,100]
# for per in percents:
#     fastai_classification(path_dir, aim_num, trial_num, per)