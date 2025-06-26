#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:18:21 2022

@author: andrewchan
"""

import sys
import os
path_dir = '/home/andrewchan/Desktop/PIPELINE' #this is the path to where the pipline is on your computer
data_dir = path_dir + '/splits'
model_dir = path_dir + '/Classification'
aim_num = 2
trial_num = 0

#%% Making data sets

'''if you want to make a dataset one at a time, uncomment this and adjust the aim_num and the trial_num accordingly'''
os.chdir(path_dir)
sys.path.append(data_dir)
from splits.Aim_2_Final_splits import make_data_splits_2
from splits.Aim_1_splits import make_data_splits_1

# if aim_num ==1:
#     make_data_splits_1(data_dir,trial_num)
# if aim_num == 2:
#     make_data_splits_2(data_dir,trial_num)

#%% do it all at once
'''if you want to make all the datasets at once, you can uncomment this for loop and run'''
for trial in range(0,20):
    make_data_splits_1(data_dir,trial)
    make_data_splits_2(data_dir,trial)
    print('trial '+str(trial) + 'finished')
