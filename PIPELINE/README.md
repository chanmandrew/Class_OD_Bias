# This is the pipline for Andrew's Project comparing Classification models and Object detection models on pneumonia detection
Read the abstract

## SETUP
You will need to do a few things before you can run the code. 
1. You will need to download this repository (PIPELINE)
2. You will need to download the images (ask andrew for a onedrive link)
3. Download and put the images into the images folder
4. Download anaconda-navigator. See the lab's tutorial for how to set that up. 
https://docs.google.com/document/d/1hHH7-mz8qTNm8p6PsZVgpbPMWNaK7WO28sCZMG5KVTM/edit?usp=sharing
5. Make two environments:
#### a. Fastai Classifcation: 
	- conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
	- pip install fastai
#### b. Object detection icevision:
	- conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
	- pip install icevision[all]
 6. **NOTE** When you run the object detection model, you have to be in the _icevision_ environment whereas if you run the Classification model, you will have to be in the _fastai_ environemnt

## Run Code

### Make datasets
- When you run the code, you will have to first make the datasets for both Aim 1 and Aim 2
- We will be running multiple trials for each aim (maybe 10 - 20 depending)
- To make the datasplits, run Make_datasets.py
  - You can either make one dataset at a time by changing the `aim_num` (aim 1 or aim 2) and `trial_num` for the trial that you are running. 
  - Alternatively, you can uncomment the bottom section (do it all at once) and run that to make your datasets. 
  - Note that this will not remake a dataset should one already exist. 
  - if for whatever reason you would like to look at the csv files, you can find them under PIPELINE -> splits -> Aim_# -> trial_# 

### For AIM 1 
- (comparing classifiers and object detectors on given data) you will get 3 csv files (train/validation/test datasets)
- You will find these csv files under splits -> Aim_1 -> trial_#
- You will train both the object detector and the classifier on these models. 
- They will output both a .pth file and a .csv file. Csv file will be used later for analysis/ROC curves

### For AIM 2
- (comparing classifiers and object detectors on different splits of training data) we will create folders containing x% of females (e.g. 100%F would be a csv with only female images and 0% only male images)
- You will find these csv files under splits -> Aim_2 -> trial_# _> #%F
- You will train both the object detector and the classifier on these models. 
- For each trial, you will end up with 5 .pth files and 5 .csv files. Csv file will be used later for analysis/ROC curves

If you would like to access either the prediction csv files or the models, you can find them under classificaiton/object detection -> Analysis/Snapshots -> aim_# -> trial_# respectively

### Files that you will be mostly working with: 

#### Make_datasplits.py
-  This is where you will make the datasplits
-  **You will need to update** `path_dir` to the path where the pipeline is stored
-  If you want to create datasets **one at a time**, you will haev to change `aim_num` and `trial_num` one at a time and uncomment the first block of code
-  If you want to create them **all at once**, uncomment the second block of code. 
-  Note these take  ~3 min to create an Aim 1 dataset and ~4 min to make an aim 2 dataset. in total, It took about 7.5 min to make both. If you want to make them all at once, it will take around 2.5 hours...run it the background!  

#### Run_classification_model.py
- This is where you will train and get predictions for classification model.
- **You will need to update** `path_dir` to the path where the pipeline is stored
- you will have to change `aim_num` and `trial_num` as before. 
- if you are running aim_2, you will also have to change `percent` for the percent of interest.  
- This can take like ~20+ min to train one model, so keep this in mind.
- Should you want to play around with the clasification model, you can find it under Classification -> Classification_Model.py

#### Run_object_detection_model.py
- This is where you will train and get predictions for object_detection model.
- **You will need to update** `path_dir` to the path where the pipeline is stored
- you will have to change `aim_num` and `trial_num` as before. 
- if you are running aim_2, you will also have to change `percent` for the percent of interest.  
- This can take like ~40+ min to train one model, so keep this in mind.
- Should you want to play around with the object detection model, you can find it under Object_detection -> Object_detection_model.py




