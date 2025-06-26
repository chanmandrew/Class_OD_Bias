#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:09:04 2022

@author: andrewchan
"""

import icevision
from icevision.all import*
import pandas as pd
from icevision.models import *
from fastai.callback.tracker import *
from pathlib import Path
import datetime

def icevision_object_detection(parent_loc, aim_num, trial_num, percent):
    #name = 'Object_detection_A1_'
    today = datetime.datetime.now()
    today = today.strftime("%m"+"_"+ "%d"+"_"+"%y")
    #data dir has all the images that I will train with. 
    #data_dir = '/home/andrewchan/Desktop/rsna-pneumonia-master/data/processed/train_jpg/'
    #%% images into the dataloader
    data_dir= '/home/andrewchan/Desktop/rsna-pneumonia-master/data/processed/train_jpg/'
    if(not(os.listdir(data_dir)) and not(os.listdir(parent_loc + '/images'))):
        return 'Please put the .jpg images in the images folder in the pipeline. If you don''t know where it is...ask andrew chan'
       
    if os.listdir(parent_loc + '/images'):
        data_dir = parent_loc + '/images'
    #%% Getting data together from csv files
    if aim_num == 1:
        tr =parent_loc+'/splits/Aim_1/trial_'+str(trial_num) +'/train_a1_.csv'
        val =parent_loc+'/splits/Aim_1/trial_'+str(trial_num) +'/val_a1_.csv'
        tes =parent_loc+'/splits/Aim_1/trial_'+str(trial_num) +'/test_a1_.csv'
    if aim_num == 2:
        tr =parent_loc+'/splits/Aim_2/trial_'+str(trial_num) +'/' + str(percent)+'%F/train_a2_'+str(percent)+'%F_.csv'
        val =parent_loc+'/splits/Aim_2/trial_'+str(trial_num) +'/' + str(percent)+'%F/val_a2_'+str(percent)+'%F_.csv'
        tes =parent_loc+'/splits/Aim_2/trial_'+str(trial_num) +'/' + str(percent)+'%F/test_a2_'+str(percent)+'%F_.csv'
    
    
    # tr ='/home/andrewchan/Desktop/Aim_1_splits/train_a1.csv'
    # val = '/home/andrewchan/Desktop/Aim_1_splits/val_a1.csv'
    # tes = '/home/andrewchan/Desktop/Aim_1_splits/test_a1.csv'
    pos_col = ['fname','xmin','ymin','xmax','ymax','label']
    neg_col = ['fname']
    dftr = pd.read_csv(tr,names = pos_col, header = None)
    df_trp = dftr.loc[dftr.label == 'pneumonia'].reset_index(drop = True)
    df_trn = dftr[dftr.label.isnull()].reset_index(drop = True).drop(pos_col[1:],axis =1)
    dfva = pd.read_csv(val,names = pos_col, header = None)
    df_valp = dfva.loc[dfva.label == 'pneumonia'].reset_index(drop = True)
    df_valn = dfva[dfva.label.isnull()].reset_index(drop = True).drop(pos_col[1:],axis =1)
    df_all_pos = pd.concat([df_trp,df_valp],axis = 0)
    df_all_neg = pd.concat([df_trn,df_valn],axis = 0)
    #%% Creating parsers
    template_record = ObjectDetectionRecord()
    
    #%% Positive image parser 
    class PneumoniaParser(Parser):
        def __init__(self, template_record, data_dir,df_pos,idmap = None):
            super().__init__(template_record=template_record)
            self.data_dir = data_dir
            self.df = df_pos #pd.read_csv(new_label)
            self.class_map = ClassMap(list(self.df['label'].unique()))
            self.idmap = idmap
            
        def __iter__(self) -> Any:
            for o in self.df.itertuples():
                yield o
                
        def __len__(self) -> int:
            return len(self.df)
        
        def record_id(self, o: Any) -> Hashable:
            return o.fname
        
        def parse_fields(self, o: Any, record: BaseRecord, is_new: bool):
            if is_new:
                record.set_filepath(self.data_dir + o.fname)
                record.set_img_size(ImgSize(width = 1024, height = 1024))
                record.detection.set_class_map(self.class_map)
            record.detection.add_labels([o.label])
            record.detection.add_bboxes([BBox.from_xyxy(o.xmin, o.ymin, o.xmax, o.ymax)])
            
    #%% Negative image parser
    class NoPneumoniaParser(Parser):
        def __init__(self, template_record, data_dir,df_neg, idmap = None):
            super().__init__(template_record=template_record)
            self.data_dir = data_dir
            self.df = df_neg #pd.read_csv(new_label) 
            #self.class_map = ClassMap(list(self.df['label'].unique()))
            self.idmap = idmap
            
        def __iter__(self) -> Any:
            for o in self.df.itertuples():
                yield o
                
        def __len__(self) -> int:
            return len(self.df)
        
        def record_id(self, o: Any) -> Hashable:
            return o.fname
        
        def parse_fields(self, o: Any, record: BaseRecord, is_new: bool):
            if is_new:
                record.set_filepath(self.data_dir + o.fname)
                record.set_img_size(ImgSize(width = 1024, height = 1024))
                #record.detection.set_class_map(self.class_map)
            
    #%% getting the data into a format for the parsers.
    
    posIdMap = IDMap(df_trp.fname.to_list() + df_valp.fname.to_list())
    negIdMap = IDMap(df_trn.fname.to_list() + df_valn.fname.to_list())
    
    pos_splits = [list(set(df_trp.fname.to_list())),list(set(df_valp.fname.to_list()))]
    neg_splits = [list(set(df_trn.fname.to_list())),list(set(df_valn.fname.to_list()))]
    
    data_splitter_pos = FixedSplitter(pos_splits)
    data_splitter_neg = FixedSplitter(neg_splits)
    
    #%% creating parsers
    pos_parser = PneumoniaParser(template_record, data_dir,df_all_pos)
    print(pos_parser.class_map)
    neg_parser = NoPneumoniaParser(template_record,data_dir,df_all_neg)
    print(neg_parser.df.head())
    
    #%% create the different splits with the parser
    tr_records_pos, val_records_pos = pos_parser.parse(data_splitter_pos)
    tr_records_neg, val_records_neg = neg_parser.parse(data_splitter_neg)
    
    #%% testing parser and making sure everything displays correctly
    show_record(tr_records_pos[15], display_label = True, figsize = (14,10))
    show_record(val_records_pos[15], display_label = True, figsize = (14,10))
    
    show_records(random.choices(tr_records_neg, k=4), ncols=2)
    show_records(random.choices(val_records_neg, k=4), ncols=2)
    
    #%% Transforms
    
    image_size = 384
    train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=512), tfms.A.Normalize()])
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
    
    #%% model selection
    
    # Just change the value of selection to try another model
    selection = 3
    
    extra_args = {}
    
    if selection == 0:
      model_type = models.mmdet.vfnet
      backbone = model_type.backbones.resnet50_fpn_mstrain_2x
    
    if selection == 1:
      model_type = models.mmdet.retinanet
      backbone = model_type.backbones.resnet50_fpn_1x
    
    elif selection == 2:
      model_type = models.mmdet.faster_rcnn
      backbone = model_type.backbones.resnet50_fpn_1x
      # extra_args['cfg_options'] = { 
      #   'model.bbox_head.loss_bbox.loss_weight': 2,
      #   'model.bbox_head.loss_cls.loss_weight': 0.8,
      #    }
    
    elif selection == 3:
      # The Retinanet model is also implemented in the torchvision library
      model_type = models.torchvision.retinanet
      backbone = model_type.backbones.resnet50_fpn
    
    elif selection == 4:
      model_type = models.ross.efficientdet
      backbone = model_type.backbones.tf_lite0
      # The efficientdet model requires an img_size parameter
      extra_args['img_size'] = image_size
    
    elif selection == 5:
      model_type = models.ultralytics.yolov5
      backbone = model_type.backbones.small
      # The yolov5 model requires an img_size parameter
      extra_args['img_size'] = image_size
    
    model_type, backbone, extra_args
    
    #%% combining the train + and - together as well as the val + and - together
    batch_size = 8
    
    train_ds = Dataset(tr_records_pos+tr_records_neg,train_tfms)
    val_ds = Dataset(val_records_pos+val_records_neg,valid_tfms)
    
    combined_train_dl = model_type.train_dl(train_ds, batch_size=batch_size, num_workers=4, shuffle=True)
    combined_valid_dl = model_type.valid_dl(val_ds, batch_size=batch_size, num_workers=4, shuffle=False)
    
    
    #%% making sure that the negative and the postve images are together. 
    show_samples(random.choices(train_ds, k=6), class_map=pos_parser.class_map, ncols=3)
    show_samples(random.choices(val_ds, k=6), class_map=pos_parser.class_map, ncols=3)
    
    #%% making the model
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    
    combined_model = model_type.model(
        backbone = backbone(pretrained=True), 
        num_classes=len(pos_parser.class_map),**extra_args)
    
    combined_learn = model_type.fastai.learner(
        dls=[combined_train_dl, combined_valid_dl], 
        model=combined_model, 
        metrics=metrics)
    
    print(combined_learn.lr_find())
    
    #%% train the model
    lr = 0.00010964782268274575 #learning rate found from above
    save_cbs = SaveModelCallback(monitor = 'COCOMetric')#, fname = 'pneumonia_bestCOCO_'+today)
    stop_cbs = EarlyStoppingCallback(monitor = 'COCOMetric', patience = 5)
    combined_learn.fine_tune(1, lr, freeze_epochs=1,cbs = [save_cbs,stop_cbs])
    
    #%% show results
    model_type.show_results(combined_model, val_ds, detection_threshold=.2,num_samples = 1)
    
    
    #%% Save model 
    
    # #learner
    #combined_learn.save('/home/andrewchan/Desktop/ice_vision/snapshots/learner_'+name+today)
    
    # # #icevision save
    from icevision.models.checkpoint import save_icevision_checkpoint
    if not(os.path.exists(parent_loc+'/Object_detection/Snapshots/Aim_'+str(aim_num) +'/trial_'+str(trial_num))):
        os.makedirs(parent_loc+'/Object_detection/Snapshots/Aim_'+str(aim_num) +'/trial_'+str(trial_num))
    if aim_num ==1:
        save_path =parent_loc+'/Object_detection/Snapshots/Aim_'+str(aim_num) +'/trial_'+str(trial_num)+'/classification_a1.pth'      
    if aim_num ==2:
        save_path =parent_loc+'/Object_detection/Snapshots/Aim_'+str(aim_num) +'/trial_'+str(trial_num)+'/classification_a2_'+str(percent)+'%F.pth'
    #checkpoint_path = '/home/andrewchan/Desktop/ice_vision/snapshots/icevision_'+name+today+'.pth'
    save_icevision_checkpoint(combined_model, 
                           model_name='torchvision.retinanet', 
                           backbone_name='resnet50_fpn',
                           classes =  pos_parser.class_map.get_classes(), 
                           img_size=image_size, 
                           filename=save_path,
                           meta={'icevision_version': '0.12.0'})
    
    #%% making a test dataloader
    
    dfts = pd.read_csv(tes,names = pos_col, header = None)
    df_tsp = dfts.loc[dfts.label == 'pneumonia'].reset_index(drop = True)
    df_tsn = dfts[dfts.label.isnull()].reset_index(drop = True).drop(pos_col[1:],axis =1)
    posIdMap_test = IDMap(list(set(df_tsp.fname.to_list())))
    negIdMap_test = IDMap(list(set(df_tsn.fname.to_list())))
    
    pos_parser_test = PneumoniaParser(template_record, data_dir,df_tsp)
    neg_parser_test = NoPneumoniaParser(template_record, data_dir,df_tsn)
    
    ts_records_pos, *_ = pos_parser_test.parse(data_splitter = SingleSplitSplitter())
    ts_records_neg, *_ = neg_parser_test.parse(data_splitter = SingleSplitSplitter())
    
    test_ds = Dataset(ts_records_pos+ts_records_neg,valid_tfms)
    test_dl = model_type.infer_dl(test_ds, batch_size=batch_size, num_workers=4, shuffle=False)
    
    
    #%% Load stuff
    
    #checkpoint_path = '/home/andrewchan/Desktop/ice_vision/snapshots/icevision_Object_detection_A1_07_27_22.pth'
    from icevision.models import checkpoint
    checkpoint_and_model = checkpoint.model_from_checkpoint(save_path, 
        model_name='torchvision.retinanet', 
        backbone_name='resnet50_fpn',
        img_size=384, 
        is_coco=False)
    
    model_type = checkpoint_and_model["model_type"]
    backbone = checkpoint_and_model["backbone"]
    class_map = checkpoint_and_model["class_map"]
    img_size = checkpoint_and_model["img_size"]
    
    model = checkpoint_and_model["model"]
    model = model.cuda()
    
    # save_cbs = SaveModelCallback(monitor = 'COCOMetric', fname = 'pneumonia_bestCOCO')
    # stop_cbs = EarlyStoppingCallback(monitor = 'COCOMetric', min_delta = 0.01, patience = 5)
    # learn_from_checkpoint = model_type.fastai.learner(
    #     dls=[combined_train_dl, combined_valid_dl], 
    #     model=model, 
    #     metrics=metrics,
    #     cbs = [save_cbs,stop_cbs])
    # learn_from_checkpoint.path = Path('/home/andrewchan/Desktop/ice_vision/snapshots')
    # learn_from_checkpoint.model_dir = Path(today)
    # learn_from_checkpoint.fine_tune(40, lr, freeze_epochs=1)
    
    
    #%% look at images from loaded model
    
    preds = model_type.predict_from_dl(model, test_dl, keep_images=True)
    show_preds(preds=preds[28:29])
    
    #%% making csv for stuff. 
    
    df_all_data = pd.read_csv(parent_loc+'/splits/compiled_data.csv')    
    def get_sex_and_truth(imgId,df = df_all_data):
        row = df_all_data.loc[df_all_data.patientId ==imgId[:-4]].iloc[0]
        ground_truth = 1
        gender = ''
        if np.isnan(row.x):
            ground_truth = 0
        gender = row['Patient Gender']
        #need to return the gender and the the ground truth
        return gender, ground_truth
    
    deads = model_type.predict_from_dl(model, test_dl, keep_images=True, detection_threshold = 0)
    df_predz = pd.DataFrame(columns = ['fname', 'sex', 'ground_truth','confidence'])
    
    for thing in deads:
        
        #gt = thing.ground_truth.as_dict()
        if(thing.ground_truth.as_dict()['detection']['bboxes']):
            truth = 1
        else:
            truth = 0
        fname = thing.ground_truth.as_dict()['common']['record_id']
        if list(thing.pred.as_dict()['detection']['scores']):
            confidence = max(thing.pred.as_dict()['detection']['scores'])
        else:
            confidence = 0
        gen, gt = get_sex_and_truth(fname)
        assert gt ==truth
        
        df_predz.loc[len(df_predz.index)] = [fname,gen,truth,confidence]
    
    if not(os.path.exists(parent_loc+'/Object_detection/Analysis/Aim_'+str(aim_num) +'/trial_'+str(trial_num))):
        os.makedirs(parent_loc+'/Object_detection/Analysis/Aim_'+str(aim_num) +'/trial_'+str(trial_num))
    if aim_num ==1:
        csv_path =parent_loc+'/Object_detection/Analysis/Aim_'+str(aim_num) +'/trial_'+str(trial_num)+'/TEST_classification_a1.csv'
        df_predz.to_csv(csv_path,index = False)
    if aim_num ==2:
        csv_path =parent_loc+'/Object_detection/Analysis/Aim_'+str(aim_num) +'/trial_'+str(trial_num)+'/TEST_classification_a2_'+str(percent)+'%F.csv'    
        df_predz.to_csv(csv_path,index = False)
    
    #df_predz.to_csv('/home/andrewchan/Desktop/Analysis/TEST_DATA_object_detection_'+name+today+'.csv',index = False)

