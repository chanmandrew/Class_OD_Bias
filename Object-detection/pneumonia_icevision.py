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

#data dir has all the images that I will train with. 
data_dir = '/home/andrewchan/Desktop/rsna-pneumonia-master/data/processed/train_jpg/'

#%% Getting data together from csv files
tr_pos = '/home/andrewchan/Desktop/ice_vision/data_v2/train_a1_pos.csv'
tr_neg = '/home/andrewchan/Desktop/ice_vision/data_v2/train_a1_neg.csv'
val_pos = '/home/andrewchan/Desktop/ice_vision/data_v2/val_a1_pos.csv'
val_neg = '/home/andrewchan/Desktop/ice_vision/data_v2/val_a1_neg.csv'

pos_col = ['fname','xmin','ymin','xmax','ymax','label']
neg_col = ['fname']


df_trp = pd.read_csv(tr_pos,names = pos_col, header = None)
df_trn = pd.read_csv(tr_neg,names = neg_col, header = None)
df_valp = pd.read_csv(val_pos,names = pos_col, header = None)
df_valn = pd.read_csv(val_neg,names = neg_col, header = None)

df_all_pos = pd.concat([df_trp,df_valp],axis = 0)
df_all_neg = pd.concat([df_trn,df_valn],axis = 0)
#%% Creating parsers
template_record = ObjectDetectionRecord()


#%% Positive image parser 
class PneumoniaParser(Parser):
    def __init__(self, template_record, data_dir):
        super().__init__(template_record=template_record)
        self.data_dir = data_dir
        self.df = df_all_pos #pd.read_csv(new_label)
        self.class_map = ClassMap(list(self.df['label'].unique()))
        
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
    def __init__(self, template_record, data_dir,idmap = None):
        super().__init__(template_record=template_record)
        self.data_dir = data_dir
        self.df = df_all_neg #pd.read_csv(new_label) 
        #self.class_map = ClassMap(list(self.df['label'].unique()))
        
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
pos_parser = PneumoniaParser(template_record, data_dir)
print(pos_parser.class_map)
neg_parser = NoPneumoniaParser(template_record,data_dir)
print(neg_parser.df.head())

#%% create the different splits with the parser
tr_records_pos, val_records_pos = pos_parser.parse(data_splitter_pos)
tr_records_neg, val_records_neg = neg_parser.parse(data_splitter_neg)

#%% testing parser and making sure everything displays correctly
show_record(tr_records_pos[15], display_label = True, figsize = (14,10))
show_record(val_records_pos[15], display_label = True, figsize = (14,10))

show_records(random.choices(tr_records_neg, k=2), ncols=2)
show_records(random.choices(val_records_neg, k=2), ncols=2)

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
val_ds = Dataset(val_records_pos+val_records_neg,train_tfms)

combined_train_dl = model_type.train_dl(train_ds, batch_size=batch_size, num_workers=4, shuffle=True)
combined_valid_dl = model_type.valid_dl(val_ds, batch_size=batch_size, num_workers=4, shuffle=False)

#%% making sure that the negative and the postve images are together. 
show_samples(random.choices(train_ds, k=6), class_map=class_map, ncols=3)
show_samples(random.choices(val_ds, k=6), class_map=class_map, ncols=3)

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
lr = 0.0002290867705596611 #4.365158383734524e-05 #1e-3
#combined_learn.fine_tune(40, lr, freeze_epochs=1)

#%% show results
model_type.show_results(combined_model, val_ds, detection_threshold=.5,num_samples = 3)

#%% Save model 

#learner
combined_learn.save('/home/andrewchan/Desktop/ice_vision/snapshots/pneumonia_icevision_learner')
combined_learn.export('/home/andrewchan/Desktop/ice_vision/snapshots/pneumonia_icevision_full_model.pkl')

#icevision save
from icevision.models.checkpoint import save_icevision_checkpoint
checkpoint_path = '/home/andrewchan/Desktop/ice_vision/snapshots/pneumonia_icevision_model_checkpoint.pth'
save_icevision_checkpoint(combined_model, 
                        model_name='torchvision.retinanet', 
                        backbone_name='resnet50_fpn',
                        classes =  pos_parser.class_map.get_classes(), 
                        img_size=image_size, 
                        filename=checkpoint_path,
                        meta={'icevision_version': '0.12.0'})

#%%
infer_dl = model_type.infer_dl(val_ds, batch_size=4, shuffle=False)
preds = model_type.predict_from_dl(combined_model, infer_dl, keep_images=True)

show_preds(preds=preds[90:95])


#%% Load a previous model
load_mod_path = '/home/andrewchan/Desktop/ice_vision/snapshots/pneumonia_icevision_all_data.pth.pth'
#model_cont = model_type.model(num_classes=len(pos_parser.class_map),**extra_args)
#model_cont.load_state_dict(torch.load(load_mod_path))

checkpoint_path = 'coco-retinanet-checkpoint-full.pth'


new_learner = model_type.fastai.learner(
    dls=[combined_train_dl, combined_valid_dl], 
    model=combined_model, 
    metrics=metrics)
new_learner.load(load_mod_path)


#%% making a test dataloader

# test_pos = '/home/andrewchan/Desktop/ice_vision/data_v2/test_a1_pos.csv'
# test_neg = '/home/andrewchan/Desktop/ice_vision/data_v2/test_a1_neg.csv'
# df_tsp = pd.read_csv(test_pos,names = pos_col, header = None)
# df_tsn= pd.read_csv(test_neg,names = neg_col, header = None)
# posIdMap_test = IDMap(df_tsp.fname.to_list())
# negIdMap_test = IDMap(df_tsn.fname.to_list())

# pos_parser_test = PneumoniaParser(template_record, data_dir,posIdMap_test)


# ts_records_pos, *_ = pos_parser.parse(data_splitter = SingleSplitSplitter(), IDMap = negIdMap_test)
# ts_records_neg, *_ = neg_parser.parse(data_splitter = SingleSplitSplitter(), IDMap = posIdMap_test)

#%%

checkpoint_and_model = checkpoint.model_from_checkpoint(checkpoint_path, 
    model_name='torchvision.retinanet', 
    backbone_name='resnet50_fpn',
    img_size=384, 
    is_coco=False)

model_type = checkpoint_and_model["model_type"]
backbone = checkpoint_and_model["backbone"]
class_map = checkpoint_and_model["class_map"]
img_size = checkpoint_and_model["img_size"]

model = checkpoint_and_model["model"]
#device=next(model.parameters()).device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


infer_dl = model_type.infer_dl(val_ds, batch_size=4, shuffle=False)
preds = model_type.predict_from_dl(model, infer_dl, keep_images=True)
show_preds(preds=preds[90:95])

#%% look at images from loaded model
new_learner.predict()

#%% Resume trainign a model
new_learner.fine_tune(40, lr, freeze_epochs=1)











