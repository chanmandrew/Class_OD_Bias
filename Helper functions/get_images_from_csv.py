# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:36:27 2022

@author: Chandrew
"""
import pandas as pd
from zipfile import ZipFile
import pydicom
from PIL import Image
import os
#%%
labels = r"C:\Users\13019\OneDrive\Desktop\segmentation\read_dicom_stuff\labels.csv"
df = pd.read_csv(labels)
#print(df['Image ID'].head())
zip_path = r"C:\Users\13019\OneDrive\Desktop\segmentation\read_dicom_stuff\Seg_data.zip"
new_path = r'C:\Users\13019\OneDrive\Desktop\segmentation\read_dicom_stuff\images'

#%% 

with ZipFile(zip_path, 'r') as zip:
    #zip.printdir()
    for i,img in df.iterrows():
       filename = img['Image ID']+'.dcm'
       zip.extract('training images/' +img['Image ID']+'.dcm',new_path)
       print(img['Image ID'])
    
#%% 
def dicom_to_jpg(in_file, out_file, out_size):
    """ Convert dicom file to jpg with specified size """
    ds = pydicom.read_file(in_file)
    size = (ds.Columns, ds.Rows)
    mode = 'L'
    im = Image.frombuffer(mode, size, ds.pixel_array,
                          "raw", mode, 0, 1).convert("L")
    im = im.resize((out_size, out_size), resample=Image.BICUBIC)
    im.save(out_file, quality=95)
    
#%%

image_dir = r'C:\Users\13019\OneDrive\Desktop\segmentation\read_dicom_stuff\images\training images'
image_loc = r'C:\Users\13019\OneDrive\Desktop\segmentation\read_dicom_stuff\images'
for dcm_file in os.listdir(image_dir):
    bn = os.path.basename(dcm_file)
    out_file = os.path.join(image_loc, bn[:-4]+".jpg")
    dicom_to_jpg(dcm_file, out_file, 1024)
    