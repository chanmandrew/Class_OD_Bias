# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:23:38 2022

@author: Chandrew
"""

'''
rle2mask takes
    filename: String of the ImageId or whatever the image file is called without the extension
    rle_list: which is the rles for the different segmentation
    width: width of the image (1024)
    height: height of the image (1024)

outputs
    numpy array which is the mask for a given image (should be all 0 or 25)
'''
#%%
import numpy as np

#%%
def rle2mask(filename,rle_list, width, height):
    overall_mask = np.zeros(width* height).reshape(width, height)
    
    for i in rle_list:
        mask = np.zeros(width* height)
        array = np.asarray([int(x) for x in i.split()])
        starts = array[0::2]
        lengths = array[1::2]
        current_position = 0
        
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position+lengths[index]] = 255
            current_position += lengths[index]
        mask = mask.reshape(width, height)
        overall_mask += mask
    
    overall_mask[overall_mask > 255] = 255
    if not ((mask==0) | (mask==255)).all():
        print(f"help. Something went wrong making a mask for {filename}")
        #plt.imshow(mask,cmap='Greys'))
    overall_mask = np.fliplr(np.rot90(overall_mask,k=3))
    return overall_mask