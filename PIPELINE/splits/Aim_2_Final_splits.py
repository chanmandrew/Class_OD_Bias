# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:04:40 2022

point of this is to make the splits for the train/validation/test sets
This is for aim 2.

@author: Chandrew
"""

import pandas as pd
from random import sample
import numpy as np
import os
#%% set parameters
def make_data_splits_2(csv_dir,trial_num):
    
    tr_per = 0.8
    ts_per = 0.1
    va_per = 0.1
    
    
    #trial_num = trial_num;
    #csv_dir = 'C:/Users/13019/OneDrive/Desktop/Aim_2_splits/trial_' + str(trial_num)
    the_dir = csv_dir+'/Aim_2/trial_' + str(trial_num)
    os.makedirs(the_dir)
    #%%
    compdata = csv_dir+'/compiled_data.csv'
    compdf = pd.read_csv(compdata) #has all the data
    
    #df is the converted one for retinanet
    df = pd.DataFrame(columns=['fileName','xmin','ymin','xmax','ymax','class']) 
    
    # get pt id and turn it into a filename
    df['fileName'] = compdf['patientId']
    suffix = '.jpg'
    df['fileName'] = df['fileName'] + suffix
    
    # get the disease status and turn it into the correct format
    df['class'] = compdf['Target'] 
    df['class'] = df['class'].replace(1,'pneumonia')
    df = df.replace(np.nan,0)
    df = df.replace(0,"")
    
    sz = 1000
    # get the x and y coords of the bounding boxes
    df['xmin'] = compdf['x'].astype(float).astype('Int64')
    df['ymin'] = compdf['y'].astype(float).astype('Int64')
    df['xmax'] = (compdf['x']+compdf['width']).astype(float).astype('Int64')
    df.loc[df['xmax']>sz,'xmax'] = sz
    df['ymax'] = (compdf['y']+compdf['height']).astype(float).astype('Int64')
    df.loc[df['ymax']>sz,'ymax'] = sz
    print('og processing done...')
    
    #%%  Creating the four different groups
    uniqueIds = csv_dir+'/Unique_Ids_Img_Id.csv'
    df_data = pd.read_csv(uniqueIds)
    f_pos = list()
    f_neg = list()
    m_pos = list()
    m_neg = list() #pd.DataFrame(df_data.columns)
    
    for ind, row in df_data.iterrows():
        if(row['Patient Gender'] == 'F' and row['Target'] == 1):
            f_pos.append(row['patientId'])
        if(row['Patient Gender'] == 'F' and row['Target'] == 0):
           f_neg.append(row['patientId'])
        if(row['Patient Gender'] == 'M' and row['Target'] == 1):
           m_pos.append(row['patientId'])
        if(row['Patient Gender'] == 'M' and row['Target'] == 0):
           m_neg.append(row['patientId'])
    print('lists finished...')
    
    
    #%% creating the female and male image Id lists. 
    
    uniqueIds = csv_dir+'/Unique_Ids_Img_Id_v2.csv'
    df_imgid = pd.read_csv(uniqueIds,dtype = str)
    
    f_ido = list()
    m_ido = list()
    
    #now I have the F_ids
    
    for ind,row in df_imgid.iterrows():
        rowz = compdf.loc[compdf['Image Index'].str[:8] == row['imageId']]
        if(rowz['Patient Gender'].iloc[0] == 'F'):
            f_ido.append(row['imageId'])
          #  print('F: ' + row['imageId'])
        if(rowz['Patient Gender'].iloc[0] == 'M'):
            m_ido.append(row['imageId'])
          #  print('M: ' + row['imageId'])
    
    print("finished pre-processing")
    
    # %% Quick tests
    # Test to make sure the things got sorted into the right groups
    def quick_test_1(fp,fn,mp,mn,compdata):
        #[fepo,fene,mapo,mane]
        tf = True
        
        df = pd.read_csv(compdata)
        if fp:
            test = df.loc[df['patientId'] == sample(fp,1)[0]]
            tf = tf and test['Patient Gender'].values[0] == 'F' and test['Target'].values[0] == 1
        if fn:
            test = df.loc[df['patientId'] == sample(fn,1)[0]]
            tf = tf and test['Patient Gender'].values[0] == 'F' and test['Target'].values[0] == 0
        if mp:
            test = df.loc[df['patientId'] == sample(mp,1)[0]]
            tf = tf and test['Patient Gender'].values[0] == 'M' and test['Target'].values[0] == 1
        if mn:
            test = df.loc[df['patientId'] == sample(mn,1)[0]]
            tf = tf and test['Patient Gender'].values[0] == 'M' and test['Target'].values[0] == 0
        if tf:
            print("Groups are where they are supposed to be!!")
        else:
            print("X there are misplaced subjects")
    
    # Test to make sure that the lists are unique
    def check_unique(l1,l2,l3,l4):
        if (len(l1) == len(set(l1)) and 
            len(l2) == len(set(l2)) and 
            len(l3) == len(set(l3)) and 
            len(l4) == len(set(l4))):
            print('All unique!')
        else:  print('X DUPLICATES WITHIN SET')
    
    #def (test,train): #hmm this is not the rigth one. Need to check the image Ids.
    
            
    def check_leakage(ts,tr,val,compdf):
        test = list()
        train = list()
        vald = list()
        for group in ts:
           for sid in group:
               test.append(compdf.loc[compdf['patientId'] == sid]['Image Index'].iloc[0][:8])
        for group in tr:
           for sid in group:
               train.append(compdf.loc[compdf['patientId'] == sid]['Image Index'].iloc[0][:8])
        for group in val:
            for sid in group:
                vald.append(compdf.loc[compdf['patientId'] == sid]['Image Index'].iloc[0][:8])
        print("test num: "+str(len(test)))
        print("train num: "+str(len(train)))
        print("val num: "+str(len(vald)))
        if (len(set(test).intersection(train)) == 0 and
            len(set(test).intersection(vald)) == 0  and
            len(set(train).intersection(vald)) == 0):
            print("there's no overlap between patients!")
        else:
            print('X THERE IS PATIENT OVERLAP')
    
    
    
    def check_ratios(test,train,val,fratio,mratio):
        tf = True
        if train[0]:
            tf = (tf and abs(len(train[0])/(len(train[0]+train[1])) - fratio) < 0.0005 and 
                    abs(len(test[0])/(len(test[0]+test[1])) - fratio) < 0.0005 and
                    abs(len(val[0])/(len(val[0]+val[1])) - fratio< 0.0005))
            print("ratio: "+ str(fratio) +'\n')
            print("f ratio train: "+ str(len(train[0])/(len(train[0]+train[1]))) +'\n')
            print("f ratio test: "+ str(len(test[0])/(len(test[0]+test[1]))) +'\n')
            print("f ratio val: "+ str(len(val[0])/(len(val[0]+val[1]))) +'\n')
            
        if train[2]:
            tf = (tf and abs(len(train[2])/(len(train[2]+train[3])) - mratio) < 0.0005 and 
                    abs(len(test[2])/(len(test[2]+test[3])) - mratio) < 0.0005 and
                    abs(len(val[2])/(len(val[2]+val[3])) - mratio< 0.0005))
            print("ratio: "+ str(mratio) +'\n')
            print("m ratio train: "+ str(len(train[2])/(len(train[2]+train[3]))) +'\n')
            print("m ratio test: "+ str(len(test[2])/(len(test[2]+test[3]))) +'\n')
            print("m ratio val: "+ str(len(val[2])/(len(val[2]+val[3]))) +'\n')
        if tf: 
            print("ratios are consistent")
        else:
            print('X ratios are not consistent!')
    #%% Getting number for each group
    ratio = np.size(f_pos)/(np.size(f_neg)+np.size(f_pos))
    num_lim = np.size(f_pos) + np.size(f_neg)
    fsplits= [1,0.75,0.5,0.25,0]
    #fsplits= [1,0]
    
    #%% creating test datasets
    test_ids = [list(), list(), list(), list()]
    test_num = [int(num_lim*0.1*ratio),int(num_lim*0.1*(1-ratio)),int(num_lim*0.1*ratio),int(num_lim*0.1*(1-ratio))]
    
    while not(len(test_ids[0])==test_num[0] and len(test_ids[1])==test_num[1]):
        new_fid = sample(f_ido,1)[0]
        rowz = df_data.loc[df_data['imageId'].str[:8] == new_fid]
        if (len(test_ids[0]) + len(rowz.loc[rowz['Target'] == 1].index) <=test_num[0] and
            len(test_ids[1]) + len(rowz.loc[rowz['Target'] == 0].index) <=test_num[1]):
            
            add_pos = rowz.loc[rowz['Target'] == 1]['patientId'].values.tolist()
            test_ids[0] += add_pos
            for sid in add_pos:
                if sid in f_pos:
                    f_pos.remove(sid)
            
            add_neg = rowz.loc[rowz['Target'] == 0]['patientId'].values.tolist()        
            test_ids[1] += add_neg
            for sid in add_neg:
                if sid in f_neg:
                    f_neg.remove(sid)         
            
            f_ido.remove(new_fid)
        
    print('making test female lists done')
    
    
    while not(len(test_ids[2])==test_num[2] and len(test_ids[3])==test_num[3]):
        new_mid = sample(m_ido,1)[0]
        rowz = df_data.loc[df_data['imageId'].str[:8] == new_mid]
        if (len(test_ids[2]) + len(rowz.loc[rowz['Target'] == 1].index) <=test_num[2] and
            len(test_ids[3]) + len(rowz.loc[rowz['Target'] == 0].index) <=test_num[3]):
            
            add_pos = rowz.loc[rowz['Target'] == 1]['patientId'].values.tolist()
            test_ids[2] += add_pos
            for sid in add_pos:
                if sid in m_pos:
                    m_pos.remove(sid)
            
            add_neg = rowz.loc[rowz['Target'] == 0]['patientId'].values.tolist()
            test_ids[3] += add_neg
            for sid in add_neg:
                if sid in m_neg:
                    m_neg.remove(sid)
    
            m_ido.remove(new_mid)
    print('making test male lists done')
    
    
    
    for percent in fsplits:
        new_dir = the_dir+'/' +str(int(percent*100))+'%F/'
        os.makedirs(new_dir)
        
        f_id = f_ido.copy()
        m_id = m_ido.copy()
        fp = f_pos.copy()
        fn = f_neg.copy()
        mp = m_pos.copy()
        mn = m_neg.copy()
        
        #Get the total num for the gender splits
        num_f_pos = ratio*num_lim*percent #check the numbers for each one of these
        num_f_neg = (1-ratio)*num_lim*percent
        num_m_pos = ratio*num_lim*(1-percent)
        num_m_neg = (1-ratio)*num_lim*(1-percent) #expect this to be 0 for Frist run
    
        total_num = [num_f_pos,num_f_neg,num_m_pos,num_m_neg]
        train_num = [int(x*tr_per) for x in total_num]
        #test_num = [int(x*ts_per) for x in total_num]
       
        val_num = [int(x*va_per) for x in total_num]
        
        
        #%% creating validation datasets
        val_ids = [list(), list(), list(), list()]
        
        
        while not(len(val_ids[0])==val_num[0] and len(val_ids[1])==val_num[1]):
            new_fid = sample(f_id,1)[0]
            rowz = df_data.loc[df_data['imageId'].str[:8] == new_fid]
            if (len(val_ids[0]) + len(rowz.loc[rowz['Target'] == 1].index) <=val_num[0] and
                len(val_ids[1]) + len(rowz.loc[rowz['Target'] == 0].index) <=val_num[1]):
                
                add_pos = rowz.loc[rowz['Target'] == 1]['patientId'].values.tolist()
                val_ids[0] += add_pos
                for sid in add_pos:
                    if sid in fp:
                        fp.remove(sid)
                
                add_neg = rowz.loc[rowz['Target'] == 0]['patientId'].values.tolist()        
                val_ids[1] += add_neg
                for sid in add_neg:
                    if sid in fn:
                        fn.remove(sid)         
                
                f_id.remove(new_fid)
            
        print('making val female lists done')
        
        
        while not(len(val_ids[2])==val_num[2] and len(val_ids[3])==val_num[3]):
            new_mid = sample(m_id,1)[0]
            rowz = df_data.loc[df_data['imageId'].str[:8] == new_mid]
            if (len(val_ids[2]) + len(rowz.loc[rowz['Target'] == 1].index) <=val_num[2] and
                len(val_ids[3]) + len(rowz.loc[rowz['Target'] == 0].index) <=val_num[3]):
                
                add_pos = rowz.loc[rowz['Target'] == 1]['patientId'].values.tolist()
                val_ids[2] += add_pos
                for sid in add_pos:
                    if sid in mp:
                        mp.remove(sid)
                
                add_neg = rowz.loc[rowz['Target'] == 0]['patientId'].values.tolist()
                val_ids[3] += add_neg
                for sid in add_neg:
                    if sid in mn:
                        mn.remove(sid)
        
                m_id.remove(new_mid)
        print('making val male lists done')
        
        #%% creating training datasets
        train_ids = [sample(fp,train_num[0]),
                     sample(fn,train_num[1]), 
                     sample(mp,train_num[2]), 
                     sample(mn,train_num[3])]
        print('all lists done')
    
    
    #%% checking some things. 
        check_leakage(test_ids,train_ids,val_ids,compdf)
        check_unique(test_ids[0],test_ids[1],test_ids[2],test_ids[3])
        check_unique(train_ids[0],train_ids[1],train_ids[2],train_ids[3])
        check_unique(val_ids[0],val_ids[1],val_ids[2],val_ids[3])
    
    #%%
        #train_path = r"C:\Users\13019\OneDrive\Desktop\Aim_2_splits"
        #test_path = r"C:\Users\13019\OneDrive\Desktop\Aim_2_splits"
        #val_path = r"C:\Users\13019\OneDrive\Desktop\Aim_2_splits"
    
        traincsv = pd.DataFrame(columns = df.columns)
        for group in train_ids:
            for ptId in group:
                addRow = df.loc[df['fileName'] ==  ptId + suffix]
                traincsv = pd.concat([traincsv,addRow],axis = 0,ignore_index = True)
            #print(traincsv.head())
        name_train = os.path.join(new_dir,'train_a2_'+str(int(percent*100))+'%F_.csv')
        traincsv.to_csv(name_train, index=False,header= False)
        #print('train done')
                   
        # validation csv
        valcsv = pd.DataFrame(columns = df.columns)
        for group in val_ids:    
            for ptId in group:
                addRow = df.loc[df['fileName'] == ptId + suffix]
                valcsv = pd.concat([valcsv,addRow],axis = 0,ignore_index = True)
        name_val = os.path.join(new_dir, 'val_a2_'+str(int(percent*100))+'%F_.csv')
        valcsv.to_csv(name_val, index=False,header= False)
        #print('val done')
        
        # testing split
        testcsv = pd.DataFrame(columns = df.columns)
        for group in test_ids:    
            for ptId in group:
                addRow = df.loc[df['fileName'] == ptId + suffix]
                testcsv = pd.concat([testcsv,addRow],axis = 0,ignore_index = True)
                #testcsv = testcsv.append(addRow,ignore_index = True)
        name_test = os.path.join(new_dir, 'test_a2_'+str(int(percent*100))+'%F_.csv')
        testcsv.to_csv(name_test, index=False,header= False)
        #print('test done')
        
        print('-----------' + str(percent*100) + ' % female dataset finished' + '-----------' )
    
    # # testing split if you want one file per folder
    # testcsv = pd.DataFrame(columns = df.columns)
    # for group in test_ids:    
    #     for ptId in group:
    #         addRow = df.loc[df['fileName'] == ptId + suffix]
    #         testcsv = pd.concat([testcsv,addRow],axis = 0,ignore_index = True)
    #         #testcsv = testcsv.append(addRow,ignore_index = True)
    # name_test = os.path.join(the_dir, 'test_a2_trial_'+str(trial_num)+'.csv')
    # testcsv.to_csv(name_test, index=False,header= False)
    # #print('test done')
    
    print('all finsihed AWHOT')

