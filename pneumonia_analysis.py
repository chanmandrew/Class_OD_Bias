#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:17:26 2023

@author: seangarin

analysis run file for the ai-fairness architecture experiment-pneumonia comparison.

Input: 

    parent - parent path to folder

    example_text_folder - rest of path to each Aim folder (ex. \Classification\Analysis\Aim_1)

    sex - Male, Female

    metric - f1, youdens used - can change but is hardcoded 

Output:

for analysis function

    metric_results.csv - includes analysis for each sex (split by percent if applicable too)
        includes: AUROC, AUPRC, TPR, FPR, TNR, FNR, TP, TN, FP, FN, NNF, PNF

for aggreagte function

    statistic.csv - inlcudes p-value for AUROC, AUPRC, TPR, FPR, TNR, FNR

    plots for each of AUROC, AUPRC, TPR, FPR, TNR, FNR comparing M/F using p-value specified (Mann-Whitney or t-test_paired)

"""

import os
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from statannotations.Annotator import Annotator
from decimal import Decimal
 
parent = r'C:\Users\sgari\Desktop\experiments\fairness_ai_architecture\A_Chan_PIPELINE'

class_aim_1_folder = parent + r'\Classification\Analysis\Aim_1'
class_aim_2_folder = parent + r'\Classification\Analysis\Aim_2'

od_aim_1_folder = parent + r'\Object_detection\Analysis\Aim_1'
od_aim_2_folder = parent + r'\Object_detection\Analysis\Aim_2'
sex = ['M','F']

def aggregate(dataframe, c_o_type, aim, metric):    
    print('*****************************\nmoving on to aggregate analysis\n*******************************')
    #Metrics across trials 
    var = ['AUROC','AUPRC','TPR','FPR','TNR','FNR']
    p_val = ['Mann-Whitney','t-test_paired']
    
    output_path = parent + '/' + c_o_type + '/Analysis/' + aim + '/results/' + metric
    
    print(aim)    
    if aim == "Aim_1":
        aim_1_c = dataframe[['Trial','Sex','AUROC','AUPRC','TPR','FPR','TNR','FNR']]
        for x in p_val:
            if x == 'Mann-Whitney':
                try:
                    os.mkdir(output_path + '/' + x)
                except FileExistsError:
                    print()
                metric_df = pd.DataFrame(columns=('Metric','u1','p_value'))
                metric_df['Metric'] = var
                metric_df.set_index('Metric', inplace=True)
                print(metric_df)
                for i in var:
                    males = aim_1_c.loc[aim_1_c['Sex'] == 'M']
                    males = males[i].tolist()
                    females = aim_1_c.loc[aim_1_c['Sex'] == 'F']
                    females = females[i].tolist()
                    u1, p = stats.mannwhitneyu(males, females, alternative = 'two-sided')
                    p = f"p={Decimal(p):.2E}"
                    metric_df.at[i,'u1'] = u1
                    metric_df.at[i,'p_value'] = p
                    print(i, '\nu1 value ',u1, '\np-value ',p, '\n')
                
                    plot = sns.boxplot(data=aim_1_c,x='Sex',y=i,orient='v')
                    plot_final = Annotator(
                        plot, 
                        data=aim_1_c, x='Sex', y=i, orient='v',
                        pairs = [('M','F')])
                    
                    plot_final.configure(test=x)
                    plot_final.apply_and_annotate()
                    plt.savefig(output_path + '/' + x + '/' + i + '_plot.jpeg')
                    plt.show()
                    plt.clf()
                    print('\n ')
                
                metric_df.to_csv(parent + '/' + c_o_type + '/Analysis/' + aim + '/results/' + metric + '/' + x + '/'+ x + '_statistics.csv')
            
            elif x == 't-test_paired':
                try:
                    os.mkdir(output_path + '/' + x)
                except FileExistsError:
                    print()
                metric_df = pd.DataFrame(columns=('Metric','statistic','p_value'))
                metric_df['Metric'] = var
                metric_df.set_index('Metric', inplace=True)
                print(metric_df)
                try:
                    output_path = parent + '/' + c_o_type + '/Analysis/' + aim + '/results/' + metric
                    os.mkdir(output_path)
                    os.mkdir(output_path + '/' + x)
                except FileExistsError:
                    print()
                for i in var:
                        males = aim_1_c.loc[aim_1_c['Sex'] == 'M']
                        males = males[i].tolist()
                        females = aim_1_c.loc[aim_1_c['Sex'] == 'F']
                        females = females[i].tolist()
                        stat, p = stats.ttest_rel(males, females, alternative = 'two-sided')
                        p = f"p={Decimal(p):.2E}"
                        metric_df.at[i,'statistic'] = stat
                        metric_df.at[i,'p_value'] = p
                        print(i, '\nstat value ',stat, '\np-value ',p, '\n')

                        
                        plot = sns.boxplot(data=aim_1_c,x='Sex',y=i,orient='v')
                        plot_final = Annotator(
                            plot, 
                            data=aim_1_c, x='Sex', y=i, orient='v',
                            pairs = [('M','F')])
                        
                        
                        plot_final.configure(test=x)
                        plot_final.apply_and_annotate()
                        plt.savefig(output_path + '/' + x + '/' + i + '_plot.jpeg')
                        plt.show()
                        plt.clf()
                        print('\n ')

        metric_df.to_csv(parent + '/' + c_o_type + '/Analysis/' + aim + '/results/' + metric + '/' + x + '/'+ x + '_statistics.csv')    

    if aim == 'Aim_2':
        percent_list = dataframe.Percent.unique()
        percent_list = percent_list.tolist()
        aim_2_c = dataframe[['Trial','Sex','Percent','AUROC','AUPRC','TPR','FPR','TNR','FNR']]    
        
        for x in p_val:
            if x == 'Mann-Whitney':
                metric_df = pd.DataFrame(columns=('Metric','Percent','u1','p_value'))
                metric_df['Metric'] = var
                metric_df.set_index('Metric', inplace=True)
                print(metric_df)
                for i in var:
                    for per in percent_list:
                    
                        try:
                            os.mkdir(output_path + '/' + x)
                        except FileExistsError:
                            print()

                        males = aim_2_c.loc[(aim_2_c['Sex'] == 'M') & (aim_2_c['Percent'] == per)]
                        males = males[i].tolist()
                        females = aim_2_c.loc[(aim_2_c['Sex'] == 'F') & (aim_2_c['Percent'] == per)]
                        females = females[i].tolist()
                        u1, p = stats.mannwhitneyu(males, females, alternative = 'two-sided')
                        p = f"p={Decimal(p):.2E}"
                        new_df = {'Metric':i,'Percent':per,'u1':u1,'p_value':p}
                        print('adding:',new_df)
                        metric_df = metric_df.append(new_df,ignore_index=True)
                        print(i, '\nu1 value ',u1, '\np-value ',p, '\n')
                                    
                        plot = sns.boxplot(data=dataframe,
                                        x='Percent', 
                                        y=i, 
                                        hue='Sex',
                                        orient='v',
                                        order=('0%F','25%F','50%F', '75%F', '100%F'))
                        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                        
                        plot_final = Annotator(
                            plot, 
                            data=analysis_df,x='Percent', y=i, hue='Sex',orient='v',
                            pairs = [
                                (('0%F','M'),('0%F','F')),
                                (('25%F','M'),('25%F','F')),
                                (('50%F','M'),('50%F','F')),
                                (('75%F','M'),('75%F','F')),
                                (('100%F','M'),('100%F','F'))
                                ]
                            )
                                            
                        plot_final.configure(test='Mann-Whitney')
                        plot_final.apply_and_annotate()
                        plt.savefig(output_path + '/' + x + '/' + i + '_plot.jpeg', bbox_inches='tight')
                        plt.show()
                        plt.clf()
                        print('\n ')
                    
                metric_df.to_csv(parent + '/' + c_o_type + '/Analysis/' + aim + '/results/' + metric + '/' + x + '/'+ x + '_statistics.csv')
            
            elif x == 't-test_paired':
                metric_df = pd.DataFrame(columns=('Metric','Percent','stat','p_value')) 
                metric_df['Metric'] = var
                metric_df.set_index('Metric', inplace=True)
                print(metric_df)        
                for i in var:
                    for per in percent_list:
                        try:
                            os.mkdir(output_path + '/' + x)
                        except FileExistsError:
                            print()
                        
                        males = aim_2_c.loc[(aim_2_c['Sex'] == 'M') & (aim_2_c['Percent'] == per)]
                        males = males[i].tolist()
                        females = aim_2_c.loc[(aim_2_c['Sex'] == 'F') & (aim_2_c['Percent'] == per)]
                        females = females[i].tolist()
                        stat, p = stats.ttest_rel(males, females, alternative = 'two-sided')
                        p = f"p={Decimal(p):.2E}"
                        new_df = {'Metric':i,'Percent':per,'stat':stat,'p_value':p}
                        print('adding:',new_df)
                        metric_df = metric_df.append(new_df,ignore_index=True)
                        print(i, '\nu1 value ', stat, '\np-value ',p, '\n')
                                    
                        plot = sns.boxplot(data=dataframe,
                                        x='Percent', 
                                        y=i, 
                                        hue='Sex',
                                        orient='v',
                                        order=('0%F','25%F','50%F', '75%F', '100%F'))
                        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                        
                        plot_final = Annotator(
                            plot, 
                            data=analysis_df,x='Percent', y=i, hue='Sex',orient='v',
                            pairs = [
                                (('0%F','M'),('0%F','F')),
                                (('25%F','M'),('25%F','F')),
                                (('50%F','M'),('50%F','F')),
                                (('75%F','M'),('75%F','F')),
                                (('100%F','M'),('100%F','F'))
                                ]
                            )
                                                
                        plot_final.configure(test='t-test_paired')
                        plot_final.apply_and_annotate()
                        plt.savefig(output_path + '/' + x + '/' + i + '_plot.jpeg',bbox_inches='tight')
                        plt.show()
                        plt.clf()
                        print('\n ')

                metric_df.to_csv(parent + '/' + c_o_type + '/Analysis/' + aim + '/results/' + metric + '/' + x + '/'+ x + '_statistics.csv')

    return metric_df

def analysis(input_path, metric):
    if "Aim_1" in input_path:
        aim_1_df = pd.DataFrame(columns = ['Aim','Type','Trial','Sex','AUROC','AUPRC','TPR','FPR','TNR','FNR','TP','TN','FP','FN','NNF','PNF'])
        print('AIM 1')
        folder = os.listdir(input_path)

        for trial in folder:
            print('folder',folder)
            if 'results' not in trial:
                trial_walk = os.path.join(input_path,trial)
                print(trial_walk)
                for root, dirs, file in os.walk(trial_walk):
                    for f in file:
                        run = os.path.join(str(trial_walk),str(f))
                        run_split = run.split(os.sep)
                        print('split',run_split)
                        c_o_type = run_split[7]
                        print('c_o_type ',c_o_type)
                        aim = run_split[9]
                        print('aim ',aim)
                        trial = run_split[10]
                        trial_split = trial.split('_')
                        trial = trial_split[1]
                        trial = ''.join(x for x in trial if x.isdigit())
                        print('trial ',trial)
                        #trial = trial_num above
                        file_df = pd.read_csv(run) 

                        try:
                            output_path = parent + '/' + c_o_type + '/Analysis/' + aim + '/results/' + metric
                            os.mkdir(output_path)
                        except FileExistsError:
                            print()
                        
                        y_true = file_df['ground_truth']

                        if "Classification" in input_path:
                            y_pred = file_df['pos_pred']
                        
                        elif "Object_detection" in input_path:
                            y_pred = file_df['confidence']


                        if metric == 'f1' or 'F1':
                            print('using', metric, 'threshold')
                            precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
                            index = np.argmax(2*recall*precision/(recall+precision))
                            threshold = thresholds[index]
                            
                        elif metric == 'youden' or 'Youden':
                            print('using', metric, 'threshold')
                            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
                            index = np.argmax(tpr - fpr)
                            threshold = thresholds[index]
                            
                        else:
                            print('using 0.5 as threshold')
                            threshold = 0.5
                        
                        
                        for s in sex: 
                            group_df = file_df.groupby(['sex']).get_group((s)).reset_index(drop=True)               
                        
                            y_true_sub = group_df['ground_truth']

                            if "Classification" in input_path:
                                y_pred = group_df['pos_pred']
                        
                            elif "Object_detection" in input_path:
                                y_pred_sub = group_df['confidence']
                        
                            auroc = metrics.roc_auc_score(y_true_sub, y_pred_sub)
                            print('auroc is',auroc)
                            
                            auprc = metrics.average_precision_score(y_true_sub, y_pred_sub)
                            print('auprc is', auprc)
                            
                            tn, fp, fn, tp = metrics.confusion_matrix(y_true_sub, (y_pred_sub > threshold).astype(float)).ravel()
                        
                            cm = metrics.confusion_matrix(y_true_sub, (y_pred_sub > threshold).astype(float))
                            print(cm)
                        
                            nnf = tn + fn
                            pnf = tp + fp
                            tpr = tp/(tp + fn)
                            tnr = tn/(fp + tn)              
                            fpr = fp/(fp + tn) 
                                    
                            fnr = fn/(tp + fn)
                            
                            aim_1_df.loc[len(aim_1_df.index)] = [aim,c_o_type,trial,s,auroc,auprc,tpr,fpr,tnr,fnr,tp,tn,fp,fn,nnf,pnf] 
                            
        aim_1_df.to_csv(parent + '/' + c_o_type + '/Analysis/' + aim + '/results/' + metric + '/' + metric + '_results.csv')                  
        aggregate(aim_1_df,c_o_type,aim,metric)
        return aim_1_df
         
           
    elif "Aim_2" in input_path: 
        aim_2_df = pd.DataFrame(columns = ['Aim','Type','Trial','Sex','Percent','AUROC','AUPRC','TPR','FPR','TNR','FNR','TP','TN','FP','FN','NNF','PNF'])
        print('AIM 2')
        folder = os.listdir(input_path)
 
        for trial in folder:
            print('folder',folder)
            if 'results' not in trial:
                trial_walk = os.path.join(input_path,trial)
                print(trial_walk)
                for root, dirs, file in os.walk(trial_walk):
                    for f in file:
                        run = os.path.join(str(trial_walk),str(f))
                        run_split = run.split(os.sep)
                        print('split',run_split)
                        c_o_type = run_split[7]
                        print('c_o_type ',c_o_type)
                        aim = run_split[9]
                        print('aim ',aim)
                        trial = run_split[10]
                        trial_split = trial.split('_')
                        trial = trial_split[1]
                        trial = ''.join(x for x in trial if x.isdigit())
                        print('trial ',trial)
                        #trial = trial_num above
                        percentage = run_split[11]
                        percentage = percentage.split('_')
                        print(percentage)
                        percent = percentage[-1]
                        percent = percent[:-4]
                        print('percent ',percent)
                        file_df = pd.read_csv(run)   
                        
                        try:
                            output_path = parent + '/' + c_o_type + '/Analysis/' + aim + '/results/'
                            os.mkdir(output_path)
                        except FileExistsError:
                            print()

                        try:
                            output_path = parent + '/' + c_o_type + '/Analysis/' + aim + '/results/' + metric
                            os.mkdir(output_path)
                        except FileExistsError:
                            print()

                        y_true = file_df['ground_truth']

                        if "Classification" in input_path:
                            y_pred = file_df['pos_pred']
                        
                        elif "Object_detection" in input_path:
                            y_pred = file_df['confidence']
                        
                        
                        if metric == 'f1' or 'F1':
                            print('using', metric, 'threshold')
                            precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
                            index = np.argmax(2*recall*precision/(recall+precision))
                            threshold = thresholds[index]
                            
                        elif metric == 'youden' or 'Youden':
                            print('using', metric, 'threshold')
                            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
                            index = np.argmax(tpr - fpr)
                            threshold = thresholds[index]
                            
                        else:
                            print('using 0.5 as threshold')
                            threshold = 0.5
                        
                        
                        for s in sex: 
                            group_df = file_df.groupby(['sex']).get_group((s)).reset_index(drop=True)               
                            
                            y_true_sub = group_df['ground_truth']
                            
                            if "Classification" in input_path:
                                y_pred = group_df['pos_pred']
                        
                            elif "Object_detection" in input_path:
                                y_pred_sub = group_df['confidence']
                            
                            
                            auroc = metrics.roc_auc_score(y_true_sub, y_pred_sub)
                            print('auroc is',auroc)
                        
                            auprc = metrics.average_precision_score(y_true_sub, y_pred_sub)
                            print('auprc is', auprc)

                            tn, fp, fn, tp = metrics.confusion_matrix(y_true_sub, (y_pred_sub > threshold).astype(float)).ravel()
                            
                            cm = metrics.confusion_matrix(y_true_sub, (y_pred_sub > threshold).astype(float))
                            print(cm)
                            
                            nnf = tn + fn
                            pnf = tp + fp
                            tpr = tp/(tp + fn)
                            tnr = tn/(fp + tn)              
                            fpr = fp/(fp + tn) 
                                        
                            fnr = fn/(tp + fn)
                            
                            aim_2_df.loc[len(aim_2_df.index)] = [aim,c_o_type,trial,s,percent,auroc,auprc,tpr,fpr,tnr,fnr,tp,tn,fp,fn,nnf,pnf]
        try:                    
            aim_2_df.to_csv(parent + '/' + c_o_type + '/Analysis/' + aim + '/results/' + metric + '/' + metric + '_results.csv')   
        except UnboundLocalError:
            print()
        aggregate(aim_2_df,c_o_type,aim,metric)   
        return aim_2_df 
    
    else:
        print('not valid')
        


metric = ['f1','youden']
folders = [class_aim_1_folder, class_aim_2_folder, od_aim_1_folder, od_aim_2_folder]

for f in folders:
    for m in metric:
        analysis_df = analysis(f, m)
