#!/usr/bin/env python
# coding: utf-8

# In[48]:
input_shape = None

#################################################
#################################################
## Function
#################################################
#################################################

from shoelace.iterator import LtrIterator
from shoelace.loss.listwise import listnet
from shoelace.loss.listwise import listmle
from shoelace.loss.listwise import listpl
from chainer import training, optimizers, links, Chain
from chainer import serializers
from chainer.dataset import convert
import chainer.functions as F
from lambdamart import LambdaMART
import xgboost as xgb
import lightgbm as lgb
from xgboost import DMatrix
import pickle
from DirectRanker import directRanker


import time
import datetime
import os
import copy

import pandas as pd
import numpy as np

#################################################
##Data 전처리
#################################################

#################################################
## Modeling
#################################################


#################################################
## Test
#################################################

# class Ranker(Chain):
#     def __init__(self, predictor, loss):
#         super(Ranker, self).__init__(predictor=predictor)
#         self.loss = loss
#
#     def __call__(self, x, t):
#         x_hat = self.predictor(x)
#         # return x_hat
#         return self.loss(x_hat, t)
#
#     def predict_result(self, x):
#         x_hat = self.predictor(x)
#         return x_hat

class Ranker(Chain):
    # def __init__(self, predictor, loss):
    #     super(Ranker, self).__init__(predictor=predictor)
    #     self.loss = loss
    def __init__(self, loss):
        super(Ranker, self).__init__()
        with self.init_scope():
            self.l1 = links.Linear(None, 16)
            self.l2 = links.Linear(None, 16)
            self.l3 = links.Linear(None, 8)
            self.l4 = links.Linear(None, 1)
            self.loss = loss

    # def __call__(self, x, t):
    #     x_hat = self.predictor(x)
    def __call__(self, x, t):
        # x_hat = self.predictor(x)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        x_hat = self.l4(h3)
        return self.loss(x_hat, t)

    def predict_result(self, x):
        # x_hat = self.predictor(x)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        x_hat = self.l4(h3)
        return x_hat


def eval(loss_function, iterator):
    """
    Evaluates the mean of given loss function over the entire batch in given
    iterator

    :param loss_function: The loss function to evaluate
    :param iterator: The iterator over the evaluation data set
    :return: The mean loss value
    """
    iterator.reset()
    # results = []
    for batch in iterator:
        input_args = convert.concat_examples(batch)
        result = loss_function.predict_result(*input_args)
        # results.append(loss_function(*input_args).data)
    # return np.mean(results)
    return result

def DataPreprocessingForTesting(originalST):

    processedST = copy.deepcopy(originalST)

    schedule_Index_List = originalST['SCHEDULE_SEQ'].drop_duplicates()

    # PW 추가
    PW = originalST["P"] * originalST["W"]
    originalST.insert(3, "PW", PW)
    processedST.insert(3, "PW", PW)


    # 각각의 schedule별로 정규화를 진행
    for i in schedule_Index_List:
        data2 = originalST[originalST['SCHEDULE_SEQ'] == i]
        data3 = copy.deepcopy(data2)

        # 각각 정규화가 필요한 column 정규화
        if data3['P'].min() == data3['P'].max():
            if data3['P'].min() == 0:
                data3['P'] = 0
            else:
                data3['P'] /= data3['P'].max()
        else:
            data3['P'] = (data3['P'] - data3['P'].min()) / (data3['P'].max() - data3['P'].min())

        if data3['W'].min() == data3['W'].max():
            if data3['W'].min() == 0:
                data3['W'] = 0
            else:
                data3['W'] /= data3['W'].max()
        else:
            data3['W'] = (data3['W'] - data3['W'].min()) / (data3['W'].max() - data3['W'].min())

        if data3['PW'].min() == data3['PW'].max():
            if data3['PW'].min() == 0:
                data3['PW'] = 0
            else:
                data3['PW'] /= data3['PW'].max()
        else:
            data3['PW'] = (data3['PW'] - data3['PW'].min()) / (data3['PW'].max() - data3['PW'].min()) 
            
        processedST[processedST['SCHEDULE_SEQ'] == i] = data3

    X_cols = ['JOBID',
              'P',
              'W',
              'PW',
              'SCHEDULE_SEQ']

    processedST = processedST[X_cols]

    return originalST, processedST

def Set_Value_By_ComparisionFlag(input_data):
    input_data[input_data > 0] = 1
    input_data[input_data < 0] = -1

    return input_data

def GerneratePairwiseJobAttrSetForTest(assigned_job, target_job, assigned_job_original, target_job_original, Machine_Job_table, priorityMachine, machineNum, flag):

    jobA_Attr = pd.Series(assigned_job, index=["P_A", "W_A"])
    jobB_Attr = pd.Series(target_job, index=["P_B", "W_B"])

    if(machineNum==2):
        ## 머신이 할당되기전 머신 KPI
        if (Machine_Job_table.empty):
            M1 = 0
            M2 = 0
            if(M1 <= M2):
                Makespan = M2
            else:
                Makespan = M1
            aM1 = 0
            aM2 = 0
            tM1 = 0
            tM2 = 0
            aMakespan = 0
            tMakespan = 0
            dMakespan = 0

            wc1 = 0
            wc2 = 0
            wc = 0
            awc1 = 0
            awc2 = 0
            twc1 = 0
            twc2 = 0
            awc = 0
            twc = 0

            SkT_M1 = 0
            SkT_M2 = 0

        ## 머신이 할당된 후 머신 KPI
        else :
            ### Makespan
            sumResult = Machine_Job_table.groupby(['MACHINEID'], as_index=False).sum()

            if (sumResult[sumResult['MACHINEID'] == 1].empty):
                M1 = 0
            else:
                M1 = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
                # M1 = aM1
            if (sumResult[sumResult['MACHINEID'] == 2].empty):
                M2 = 0
            else:
                M2 = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]
                # M2 = aM2

            if (M1 <= M2):
                Makespan = M2
            else:
                Makespan = M1

            SkT_M1 = Makespan - M1
            SkT_M2 = Makespan - M2

            if(priorityMachine==1):
                aM1 = M1 + assigned_job_original[0]
                aM2 = M2
                tM1 = M1 + target_job_original[0]
                tM2 = M2
            else: # machine이 2인 경우
                aM1 = M1
                aM2 = M2 + assigned_job_original[0]
                tM1 = M1
                tM2 = M2 + target_job_original[0]

            if (aM1 >= aM2):
                aMakespan = aM1
                dMakespan = aM1 - Makespan
            elif (aM1 < aM2):
                aMakespan = aM2
                dMakespan = aM2 - Makespan

            if (tM1 >= tM2):
                tMakespan = tM1
            elif (tM1 < tM2):
                tMakespan = tM2

            ### Weighted Completion Time
            temp = Machine_Job_table
            temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
            temp['WCP'] = temp['P_cumsum'] * temp['W']

            sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()
            sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
            sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]

            if (sumWCP1.empty):
                wc1 = 0
            else:
                wc1 = sumWCP1.iloc[0, 8]
            if (sumWCP2.empty):
                wc2 = 0
            else:
                wc2 = sumWCP2.iloc[0, 8]
            wc = wc1 + wc2

            if (priorityMachine == 1):
                awc1 = wc1 + aM1 * assigned_job_original[1]
                awc2 = wc2
                twc1 = wc1 + tM1 * target_job_original[1]
                twc2 = wc2
            else:  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2 + aM2 * assigned_job_original[1]
                twc1 = wc1
                twc2 = wc2 + tM2 * target_job_original[1]
            awc = awc1 + awc2
            twc = twc1 + twc2

        machine_now = pd.Series([M1, M2, wc1, wc2, wc, Makespan, SkT_M1, SkT_M2], index=['M1', 'M2', "wc1", "wc2", "wc", "Makespan", "Slack_M1", "Slack_M2"])

        compare_Attr_A = pd.Series([aM1, aM2, aMakespan, awc1, awc2, awc],
                                   index=['assign_M1', 'assign_M2', 'assign_Makespan', 'assign_WC1', 'assign_WC2', 'assign_WC'])
        compare_Attr_B = pd.Series([tM1, tM2, tMakespan, twc1, twc2, twc],
                                   index=['target_M1', 'target_M2', 'target_Makespan', 'target_WC1', 'target_WC2', 'target_WC'])

        compare_Attr = pd.Series([dMakespan], index=["dMakespan"])
        ''''''
        compare_AB_Diff = compare_Attr_A.reset_index(drop=True) - compare_Attr_B.reset_index(drop=True)
        compare_AB_Diff.index = ["M1_AB_Diff", "M2_AB_Diff", "Makespan_AB_Diff","wc1_Diff", "wc2_Diff", "wc_Diff"]

        if(flag==0):
            KPIInfo_Attr = machine_now.append(compare_Attr_A)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr_B)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            insertAttributes = jobA_Attr.append(jobB_Attr)
            insertAttributes = insertAttributes.append(KPIInfo_Attr)
        elif(flag==1):
            KPIInfo_Attr = machine_now.append(compare_Attr)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_AB_Diff)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]

            insertAttributes = jobAB_Diff.append(KPIInfo_Attr)
        elif(flag==2):
            KPIInfo_Attr = machine_now.append(compare_Attr)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            compare_Attr_AB_Flag = copy.deepcopy(compare_AB_Diff)
            compare_Attr_AB_Flag = Set_Value_By_ComparisionFlag(compare_Attr_AB_Flag)
            compare_Attr_AB_Flag.index = ["M1_AB_Flag", "M2_AB_Flag", "Makespan_AB_Flag", "wc1_Flag", "wc2_Flag",
                                          "wc_Flag"]

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Flag = copy.deepcopy(jobAB_Diff)
            jobAB_Flag = Set_Value_By_ComparisionFlag(jobAB_Flag)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]
            jobAB_Flag.index = ["P_AB_Flag", "W_AB_Flag"]

            insertAttributes = jobA_Attr.append(jobB_Attr)
            insertAttributes = jobAB_Flag.append(KPIInfo_Attr)
            insertAttributes = insertAttributes.append(compare_Attr_AB_Flag)
        elif(flag==3):
            KPIInfo_Attr = machine_now.append(compare_Attr_A)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr_B)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_AB_Diff)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            compare_Attr_AB_Flag = copy.deepcopy(compare_AB_Diff)
            compare_Attr_AB_Flag = Set_Value_By_ComparisionFlag(compare_Attr_AB_Flag)
            compare_Attr_AB_Flag.index = ["M1_AB_Flag", "M2_AB_Flag", "Makespan_AB_Flag", "wc1_Flag", "wc2_Flag", "wc_Flag"]

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Flag = copy.deepcopy(jobAB_Diff)
            jobAB_Flag = Set_Value_By_ComparisionFlag(jobAB_Flag)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]
            jobAB_Flag.index = ["P_AB_Flag", "W_AB_Flag"]

            insertAttributes = jobA_Attr.append(jobB_Attr)
            insertAttributes = insertAttributes.append(jobAB_Diff)
            insertAttributes = insertAttributes.append(jobAB_Flag)
            insertAttributes = insertAttributes.append(KPIInfo_Attr)
            insertAttributes = insertAttributes.append(compare_Attr_AB_Flag)

    elif(machineNum==5):
        ## 머신이 할당되기전 머신 KPI
        if (Machine_Job_table.empty):
            M1 = 0
            M2 = 0
            M3 = 0
            M4 = 0
            M5 = 0
            if (M1 >= M2 and M1 >= M3 and M1 >= M4 and M1 >= M5):
                Makespan = M1
            elif (M2 >= M1 and M2 >= M3 and M2 >= M4 and M2 >= M5):
                Makespan = M2
            elif (M3 >= M1 and M3 >= M2 and M3 >= M4 and M3 >= M5):
                Makespan = M3
            elif (M4 >= M1 and M4 >= M2 and M4 >= M3 and M4 >= M5):
                Makespan = M4
            elif (M5 >= M1 and M5 >= M2 and M5 >= M3 and M5 >= M4):
                Makespan = M5
            aM1 = 0
            aM2 = 0
            aM3 = 0
            aM4 = 0
            aM5 = 0
            tM1 = 0
            tM2 = 0
            tM3 = 0
            tM4 = 0
            tM5 = 0
            aMakespan = 0
            tMakespan = 0
            dMakespan = 0

            wc1 = 0
            wc2 = 0
            wc3 = 0
            wc4 = 0
            wc5 = 0
            wc = 0
            awc1 = 0
            awc2 = 0
            awc3 = 0
            awc4 = 0
            awc5 = 0
            twc1 = 0
            twc2 = 0
            twc3 = 0
            twc4 = 0
            twc5 = 0
            awc = 0
            twc = 0

            SkT_M1 = 0
            SkT_M2 = 0
            SkT_M3 = 0
            SkT_M4 = 0
            SkT_M5 = 0

        ## 머신이 할당된 후 머신 KPI
        else:
            ### Makespan
            sumResult = Machine_Job_table.groupby(['MACHINEID'], as_index=False).sum()

            if (sumResult[sumResult['MACHINEID'] == 1].empty):
                M1 = 0
            else:
                M1 = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 2].empty):
                M2 = 0
            else:
                M2 = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 3].empty):
                M3 = 0
            else:
                M3 = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 4].empty):
                M4 = 0
            else:
                M4 = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 5].empty):
                M5 = 0
            else:
                M5 = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]

            if (M1 >= M2 and M1 >= M3 and M1 >= M4 and M1 >= M5):
                Makespan = M1
            elif (M2 >= M1 and M2 >= M3 and M2 >= M4 and M2 >= M5):
                Makespan = M2
            elif (M3 >= M1 and M3 >= M2 and M3 >= M4 and M3 >= M5):
                Makespan = M3
            elif (M4 >= M1 and M4 >= M2 and M4 >= M3 and M4 >= M5):
                Makespan = M4
            elif (M5 >= M1 and M5 >= M2 and M5 >= M3 and M5 >= M4):
                Makespan = M5

            ''''''
            SkT_M1 = Makespan - M1
            SkT_M2 = Makespan - M2
            SkT_M3 = Makespan - M3
            SkT_M4 = Makespan - M4
            SkT_M5 = Makespan - M5

            if (priorityMachine == 1):
                aM1 = M1 + assigned_job_original[0]
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                tM1 = M1 + target_job_original[0]
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
            elif(priorityMachine==2):  # machine이 2인 경우
                aM1 = M1
                aM2 = M2 + assigned_job_original[0]
                aM3 = M3
                aM4 = M4
                aM5 = M5
                tM1 = M1
                tM2 = M2 + target_job_original[0]
                tM3 = M3
                tM4 = M4
                tM5 = M5
            elif (priorityMachine == 3):  # machine이 2인 경우
                aM1 = M1
                aM2 = M2
                aM3 = M3 + assigned_job_original[0]
                aM4 = M4
                aM5 = M5
                tM1 = M1
                tM2 = M2
                tM3 = M3 + target_job_original[0]
                tM4 = M4
                tM5 = M5
            elif (priorityMachine == 4):  # machine이 2인 경우
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4 + assigned_job_original[0]
                aM5 = M5
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4 + target_job_original[0]
                tM5 = M5
            elif (priorityMachine == 5):  # machine이 2인 경우
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5 + assigned_job_original[0]
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5 + target_job_original[0]

            if (aM1 >= aM2 and aM1 >= aM3 and aM1 >= aM4 and aM1 >= aM5):
                aMakespan = aM1
                dMakespan = aM1 - Makespan
            elif (aM2 >= aM1 and aM2 >= aM3 and aM2 >= aM4 and aM2 >= aM5):
                aMakespan = aM2
                dMakespan = aM2 - Makespan
            elif (aM3 >= aM1 and aM3 >= aM2 and aM3 >= aM4 and aM3 >= aM5):
                aMakespan = aM3
                dMakespan = aM3 - Makespan
            elif (aM4 >= aM1 and aM4 >= aM2 and aM4 >= aM3 and aM4 >= aM5):
                aMakespan = aM4
                dMakespan = aM4 - Makespan
            elif (aM5 >= aM1 and aM5 >= aM2 and aM5 >= aM3 and aM5 >= aM4):
                aMakespan = aM5
                dMakespan = aM5 - Makespan

            if (tM1 >= tM2 and tM1 >= tM3 and tM1 >= tM4 and tM1 >= tM5):
                tMakespan = tM1
            elif (tM2 >= tM1 and tM2 >= tM3 and tM2 >= tM4 and tM2 >= tM5):
                tMakespan = tM2
            elif (tM3 >= tM1 and tM3 >= tM2 and tM3 >= tM4 and tM3 >= tM5):
                tMakespan = tM3
            elif (tM4 >= tM1 and tM4 >= tM2 and tM4 >= tM3 and tM4 >= tM5):
                tMakespan = tM4
            elif (tM5 >= tM1 and tM5 >= tM2 and tM5 >= tM3 and tM5 >= tM4):
                tMakespan = tM5

            ### Weighted Completion Time
            temp = Machine_Job_table
            temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
            temp['WCP'] = temp['P_cumsum'] * temp['W']

            sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()
            sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
            sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]
            sumWCP3 = sumWCP[sumWCP['MACHINEID'] == 3]
            sumWCP4 = sumWCP[sumWCP['MACHINEID'] == 4]
            sumWCP5 = sumWCP[sumWCP['MACHINEID'] == 5]

            if (sumWCP1.empty):
                wc1 = 0
            else:
                wc1 = sumWCP1.iloc[0, 8]
            if (sumWCP2.empty):
                wc2 = 0
            else:
                wc2 = sumWCP2.iloc[0, 8]
            if (sumWCP3.empty):
                wc3 = 0
            else:
                wc3 = sumWCP3.iloc[0, 8]
            if (sumWCP4.empty):
                wc4 = 0
            else:
                wc4 = sumWCP4.iloc[0, 8]
            if (sumWCP5.empty):
                wc5 = 0
            else:
                wc5 = sumWCP5.iloc[0, 8]
            wc = wc1 + wc2 + wc3 + wc4 + wc5

            if (priorityMachine == 1):
                awc1 = wc1 + aM1 * assigned_job_original[1]
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                twc1 = wc1 + tM1 * target_job_original[1]
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
            elif (priorityMachine == 2):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2 + aM2 * assigned_job_original[1]
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                twc1 = wc1
                twc2 = wc2 + tM2 * target_job_original[1]
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
            elif (priorityMachine == 3):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3 + aM3 * assigned_job_original[1]
                awc4 = wc4
                awc5 = wc5
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3 + tM3 * target_job_original[1]
                twc4 = wc4
                twc5 = wc5
            elif (priorityMachine == 4):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4 + aM4 * assigned_job_original[1]
                awc5 = wc5
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4 + tM4 * target_job_original[1]
                twc5 = wc5
            elif (priorityMachine == 5):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5 + aM5 * assigned_job_original[1]
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5 + tM5 * target_job_original[1]

            awc = awc1 + awc2 + awc3 + awc4 + awc5
            twc = twc1 + twc2 + twc3 + twc4 + twc5
        '''고치기'''
        machine_now = pd.Series([M1, M2, M3, M4, M5, wc1, wc2, wc3, wc4, wc5, wc, Makespan, SkT_M1, SkT_M2, SkT_M3, SkT_M4, SkT_M5],
                                index=['M1', 'M2', 'M3', 'M4', 'M5', "wc1", "wc2", 'wc3', 'wc4', 'wc5', "wc", "Makespan", "Slack_M1", "Slack_M2", "Slack_M3", "Slack_M4", "Slack_M5"])

        compare_Attr_A = pd.Series([aM1, aM2, aM3, aM4, aM5, aMakespan, awc1, awc2, awc3, awc4, awc5, awc],
                                   index=['assign_M1', 'assign_M2', 'assign_M3', 'assign_M4', 'assign_M5', 'assign_Makespan',
                                          'assign_WC1', 'assign_WC2', 'assign_WC3', 'assign_WC4', 'assign_WC5',
                                          "assign_WC"])
        compare_Attr_B = pd.Series([tM1, tM2, tM3, tM4, tM5, tMakespan, twc1, twc2, twc3, twc4, twc5, twc],
                                   index=['target_M1', 'target_M2', 'target_M3', 'target_M4', 'target_M5', 'target_Makespan',
                                          'target_WC1', 'target_WC2', 'target_WC3', 'target_WC4', 'target_WC5',
                                          'target_WC'])

        compare_Attr = pd.Series([dMakespan], index=["dMakespan"])
        ''''''
        compare_AB_Diff = compare_Attr_A.reset_index(drop=True) - compare_Attr_B.reset_index(drop=True)
        compare_AB_Diff.index = ["M1_AB_Diff", "M2_AB_Diff", "M3_AB_Diff", "M4_AB_Diff", "M5_AB_Diff", "Makespan_AB_Diff",
                                      "wc1_Diff", "wc2_Diff", "wc3_Diff", "wc4_Diff", "wc5_Diff", "wc_Diff"]

        if (flag == 0):
            KPIInfo_Attr = machine_now.append(compare_Attr_A)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr_B)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            insertAttributes = jobA_Attr.append(jobB_Attr)
            insertAttributes = insertAttributes.append(KPIInfo_Attr)
        elif (flag == 1):
            KPIInfo_Attr = machine_now.append(compare_Attr)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_AB_Diff)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]

            insertAttributes = jobAB_Diff.append(KPIInfo_Attr)
        elif (flag == 2):
            KPIInfo_Attr = machine_now.append(compare_Attr)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            compare_Attr_AB_Flag = copy.deepcopy(compare_AB_Diff)
            compare_Attr_AB_Flag = Set_Value_By_ComparisionFlag(compare_Attr_AB_Flag)
            compare_Attr_AB_Flag.index = ["M1_AB_Flag", "M2_AB_Flag", "M3_AB_Flag", "M4_AB_Flag", "M5_AB_Flag", "Makespan_AB_Flag",
                                          "wc1_Flag", "wc2_Flag", "wc3_Flag", "wc4_Flag", "wc5_Flag",
                                          "wc_Flag"]

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Flag = copy.deepcopy(jobAB_Diff)
            jobAB_Flag = Set_Value_By_ComparisionFlag(jobAB_Flag)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]
            jobAB_Flag.index = ["P_AB_Flag", "W_AB_Flag"]

            insertAttributes = jobAB_Flag.append(KPIInfo_Attr)
            insertAttributes = insertAttributes.append(compare_Attr_AB_Flag)
        elif (flag == 3):
            KPIInfo_Attr = machine_now.append(compare_Attr_A)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr_B)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_AB_Diff)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            compare_Attr_AB_Flag = copy.deepcopy(compare_AB_Diff)
            compare_Attr_AB_Flag = Set_Value_By_ComparisionFlag(compare_Attr_AB_Flag)
            compare_Attr_AB_Flag.index = ["M1_AB_Flag", "M2_AB_Flag", "M3_AB_Flag", "M4_AB_Flag", "M5_AB_Flag",
                                          "Makespan_AB_Flag",
                                          "wc1_Flag", "wc2_Flag", "wc3_Flag", "wc4_Flag", "wc5_Flag", "wc_Flag"]

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Flag = copy.deepcopy(jobAB_Diff)
            jobAB_Flag = Set_Value_By_ComparisionFlag(jobAB_Flag)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]
            jobAB_Flag.index = ["P_AB_Flag", "W_AB_Flag"]

            insertAttributes = jobA_Attr.append(jobB_Attr)
            insertAttributes = insertAttributes.append(jobAB_Diff)
            insertAttributes = insertAttributes.append(jobAB_Flag)
            insertAttributes = insertAttributes.append(KPIInfo_Attr)
            insertAttributes = insertAttributes.append(compare_Attr_AB_Flag)

    elif (machineNum == 10):
        ## 머신이 할당되기전 머신 KPI
        if (Machine_Job_table.empty):
            M1 = 0
            M2 = 0
            M3 = 0
            M4 = 0
            M5 = 0
            M6 = 0
            M7 = 0
            M8 = 0
            M9 = 0
            M10 = 0

            if (M1 >= M2 and M1 >= M3 and M1 >= M4 and M1 >= M5 and
                    M1 >= M6 and M1 >= M7 and M1 >= M8 and M1 >= M9 and M1 >= M10):
                Makespan = M1
            elif (M2 >= M1 and M2 >= M3 and M2 >= M4 and M2 >= M5 and
                  M2 >= M6 and M2 >= M7 and M2 >= M8 and M2 >= M9 and M2 >= M10):
                Makespan = M2
            elif (M3 >= M1 and M3 >= M2 and M3 >= M4 and M3 >= M5 and
                  M3 >= M6 and M3 >= M7 and M3 >= M8 and M3 >= M9 and M3 >= M10):
                Makespan = M3
            elif (M4 >= M1 and M4 >= M2 and M4 >= M3 and M4 >= M5 and
                  M4 >= M6 and M4 >= M7 and M4 >= M8 and M4 >= M9 and M4 >= M10):
                Makespan = M4
            elif (M5 >= M1 and M5 >= M2 and M5 >= M3 and M5 >= M4 and
                  M5 >= M6 and M5 >= M7 and M5 >= M8 and M5 >= M9 and M5 >= M10):
                Makespan = M5
            elif (M6 >= M1 and M6 >= M2 and M6 >= M3 and M6 >= M4 and
                  M6 >= M5 and M6 >= M7 and M6 >= M8 and M6 >= M9 and M6 >= M10):
                Makespan = M6
            elif (M7 >= M1 and M7 >= M2 and M7 >= M3 and M7 >= M4 and
                  M7 >= M5 and M7 >= M6 and M7 >= M8 and M7 >= M9 and M7 >= M10):
                Makespan = M7
            elif (M8 >= M1 and M8 >= M2 and M8 >= M3 and M8 >= M4 and
                  M8 >= M5 and M8 >= M6 and M8 >= M7 and M8 >= M9 and M8 >= M10):
                Makespan = M8
            elif (M9 >= M1 and M9 >= M2 and M9 >= M3 and M9 >= M4 and
                  M9 >= M5 and M9 >= M6 and M9 >= M7 and M9 >= M8 and M9 >= M10):
                Makespan = M9
            elif (M10 >= M1 and M10 >= M2 and M10 >= M3 and M10 >= M4 and
                  M10 >= M5 and M10 >= M6 and M10 >= M7 and M10 >= M8 and M10 >= M9):
                Makespan = M10

            aM1 = 0
            aM2 = 0
            aM3 = 0
            aM4 = 0
            aM5 = 0
            aM6 = 0
            aM7 = 0
            aM8 = 0
            aM9 = 0
            aM10 = 0
            tM1 = 0
            tM2 = 0
            tM3 = 0
            tM4 = 0
            tM5 = 0
            tM6 = 0
            tM7 = 0
            tM8 = 0
            tM9 = 0
            tM10 = 0
            aMakespan = 0
            tMakespan = 0
            dMakespan = 0

            wc1 = 0
            wc2 = 0
            wc3 = 0
            wc4 = 0
            wc5 = 0
            wc6 = 0
            wc7 = 0
            wc8 = 0
            wc9 = 0
            wc10 = 0
            wc = 0
            awc1 = 0
            awc2 = 0
            awc3 = 0
            awc4 = 0
            awc5 = 0
            awc6 = 0
            awc7 = 0
            awc8 = 0
            awc9 = 0
            awc10 = 0
            twc1 = 0
            twc2 = 0
            twc3 = 0
            twc4 = 0
            twc5 = 0
            twc6 = 0
            twc7 = 0
            twc8 = 0
            twc9 = 0
            twc10 = 0
            awc = 0
            twc = 0

            SkT_M1 = 0
            SkT_M2 = 0
            SkT_M3 = 0
            SkT_M4 = 0
            SkT_M5 = 0
            SkT_M6 = 0
            SkT_M7 = 0
            SkT_M8 = 0
            SkT_M9 = 0
            SkT_M10 = 0

        ## 머신이 할당된 후 머신 KPI
        else:
            ### Makespan
            sumResult = Machine_Job_table.groupby(['MACHINEID'], as_index=False).sum()

            if (sumResult[sumResult['MACHINEID'] == 1].empty):
                M1 = 0
            else:
                M1 = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 2].empty):
                M2 = 0
            else:
                M2 = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 3].empty):
                M3 = 0
            else:
                M3 = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 4].empty):
                M4 = 0
            else:
                M4 = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 5].empty):
                M5 = 0
            else:
                M5 = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 6].empty):
                M6 = 0
            else:
                M6 = sumResult[sumResult['MACHINEID'] == 6].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 7].empty):
                M7 = 0
            else:
                M7 = sumResult[sumResult['MACHINEID'] == 7].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 8].empty):
                M8 = 0
            else:
                M8 = sumResult[sumResult['MACHINEID'] == 8].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 9].empty):
                M9 = 0
            else:
                M9 = sumResult[sumResult['MACHINEID'] == 9].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 10].empty):
                M10 = 0
            else:
                M10 = sumResult[sumResult['MACHINEID'] == 10].P.iloc[0]

            if (M1 >= M2 and M1 >= M3 and M1 >= M4 and M1 >= M5 and
                    M1 >= M6 and M1 >= M7 and M1 >= M8 and M1 >= M9 and M1 >= M10):
                Makespan = M1
            elif (M2 >= M1 and M2 >= M3 and M2 >= M4 and M2 >= M5 and
                  M2 >= M6 and M2 >= M7 and M2 >= M8 and M2 >= M9 and M2 >= M10):
                Makespan = M2
            elif (M3 >= M1 and M3 >= M2 and M3 >= M4 and M3 >= M5 and
                  M3 >= M6 and M3 >= M7 and M3 >= M8 and M3 >= M9 and M3 >= M10):
                Makespan = M3
            elif (M4 >= M1 and M4 >= M2 and M4 >= M3 and M4 >= M5 and
                  M4 >= M6 and M4 >= M7 and M4 >= M8 and M4 >= M9 and M4 >= M10):
                Makespan = M4
            elif (M5 >= M1 and M5 >= M2 and M5 >= M3 and M5 >= M4 and
                  M5 >= M6 and M5 >= M7 and M5 >= M8 and M5 >= M9 and M5 >= M10):
                Makespan = M5
            elif (M6 >= M1 and M6 >= M2 and M6 >= M3 and M6 >= M4 and
                  M6 >= M5 and M6 >= M7 and M6 >= M8 and M6 >= M9 and M6 >= M10):
                Makespan = M6
            elif (M7 >= M1 and M7 >= M2 and M7 >= M3 and M7 >= M4 and
                  M7 >= M5 and M7 >= M6 and M7 >= M8 and M7 >= M9 and M7 >= M10):
                Makespan = M7
            elif (M8 >= M1 and M8 >= M2 and M8 >= M3 and M8 >= M4 and
                  M8 >= M5 and M8 >= M6 and M8 >= M7 and M8 >= M9 and M8 >= M10):
                Makespan = M8
            elif (M9 >= M1 and M9 >= M2 and M9 >= M3 and M9 >= M4 and
                  M9 >= M5 and M9 >= M6 and M9 >= M7 and M9 >= M8 and M9 >= M10):
                Makespan = M9
            elif (M10 >= M1 and M10 >= M2 and M10 >= M3 and M10 >= M4 and
                  M10 >= M5 and M10 >= M6 and M10 >= M7 and M10 >= M8 and M10 >= M9):
                Makespan = M10

            SkT_M1 = Makespan - M1
            SkT_M2 = Makespan - M2
            SkT_M3 = Makespan - M3
            SkT_M4 = Makespan - M4
            SkT_M5 = Makespan - M5
            SkT_M6 = Makespan - M6
            SkT_M7 = Makespan - M7
            SkT_M8 = Makespan - M8
            SkT_M9 = Makespan - M9
            SkT_M10 = Makespan - M10

            if (priorityMachine == 1):
                aM1 = M1 + assigned_job_original[0]
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                tM1 = M1 + target_job_original[0]
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
            elif (priorityMachine == 2):  # machine이 2인 경우
                aM1 = M1
                aM2 = M2 + assigned_job_original[0]
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                tM1 = M1
                tM2 = M2 + target_job_original[0]
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
            elif (priorityMachine == 3):
                aM1 = M1
                aM2 = M2
                aM3 = M3 + assigned_job_original[0]
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                tM1 = M1
                tM2 = M2
                tM3 = M3 + target_job_original[0]
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
            elif (priorityMachine == 4):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4 + assigned_job_original[0]
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4 + target_job_original[0]
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
            elif (priorityMachine == 5):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5 + assigned_job_original[0]
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5 + target_job_original[0]
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
            elif (priorityMachine == 6):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6 + assigned_job_original[0]
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6 + target_job_original[0]
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
            elif (priorityMachine == 7):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7 + assigned_job_original[0]
                aM8 = M8
                aM9 = M9
                aM10 = M10
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7 + target_job_original[0]
                tM8 = M8
                tM9 = M9
                tM10 = M10
            elif (priorityMachine == 8):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8 + assigned_job_original[0]
                aM9 = M9
                aM10 = M10
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8 + target_job_original[0]
                tM9 = M9
                tM10 = M10
            elif (priorityMachine == 9):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9 + assigned_job_original[0]
                aM10 = M10
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9 + target_job_original[0]
                tM10 = M10
            elif (priorityMachine == 10):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10 + assigned_job_original[0]
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10 + target_job_original[0]

            if (aM1 >= aM2 and aM1 >= aM3 and aM1 >= aM4 and aM1 >= aM5 and
                    aM1 >= aM6 and aM1 >= aM7 and aM1 >= aM8 and aM1 >= aM9 and aM1 >= aM10):
                aMakespan = aM1
                dMakespan = aM1 - Makespan
            elif (aM2 >= aM1 and aM2 >= aM3 and aM2 >= aM4 and aM2 >= aM5 and
                  aM2 >= aM6 and aM2 >= aM7 and aM2 >= aM8 and aM2 >= aM9 and aM2 >= aM10):
                aMakespan = aM2
                dMakespan = aM2 - Makespan
            elif (aM3 >= aM1 and aM3 >= aM2 and aM3 >= aM4 and aM3 >= aM5 and
                  aM3 >= aM6 and aM3 >= aM7 and aM3 >= aM8 and aM3 >= aM9 and aM3 >= aM10):
                aMakespan = aM3
                dMakespan = aM3 - Makespan
            elif (aM4 >= aM1 and aM4 >= aM2 and aM4 >= aM3 and aM4 >= aM5 and
                  aM4 >= aM6 and aM4 >= aM7 and aM4 >= aM8 and aM4 >= aM9 and aM4 >= aM10):
                aMakespan = aM4
                dMakespan = aM4 - Makespan
            elif (aM5 >= aM1 and aM5 >= aM2 and aM5 >= aM3 and aM5 >= aM4 and
                  aM5 >= aM6 and aM5 >= aM7 and aM5 >= aM8 and aM5 >= aM9 and aM5 >= aM10):
                aMakespan = aM5
                dMakespan = aM5 - Makespan
            elif (aM6 >= aM1 and aM6 >= aM2 and aM6 >= aM3 and aM6 >= aM4 and
                  aM6 >= aM5 and aM6 >= aM7 and aM6 >= aM8 and aM6 >= aM9 and aM6 >= aM10):
                aMakespan = aM6
                dMakespan = aM6 - Makespan
            elif (aM7 >= aM1 and aM7 >= aM2 and aM7 >= aM3 and aM7 >= aM4 and
                  aM7 >= aM5 and aM7 >= aM6 and aM7 >= aM8 and aM7 >= aM9 and aM7 >= aM10):
                aMakespan = aM7
                dMakespan = aM7 - Makespan
            elif (aM8 >= aM1 and aM8 >= aM2 and aM8 >= aM3 and aM8 >= aM4 and
                  aM8 >= aM5 and aM8 >= aM6 and aM8 >= aM7 and aM8 >= aM9 and aM8 >= aM10):
                aMakespan = aM8
                dMakespan = aM8 - Makespan
            elif (aM9 >= aM1 and aM9 >= aM2 and aM9 >= aM3 and aM9 >= aM4 and
                  aM9 >= aM5 and aM9 >= aM6 and aM9 >= aM7 and aM9 >= aM8 and aM9 >= aM10):
                aMakespan = aM9
                dMakespan = aM9 - Makespan
            elif (aM10 >= aM1 and aM10 >= aM2 and aM10 >= aM3 and aM10 >= aM4 and
                  aM10 >= aM5 and aM10 >= aM6 and aM10 >= aM7 and aM10 >= aM8 and aM10 >= aM9):
                aMakespan = aM10
                dMakespan = aM10 - Makespan

            if (tM1 >= tM2 and tM1 >= tM3 and tM1 >= tM4 and tM1 >= tM5 and
                tM1 >= tM6 and tM1 >= tM7 and tM1 >= tM8 and tM1 >= tM9 and tM1 >= tM10):
                tMakespan = tM1
            elif (tM2 >= tM1 and tM2 >= tM3 and tM2 >= tM4 and tM2 >= tM5 and
                tM2 >= tM6 and tM2 >= tM7 and tM2 >= tM8 and tM2 >= tM9 and tM2 >= tM10):
                tMakespan = tM2
            elif (tM3 >= tM1 and tM3 >= tM2 and tM3 >= tM4 and tM3 >= tM5 and
                tM3 >= tM6 and tM3 >= tM7 and tM3 >= tM8 and tM3 >= tM9 and tM3 >= tM10):
                tMakespan = tM3
            elif (tM4 >= tM1 and tM4 >= tM2 and tM4 >= tM3 and tM4 >= tM5 and
                tM4 >= tM6 and tM4 >= tM7 and tM4 >= tM8 and tM4 >= tM9 and tM4 >= tM10):
                tMakespan = tM4
            elif (tM5 >= tM1 and tM5 >= tM2 and tM5 >= tM3 and tM5 >= tM4 and
                tM5 >= tM6 and tM5 >= tM7 and tM5 >= tM8 and tM5 >= tM9 and tM5 >= tM10):
                tMakespan = tM5
            elif (tM6 >= tM1 and tM6 >= tM2 and tM6 >= tM3 and tM6 >= tM4 and
                tM6 >= tM5 and tM6 >= tM7 and tM6 >= tM8 and tM6 >= tM9 and tM6 >= tM10):
                tMakespan = tM6
            elif (tM7 >= tM1 and tM7 >= tM2 and tM7 >= tM3 and tM7 >= tM4 and
                tM7 >= tM5 and tM7 >= tM6 and tM7 >= tM8 and tM7 >= tM9 and tM7 >= tM10):
                tMakespan = tM7
            elif (tM8 >= tM1 and tM8 >= tM2 and tM8 >= tM3 and tM8 >= tM4 and
                tM8 >= tM5 and tM8 >= tM6 and tM8 >= tM7 and tM8 >= tM9 and tM8 >= tM10):
                tMakespan = tM8
            elif (tM9 >= tM1 and tM9 >= tM2 and tM9 >= tM3 and tM9 >= tM4 and
                tM9 >= tM5 and tM9 >= tM6 and tM9 >= tM7 and tM9 >= tM8 and tM9 >= tM10):
                tMakespan = tM9
            elif (tM10 >= tM1 and tM10 >= tM2 and tM10 >= tM3 and tM10 >= tM4 and
                tM10 >= tM5 and tM10 >= tM6 and tM10 >= tM7 and tM10 >= tM8 and tM10 >= tM9):
                tMakespan = tM10

            ### Weighted Completion Time
            temp = Machine_Job_table
            temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
            temp['WCP'] = temp['P_cumsum'] * temp['W']

            sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()
            sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
            sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]
            sumWCP3 = sumWCP[sumWCP['MACHINEID'] == 3]
            sumWCP4 = sumWCP[sumWCP['MACHINEID'] == 4]
            sumWCP5 = sumWCP[sumWCP['MACHINEID'] == 5]
            sumWCP6 = sumWCP[sumWCP['MACHINEID'] == 6]
            sumWCP7 = sumWCP[sumWCP['MACHINEID'] == 7]
            sumWCP8 = sumWCP[sumWCP['MACHINEID'] == 8]
            sumWCP9 = sumWCP[sumWCP['MACHINEID'] == 9]
            sumWCP10 = sumWCP[sumWCP['MACHINEID'] == 10]

            if (sumWCP1.empty):
                wc1 = 0
            else:
                wc1 = sumWCP1.iloc[0, 8]
            if (sumWCP2.empty):
                wc2 = 0
            else:
                wc2 = sumWCP2.iloc[0, 8]
            if (sumWCP3.empty):
                wc3 = 0
            else:
                wc3 = sumWCP3.iloc[0, 8]
            if (sumWCP4.empty):
                wc4 = 0
            else:
                wc4 = sumWCP4.iloc[0, 8]
            if (sumWCP5.empty):
                wc5 = 0
            else:
                wc5 = sumWCP5.iloc[0, 8]
            if (sumWCP6.empty):
                wc6 = 0
            else:
                wc6 = sumWCP6.iloc[0, 8]
            if (sumWCP7.empty):
                wc7 = 0
            else:
                wc7 = sumWCP7.iloc[0, 8]
            if (sumWCP8.empty):
                wc8 = 0
            else:
                wc8 = sumWCP8.iloc[0, 8]
            if (sumWCP9.empty):
                wc9 = 0
            else:
                wc9 = sumWCP9.iloc[0, 8]
            if (sumWCP10.empty):
                wc10 = 0
            else:
                wc10 = sumWCP10.iloc[0, 8]
            wc = wc1 + wc2 + wc3 + wc4 + wc5 + wc6 + wc7 + wc8 + wc9 + wc10

            if (priorityMachine == 1):
                awc1 = wc1 + aM1 * assigned_job_original[1]
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                twc1 = wc1 + tM1 * target_job_original[1]
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
            elif (priorityMachine == 2):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2 + aM2 * assigned_job_original[1]
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                twc1 = wc1
                twc2 = wc2 + tM2 * target_job_original[1]
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
            elif (priorityMachine == 3):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3 + aM3 * assigned_job_original[1]
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3 + tM3 * target_job_original[1]
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
            elif (priorityMachine == 4):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4 + aM4 * assigned_job_original[1]
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4 + tM4 * target_job_original[1]
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
            elif (priorityMachine == 5):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5 + aM5 * assigned_job_original[1]
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5 + tM5 * target_job_original[1]
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
            elif (priorityMachine == 6):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6 + aM6 * assigned_job_original[1]
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6 + tM6 * target_job_original[1]
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
            elif (priorityMachine == 7):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7 + aM7 * assigned_job_original[1]
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7 + tM7 * target_job_original[1]
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
            elif (priorityMachine == 8):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8 + aM8 * assigned_job_original[1]
                awc9 = wc9
                awc10 = wc10
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8 + tM8 * target_job_original[1]
                twc9 = wc9
                twc10 = wc10
            elif (priorityMachine == 9):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9 + aM9 * assigned_job_original[1]
                awc10 = wc10
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9 + tM9 * target_job_original[1]
                twc10 = wc10
            elif (priorityMachine == 10):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10 + aM10 * assigned_job_original[1]
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10 + tM10 * target_job_original[1]

            awc = awc1 + awc2 + awc3 + awc4 + awc5 + awc6 + awc7 + awc8 + awc9 + awc10
            twc = twc1 + twc2 + twc3 + twc4 + twc5 + twc6 + twc7 + twc8 + twc9 + twc10
        '''고치기'''
        machine_now = pd.Series(
            [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, wc1, wc2, wc3, wc4, wc5, wc6, wc7, wc8, wc9, wc10, wc,
             Makespan, SkT_M1, SkT_M2, SkT_M3, SkT_M4, SkT_M5, SkT_M6, SkT_M7, SkT_M8, SkT_M9, SkT_M10],
            index=['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', "wc1", "wc2", 'wc3', 'wc4', 'wc5',
                   'wc6', 'wc7', 'wc8', 'wc9', 'wc10', "wc", "Makespan",
                   "Slack_M1", "Slack_M2", "Slack_M3", "Slack_M4", "Slack_M5",
                   "Slack_M6", "Slack_M7", "Slack_M8", "Slack_M9", "Slack_M10"])

        compare_Attr_A = pd.Series([aM1, aM2, aM3, aM4, aM5, aM6, aM7, aM8, aM9, aM10, aMakespan,
                                    awc1, awc2, awc3, awc4, awc5, awc6, awc7, awc8, awc9, awc10, awc],
                                   index=['assign_M1', 'assign_M2', 'assign_M3', 'assign_M4', 'assign_M5',
                                          'assign_M6', 'assign_M7', 'assign_M8', 'assign_M9', 'assign_M10',
                                          'assign_Makespan',
                                          'assign_WC1', 'assign_WC2', 'assign_WC3', 'assign_WC4', 'assign_WC5',
                                          'assign_WC6', 'assign_WC7', 'assign_WC8', 'assign_WC9', 'assign_WC10',
                                          "assign_WC"])
        compare_Attr_B = pd.Series([tM1, tM2, tM3, tM4, tM5, tM6, tM7, tM8, tM9, tM10, tMakespan,
                                    twc1, twc2, twc3, twc4, twc5, twc6, twc7, twc8, twc9, twc10, twc],
                                   index=['target_M1', 'target_M2', 'target_M3', 'target_M4', 'target_M5',
                                          'target_M6', 'target_M7', 'target_M8', 'target_M9', 'target_M10',
                                          'target_Makespan',
                                          'target_WC1', 'target_WC2', 'target_WC3', 'target_WC4', 'target_WC5',
                                          'target_WC6', 'target_WC7', 'target_WC8', 'target_WC9', 'target_WC10',
                                          'target_WC'])

        compare_Attr = pd.Series([dMakespan], index=["dMakespan"])
        ''''''
        compare_AB_Diff = compare_Attr_A.reset_index(drop=True) - compare_Attr_B.reset_index(drop=True)
        compare_AB_Diff.index = ["M1_AB_Diff", "M2_AB_Diff", "M3_AB_Diff", "M4_AB_Diff", "M5_AB_Diff",
                                      "M6_AB_Diff", "M7_AB_Diff", "M8_AB_Diff", "M9_AB_Diff", "M10_AB_Diff",
                                      "Makespan_AB_Diff",
                                      "wc1_Diff", "wc2_Diff", "wc3_Diff", "wc4_Diff", "wc5_Diff",
                                      "wc6_Diff", "wc7_Diff", "wc8_Diff", "wc9_Diff", "wc10_Diff", "wc_Diff"]

        if (flag == 0):
            KPIInfo_Attr = machine_now.append(compare_Attr_A)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr_B)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            insertAttributes = jobA_Attr.append(jobB_Attr)
            insertAttributes = insertAttributes.append(KPIInfo_Attr)
        elif (flag == 1):
            KPIInfo_Attr = machine_now.append(compare_Attr)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_AB_Diff)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]

            insertAttributes = jobAB_Diff.append(KPIInfo_Attr)
        elif (flag == 2):
            KPIInfo_Attr = machine_now.append(compare_Attr)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            compare_Attr_AB_Flag = copy.deepcopy(compare_AB_Diff)
            compare_Attr_AB_Flag = Set_Value_By_ComparisionFlag(compare_Attr_AB_Flag)
            compare_Attr_AB_Flag.index = ["M1_AB_Flag", "M2_AB_Flag", "M3_AB_Flag", "M4_AB_Flag", "M5_AB_Flag",
                                      "M6_AB_Flag", "M7_AB_Flag", "M8_AB_Flag", "M9_AB_Flag", "M10_AB_Flag",
                                      "Makespan_AB_Flag",
                                      "wc1_Flag", "wc2_Flag", "wc3_Flag", "wc4_Flag", "wc5_Flag",
                                      "wc6_Flag", "wc7_Flag", "wc8_Flag", "wc9_Flag", "wc10_Flag", "wc_Flag"]

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Flag = copy.deepcopy(jobAB_Diff)
            jobAB_Flag = Set_Value_By_ComparisionFlag(jobAB_Flag)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]
            jobAB_Flag.index = ["P_AB_Flag", "W_AB_Flag"]

            insertAttributes = jobAB_Flag.append(KPIInfo_Attr)
            insertAttributes = insertAttributes.append(compare_Attr_AB_Flag)
        elif (flag == 3):
            KPIInfo_Attr = machine_now.append(compare_Attr_A)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr_B)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_AB_Diff)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            compare_Attr_AB_Flag = copy.deepcopy(compare_AB_Diff)
            compare_Attr_AB_Flag = Set_Value_By_ComparisionFlag(compare_Attr_AB_Flag)
            compare_Attr_AB_Flag.index = ["M1_AB_Flag", "M2_AB_Flag", "M3_AB_Flag", "M4_AB_Flag", "M5_AB_Flag",
                                      "M6_AB_Flag", "M7_AB_Flag", "M8_AB_Flag", "M9_AB_Flag", "M10_AB_Flag",
                                      "Makespan_AB_Flag",
                                      "wc1_Flag", "wc2_Flag", "wc3_Flag", "wc4_Flag", "wc5_Flag",
                                      "wc6_Flag", "wc7_Flag", "wc8_Flag", "wc9_Flag", "wc10_Flag", "wc_Flag"]

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Flag = copy.deepcopy(jobAB_Diff)
            jobAB_Flag = Set_Value_By_ComparisionFlag(jobAB_Flag)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]
            jobAB_Flag.index = ["P_AB_Flag", "W_AB_Flag"]

            insertAttributes = jobA_Attr.append(jobB_Attr)
            insertAttributes = insertAttributes.append(jobAB_Diff)
            insertAttributes = insertAttributes.append(jobAB_Flag)
            insertAttributes = insertAttributes.append(KPIInfo_Attr)
            insertAttributes = insertAttributes.append(compare_Attr_AB_Flag)

    elif (machineNum == 15):
        ## 머신이 할당되기전 머신 KPI
        if (Machine_Job_table.empty):
            M1 = 0
            M2 = 0
            M3 = 0
            M4 = 0
            M5 = 0
            M6 = 0
            M7 = 0
            M8 = 0
            M9 = 0
            M10 = 0
            M11 = 0
            M12 = 0
            M13 = 0
            M14 = 0
            M15 = 0

            if (M1 >= M2 and M1 >= M3 and M1 >= M4 and M1 >= M5 and
                    M1 >= M6 and M1 >= M7 and M1 >= M8 and M1 >= M9 and M1 >= M10
                and M1 >= M11 and M1 >= M12 and M1 >= M13 and M1 >= M14 and M1 >= M15):
                Makespan = M1
            elif (M2 >= M1 and M2 >= M3 and M2 >= M4 and M2 >= M5 and
                  M2 >= M6 and M2 >= M7 and M2 >= M8 and M2 >= M9 and M2 >= M10
                and M2 >= M11 and M2 >= M12 and M2 >= M13 and M2 >= M14 and M2 >= M15):
                Makespan = M2
            elif (M3 >= M1 and M3 >= M2 and M3 >= M4 and M3 >= M5 and
                  M3 >= M6 and M3 >= M7 and M3 >= M8 and M3 >= M9 and M3 >= M10
                and M3 >= M11 and M3 >= M12 and M3 >= M13 and M3 >= M14 and M3 >= M15):
                Makespan = M3
            elif (M4 >= M1 and M4 >= M2 and M4 >= M3 and M4 >= M5 and
                  M4 >= M6 and M4 >= M7 and M4 >= M8 and M4 >= M9 and M4 >= M10
                and M4 >= M11 and M4 >= M12 and M4 >= M13 and M4 >= M14 and M4 >= M15):
                Makespan = M4
            elif (M5 >= M1 and M5 >= M2 and M5 >= M3 and M5 >= M4 and
                  M5 >= M6 and M5 >= M7 and M5 >= M8 and M5 >= M9 and M5 >= M10
                and M5 >= M11 and M5 >= M12 and M5 >= M13 and M5 >= M14 and M5 >= M15):
                Makespan = M5
            elif (M6 >= M1 and M6 >= M2 and M6 >= M3 and M6 >= M4 and
                  M6 >= M5 and M6 >= M7 and M6 >= M8 and M6 >= M9 and M6 >= M10
                and M6 >= M11 and M6 >= M12 and M6 >= M13 and M6 >= M14 and M6 >= M15):
                Makespan = M6
            elif (M7 >= M1 and M7 >= M2 and M7 >= M3 and M7 >= M4 and
                  M7 >= M5 and M7 >= M6 and M7 >= M8 and M7 >= M9 and M7 >= M10
                and M7 >= M11 and M7 >= M12 and M7 >= M13 and M7 >= M14 and M7 >= M15):
                Makespan = M7
            elif (M8 >= M1 and M8 >= M2 and M8 >= M3 and M8 >= M4 and
                  M8 >= M5 and M8 >= M6 and M8 >= M7 and M8 >= M9 and M8 >= M10
                and M8 >= M11 and M8 >= M12 and M8 >= M13 and M8 >= M14 and M8 >= M15):
                Makespan = M8
            elif (M9 >= M1 and M9 >= M2 and M9 >= M3 and M9 >= M4 and
                  M9 >= M5 and M9 >= M6 and M9 >= M7 and M9 >= M8 and M9 >= M10
                and M9 >= M11 and M9 >= M12 and M9 >= M13 and M9 >= M14 and M9 >= M15):
                Makespan = M9
            elif (M10 >= M1 and M10 >= M2 and M10 >= M3 and M10 >= M4 and
                  M10 >= M5 and M10 >= M6 and M10 >= M7 and M10 >= M8 and M10 >= M9
                and M10 >= M11 and M10 >= M12 and M10 >= M13 and M10 >= M14 and M10 >= M15):
                Makespan = M10
            elif (M11 >= M1 and M11 >= M2 and M11 >= M3 and M11 >= M4 and
                  M11 >= M5 and M11 >= M6 and M11 >= M7 and M11 >= M8 and M11 >= M9
                and M11 >= M10 and M11 >= M12 and M11 >= M13 and M11 >= M14 and M11 >= M15):
                Makespan = M11
            elif (M12 >= M1 and M12 >= M2 and M12 >= M3 and M12 >= M4 and
                  M12 >= M5 and M12 >= M6 and M12 >= M7 and M12 >= M8 and M12 >= M9
                and M12 >= M11 and M12 >= M10 and M12 >= M13 and M12 >= M14 and M12 >= M15):
                Makespan = M12
            elif (M13 >= M1 and M13 >= M2 and M13 >= M3 and M13 >= M4 and
                  M13 >= M5 and M13 >= M6 and M13 >= M7 and M13 >= M8 and M13 >= M9
                and M13 >= M11 and M13 >= M12 and M13 >= M10 and M13 >= M14 and M13 >= M15):
                Makespan = M13
            elif (M14 >= M1 and M14 >= M2 and M14 >= M3 and M14 >= M4 and
                  M14 >= M5 and M14 >= M6 and M14 >= M7 and M14 >= M8 and M14 >= M9
                and M14 >= M11 and M14 >= M12 and M14 >= M13 and M14 >= M10 and M14 >= M15):
                Makespan = M14
            elif (M15 >= M1 and M15 >= M2 and M15 >= M3 and M15 >= M4 and
                  M15 >= M5 and M15 >= M6 and M15 >= M7 and M15 >= M8 and M15 >= M9
                and M15 >= M11 and M15 >= M12 and M15 >= M13 and M15 >= M14 and M15 >= M10):
                Makespan = M15

            aM1 = 0
            aM2 = 0
            aM3 = 0
            aM4 = 0
            aM5 = 0
            aM6 = 0
            aM7 = 0
            aM8 = 0
            aM9 = 0
            aM10 = 0
            aM11 = 0
            aM12 = 0
            aM13 = 0
            aM14 = 0
            aM15 = 0
            tM1 = 0
            tM2 = 0
            tM3 = 0
            tM4 = 0
            tM5 = 0
            tM6 = 0
            tM7 = 0
            tM8 = 0
            tM9 = 0
            tM10 = 0
            tM11 = 0
            tM12 = 0
            tM13 = 0
            tM14 = 0
            tM15 = 0
            aMakespan = 0
            tMakespan = 0
            dMakespan = 0

            wc1 = 0
            wc2 = 0
            wc3 = 0
            wc4 = 0
            wc5 = 0
            wc6 = 0
            wc7 = 0
            wc8 = 0
            wc9 = 0
            wc10 = 0
            wc11 = 0
            wc12 = 0
            wc13 = 0
            wc14 = 0
            wc15 = 0
            wc = 0
            awc1 = 0
            awc2 = 0
            awc3 = 0
            awc4 = 0
            awc5 = 0
            awc6 = 0
            awc7 = 0
            awc8 = 0
            awc9 = 0
            awc10 = 0
            awc11 = 0
            awc12 = 0
            awc13 = 0
            awc14 = 0
            awc15 = 0
            twc1 = 0
            twc2 = 0
            twc3 = 0
            twc4 = 0
            twc5 = 0
            twc6 = 0
            twc7 = 0
            twc8 = 0
            twc9 = 0
            twc10 = 0
            twc11 = 0
            twc12 = 0
            twc13 = 0
            twc14 = 0
            twc15 = 0
            awc = 0
            twc = 0

            SkT_M1 = 0
            SkT_M2 = 0
            SkT_M3 = 0
            SkT_M4 = 0
            SkT_M5 = 0
            SkT_M6 = 0
            SkT_M7 = 0
            SkT_M8 = 0
            SkT_M9 = 0
            SkT_M10 = 0
            SkT_M11 = 0
            SkT_M12 = 0
            SkT_M13 = 0
            SkT_M14 = 0
            SkT_M15 = 0

        ## 머신이 할당된 후 머신 KPI
        else:
            ### Makespan
            sumResult = Machine_Job_table.groupby(['MACHINEID'], as_index=False).sum()

            if (sumResult[sumResult['MACHINEID'] == 1].empty):
                M1 = 0
            else:
                M1 = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 2].empty):
                M2 = 0
            else:
                M2 = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 3].empty):
                M3 = 0
            else:
                M3 = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 4].empty):
                M4 = 0
            else:
                M4 = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 5].empty):
                M5 = 0
            else:
                M5 = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 6].empty):
                M6 = 0
            else:
                M6 = sumResult[sumResult['MACHINEID'] == 6].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 7].empty):
                M7 = 0
            else:
                M7 = sumResult[sumResult['MACHINEID'] == 7].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 8].empty):
                M8 = 0
            else:
                M8 = sumResult[sumResult['MACHINEID'] == 8].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 9].empty):
                M9 = 0
            else:
                M9 = sumResult[sumResult['MACHINEID'] == 9].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 10].empty):
                M10 = 0
            else:
                M10 = sumResult[sumResult['MACHINEID'] == 10].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 11].empty):
                M11 = 0
            else:
                M11 = sumResult[sumResult['MACHINEID'] == 11].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 12].empty):
                M12 = 0
            else:
                M12 = sumResult[sumResult['MACHINEID'] == 12].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 13].empty):
                M13 = 0
            else:
                M13 = sumResult[sumResult['MACHINEID'] == 13].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 14].empty):
                M14 = 0
            else:
                M14 = sumResult[sumResult['MACHINEID'] == 14].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 15].empty):
                M15 = 0
            else:
                M15 = sumResult[sumResult['MACHINEID'] == 15].P.iloc[0]

            if (M1 >= M2 and M1 >= M3 and M1 >= M4 and M1 >= M5 and
                    M1 >= M6 and M1 >= M7 and M1 >= M8 and M1 >= M9 and M1 >= M10
                    and M1 >= M11 and M1 >= M12 and M1 >= M13 and M1 >= M14 and M1 >= M15):
                Makespan = M1
            elif (M2 >= M1 and M2 >= M3 and M2 >= M4 and M2 >= M5 and
                  M2 >= M6 and M2 >= M7 and M2 >= M8 and M2 >= M9 and M2 >= M10
                  and M2 >= M11 and M2 >= M12 and M2 >= M13 and M2 >= M14 and M2 >= M15):
                Makespan = M2
            elif (M3 >= M1 and M3 >= M2 and M3 >= M4 and M3 >= M5 and
                  M3 >= M6 and M3 >= M7 and M3 >= M8 and M3 >= M9 and M3 >= M10
                  and M3 >= M11 and M3 >= M12 and M3 >= M13 and M3 >= M14 and M3 >= M15):
                Makespan = M3
            elif (M4 >= M1 and M4 >= M2 and M4 >= M3 and M4 >= M5 and
                  M4 >= M6 and M4 >= M7 and M4 >= M8 and M4 >= M9 and M4 >= M10
                  and M4 >= M11 and M4 >= M12 and M4 >= M13 and M4 >= M14 and M4 >= M15):
                Makespan = M4
            elif (M5 >= M1 and M5 >= M2 and M5 >= M3 and M5 >= M4 and
                  M5 >= M6 and M5 >= M7 and M5 >= M8 and M5 >= M9 and M5 >= M10
                  and M5 >= M11 and M5 >= M12 and M5 >= M13 and M5 >= M14 and M5 >= M15):
                Makespan = M5
            elif (M6 >= M1 and M6 >= M2 and M6 >= M3 and M6 >= M4 and
                  M6 >= M5 and M6 >= M7 and M6 >= M8 and M6 >= M9 and M6 >= M10
                  and M6 >= M11 and M6 >= M12 and M6 >= M13 and M6 >= M14 and M6 >= M15):
                Makespan = M6
            elif (M7 >= M1 and M7 >= M2 and M7 >= M3 and M7 >= M4 and
                  M7 >= M5 and M7 >= M6 and M7 >= M8 and M7 >= M9 and M7 >= M10
                  and M7 >= M11 and M7 >= M12 and M7 >= M13 and M7 >= M14 and M7 >= M15):
                Makespan = M7
            elif (M8 >= M1 and M8 >= M2 and M8 >= M3 and M8 >= M4 and
                  M8 >= M5 and M8 >= M6 and M8 >= M7 and M8 >= M9 and M8 >= M10
                  and M8 >= M11 and M8 >= M12 and M8 >= M13 and M8 >= M14 and M8 >= M15):
                Makespan = M8
            elif (M9 >= M1 and M9 >= M2 and M9 >= M3 and M9 >= M4 and
                  M9 >= M5 and M9 >= M6 and M9 >= M7 and M9 >= M8 and M9 >= M10
                  and M9 >= M11 and M9 >= M12 and M9 >= M13 and M9 >= M14 and M9 >= M15):
                Makespan = M9
            elif (M10 >= M1 and M10 >= M2 and M10 >= M3 and M10 >= M4 and
                  M10 >= M5 and M10 >= M6 and M10 >= M7 and M10 >= M8 and M10 >= M9
                  and M10 >= M11 and M10 >= M12 and M10 >= M13 and M10 >= M14 and M10 >= M15):
                Makespan = M10
            elif (M11 >= M1 and M11 >= M2 and M11 >= M3 and M11 >= M4 and
                  M11 >= M5 and M11 >= M6 and M11 >= M7 and M11 >= M8 and M11 >= M9
                  and M11 >= M10 and M11 >= M12 and M11 >= M13 and M11 >= M14 and M11 >= M15):
                Makespan = M11
            elif (M12 >= M1 and M12 >= M2 and M12 >= M3 and M12 >= M4 and
                  M12 >= M5 and M12 >= M6 and M12 >= M7 and M12 >= M8 and M12 >= M9
                  and M12 >= M11 and M12 >= M10 and M12 >= M13 and M12 >= M14 and M12 >= M15):
                Makespan = M12
            elif (M13 >= M1 and M13 >= M2 and M13 >= M3 and M13 >= M4 and
                  M13 >= M5 and M13 >= M6 and M13 >= M7 and M13 >= M8 and M13 >= M9
                  and M13 >= M11 and M13 >= M12 and M13 >= M10 and M13 >= M14 and M13 >= M15):
                Makespan = M13
            elif (M14 >= M1 and M14 >= M2 and M14 >= M3 and M14 >= M4 and
                  M14 >= M5 and M14 >= M6 and M14 >= M7 and M14 >= M8 and M14 >= M9
                  and M14 >= M11 and M14 >= M12 and M14 >= M13 and M14 >= M10 and M14 >= M15):
                Makespan = M14
            elif (M15 >= M1 and M15 >= M2 and M15 >= M3 and M15 >= M4 and
                  M15 >= M5 and M15 >= M6 and M15 >= M7 and M15 >= M8 and M15 >= M9
                  and M15 >= M11 and M15 >= M12 and M15 >= M13 and M15 >= M14 and M15 >= M10):
                Makespan = M15

            SkT_M1 = Makespan - M1
            SkT_M2 = Makespan - M2
            SkT_M3 = Makespan - M3
            SkT_M4 = Makespan - M4
            SkT_M5 = Makespan - M5
            SkT_M6 = Makespan - M6
            SkT_M7 = Makespan - M7
            SkT_M8 = Makespan - M8
            SkT_M9 = Makespan - M9
            SkT_M10 = Makespan - M10
            SkT_M11 = Makespan - M11
            SkT_M12 = Makespan - M12
            SkT_M13 = Makespan - M13
            SkT_M14 = Makespan - M14
            SkT_M15 = Makespan - M15

            if (priorityMachine == 1):
                aM1 = M1 + assigned_job_original[0]
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1 + target_job_original[0]
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 2):  # machine이 2인 경우
                aM1 = M1
                aM2 = M2 + assigned_job_original[0]
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2 + target_job_original[0]
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 3):
                aM1 = M1
                aM2 = M2
                aM3 = M3 + assigned_job_original[0]
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3 + target_job_original[0]
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 4):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4 + assigned_job_original[0]
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4 + target_job_original[0]
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 5):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5 + assigned_job_original[0]
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5 + target_job_original[0]
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 6):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6 + assigned_job_original[0]
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6 + target_job_original[0]
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 7):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7 + assigned_job_original[0]
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7 + target_job_original[0]
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 8):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8 + assigned_job_original[0]
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8 + target_job_original[0]
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 9):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9 + assigned_job_original[0]
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9 + target_job_original[0]
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 10):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10 + assigned_job_original[0]
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10 + target_job_original[0]
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 11):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11 + assigned_job_original[0]
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11 + target_job_original[0]
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 12):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12 + assigned_job_original[0]
                aM13 = M13
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12 + target_job_original[0]
                tM13 = M13
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 13):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13 + assigned_job_original[0]
                aM14 = M14
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13 + target_job_original[0]
                tM14 = M14
                tM15 = M15
            elif (priorityMachine == 14):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14 + assigned_job_original[0]
                aM15 = M15
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14 + target_job_original[0]
                tM15 = M15
            elif (priorityMachine == 15):
                aM1 = M1
                aM2 = M2
                aM3 = M3
                aM4 = M4
                aM5 = M5
                aM6 = M6
                aM7 = M7
                aM8 = M8
                aM9 = M9
                aM10 = M10
                aM11 = M11
                aM12 = M12
                aM13 = M13
                aM14 = M14
                aM15 = M15 + assigned_job_original[0]
                tM1 = M1
                tM2 = M2
                tM3 = M3
                tM4 = M4
                tM5 = M5
                tM6 = M6
                tM7 = M7
                tM8 = M8
                tM9 = M9
                tM10 = M10
                tM11 = M11
                tM12 = M12
                tM13 = M13
                tM14 = M14
                tM15 = M15 + target_job_original[0]

            if (aM1 >= aM2 and aM1 >= aM3 and aM1 >= aM4 and aM1 >= aM5 and
                    aM1 >= aM6 and aM1 >= aM7 and aM1 >= aM8 and aM1 >= aM9 and aM1 >= aM10
                    and aM1 >= aM11 and aM1 >= aM12 and aM1 >= aM13 and aM1 >= aM14 and aM1 >= aM15):
                aMakespan = aM1
                dMakespan = aM1 - Makespan
            elif (aM2 >= aM1 and aM2 >= aM3 and aM2 >= aM4 and aM2 >= aM5 and
                  aM2 >= aM6 and aM2 >= aM7 and aM2 >= aM8 and aM2 >= aM9 and aM2 >= aM10
                  and aM2 >= aM11 and aM2 >= aM12 and aM2 >= aM13 and aM2 >= aM14 and aM2 >= aM15):
                aMakespan = aM2
                dMakespan = aM2 - Makespan
            elif (aM3 >= aM1 and aM3 >= aM2 and aM3 >= aM4 and aM3 >= aM5 and
                  aM3 >= aM6 and aM3 >= aM7 and aM3 >= aM8 and aM3 >= aM9 and aM3 >= aM10
                  and aM3 >= aM11 and aM3 >= aM12 and aM3 >= aM13 and aM3 >= aM14 and aM3 >= aM15):
                aMakespan = aM3
                dMakespan = aM3 - Makespan
            elif (aM4 >= aM1 and aM4 >= aM2 and aM4 >= aM3 and aM4 >= aM5 and
                  aM4 >= aM6 and aM4 >= aM7 and aM4 >= aM8 and aM4 >= aM9 and aM4 >= aM10
                  and aM4 >= aM11 and aM4 >= aM12 and aM4 >= aM13 and aM4 >= aM14 and aM4 >= aM15):
                aMakespan = aM4
                dMakespan = aM4 - Makespan
            elif (aM5 >= aM1 and aM5 >= aM2 and aM5 >= aM3 and aM5 >= aM4 and
                  aM5 >= aM6 and aM5 >= aM7 and aM5 >= aM8 and aM5 >= aM9 and aM5 >= aM10
                  and aM5 >= aM11 and aM5 >= aM12 and aM5 >= aM13 and aM5 >= aM14 and aM5 >= aM15):
                aMakespan = aM5
                dMakespan = aM5 - Makespan
            elif (aM6 >= aM1 and aM6 >= aM2 and aM6 >= aM3 and aM6 >= aM4 and
                  aM6 >= aM5 and aM6 >= aM7 and aM6 >= aM8 and aM6 >= aM9 and aM6 >= aM10
                  and aM6 >= aM11 and aM6 >= aM12 and aM6 >= aM13 and aM6 >= aM14 and aM6 >= aM15):
                aMakespan = aM6
                dMakespan = aM6 - Makespan
            elif (aM7 >= aM1 and aM7 >= aM2 and aM7 >= aM3 and aM7 >= aM4 and
                  aM7 >= aM5 and aM7 >= aM6 and aM7 >= aM8 and aM7 >= aM9 and aM7 >= aM10
                  and aM7 >= aM11 and aM7 >= aM12 and aM7 >= aM13 and aM7 >= aM14 and aM7 >= aM15):
                aMakespan = aM7
                dMakespan = aM7 - Makespan
            elif (aM8 >= aM1 and aM8 >= aM2 and aM8 >= aM3 and aM8 >= aM4 and
                  aM8 >= aM5 and aM8 >= aM6 and aM8 >= aM7 and aM8 >= aM9 and aM8 >= aM10
                  and aM8 >= aM11 and aM8 >= aM12 and aM8 >= aM13 and aM8 >= aM14 and aM8 >= aM15):
                aMakespan = aM8
                dMakespan = aM8 - Makespan
            elif (aM9 >= aM1 and aM9 >= aM2 and aM9 >= aM3 and aM9 >= aM4 and
                  aM9 >= aM5 and aM9 >= aM6 and aM9 >= aM7 and aM9 >= aM8 and aM9 >= aM10
                  and aM9 >= aM11 and aM9 >= aM12 and aM9 >= aM13 and aM9 >= aM14 and aM9 >= aM15):
                aMakespan = aM9
                dMakespan = aM9 - Makespan
            elif (aM10 >= aM1 and aM10 >= aM2 and aM10 >= aM3 and aM10 >= aM4 and
                  aM10 >= aM5 and aM10 >= aM6 and aM10 >= aM7 and aM10 >= aM8 and aM10 >= aM9
                  and aM10 >= aM11 and aM10 >= aM12 and aM10 >= aM13 and aM10 >= aM14 and aM10 >= aM15):
                aMakespan = aM10
                dMakespan = aM10 - Makespan
            elif (aM11 >= aM1 and aM11 >= aM2 and aM11 >= aM3 and aM11 >= aM4 and
                  aM11 >= aM5 and aM11 >= aM6 and aM11 >= aM7 and aM11 >= aM8 and aM11 >= aM9
                  and aM11 >= aM10 and aM11 >= aM12 and aM11 >= aM13 and aM11 >= aM14 and aM11 >= aM15):
                aMakespan = aM11
                dMakespan = aM11 - Makespan
            elif (aM12 >= aM1 and aM12 >= aM2 and aM12 >= aM3 and aM12 >= aM4 and
                  aM12 >= aM5 and aM12 >= aM6 and aM12 >= aM7 and aM12 >= aM8 and aM12 >= aM9
                  and aM12 >= aM10 and aM12 >= aM11 and aM12 >= aM13 and aM12 >= aM14 and aM12 >= aM15):
                aMakespan = aM12
                dMakespan = aM12 - Makespan
            elif (aM13 >= aM1 and aM13 >= aM2 and aM13 >= aM3 and aM13 >= aM4 and
                  aM13 >= aM5 and aM13 >= aM6 and aM13 >= aM7 and aM13 >= aM8 and aM13 >= aM9
                  and aM13 >= aM11 and aM13 >= aM12 and aM13 >= aM10 and aM13 >= aM14 and aM13 >= aM15):
                aMakespan = aM13
                dMakespan = aM13 - Makespan
            elif (aM14 >= aM1 and aM14 >= aM2 and aM14 >= aM3 and aM14 >= aM4 and
                  aM14 >= aM5 and aM14 >= aM6 and aM14 >= aM7 and aM14 >= aM8 and aM14 >= aM9
                  and aM14 >= aM11 and aM14 >= aM12 and aM14 >= aM13 and aM14 >= aM10 and aM14 >= aM15):
                aMakespan = aM14
                dMakespan = aM14 - Makespan
            elif (aM15 >= aM1 and aM15 >= aM2 and aM15 >= aM3 and aM15 >= aM4 and
                  aM15 >= aM5 and aM15 >= aM6 and aM15 >= aM7 and aM15 >= aM8 and aM15 >= aM9
                  and aM15 >= aM11 and aM15 >= aM12 and aM15 >= aM13 and aM15 >= aM14 and aM15 >= aM10):
                aMakespan = aM15
                dMakespan = aM15 - Makespan

            if (tM1 >= tM2 and tM1 >= tM3 and tM1 >= tM4 and tM1 >= tM5 and
                    tM1 >= tM6 and tM1 >= tM7 and tM1 >= tM8 and tM1 >= tM9 and tM1 >= tM10
                    and tM1 >= tM11 and tM1 >= tM12 and tM1 >= tM13 and tM1 >= tM14 and tM1 >= tM15):
                tMakespan = tM1
            elif (tM2 >= tM1 and tM2 >= tM3 and tM2 >= tM4 and tM2 >= tM5 and
                  tM2 >= tM6 and tM2 >= tM7 and tM2 >= tM8 and tM2 >= tM9 and tM2 >= tM10
                  and tM2 >= tM11 and tM2 >= tM12 and tM2 >= tM13 and tM2 >= tM14 and tM2 >= tM15):
                tMakespan = tM2
            elif (tM3 >= tM1 and tM3 >= tM2 and tM3 >= tM4 and tM3 >= tM5 and
                  tM3 >= tM6 and tM3 >= tM7 and tM3 >= tM8 and tM3 >= tM9 and tM3 >= tM10
                  and tM3 >= tM11 and tM3 >= tM12 and tM3 >= tM13 and tM3 >= tM14 and tM3 >= tM15):
                tMakespan = tM3
            elif (tM4 >= tM1 and tM4 >= tM2 and tM4 >= tM3 and tM4 >= tM5 and
                  tM4 >= tM6 and tM4 >= tM7 and tM4 >= tM8 and tM4 >= tM9 and tM4 >= tM10
                  and tM4 >= tM11 and tM4 >= tM12 and tM4 >= tM13 and tM4 >= tM14 and tM4 >= tM15):
                tMakespan = tM4
            elif (tM5 >= tM1 and tM5 >= tM2 and tM5 >= tM3 and tM5 >= tM4 and
                  tM5 >= tM6 and tM5 >= tM7 and tM5 >= tM8 and tM5 >= tM9 and tM5 >= tM10
                  and tM5 >= tM11 and tM5 >= tM12 and tM5 >= tM13 and tM5 >= tM14 and tM5 >= tM15):
                tMakespan = tM5
            elif (tM6 >= tM1 and tM6 >= tM2 and tM6 >= tM3 and tM6 >= tM4 and
                  tM6 >= tM5 and tM6 >= tM7 and tM6 >= tM8 and tM6 >= tM9 and tM6 >= tM10
                  and tM6 >= tM11 and tM6 >= tM12 and tM6 >= tM13 and tM6 >= tM14 and tM6 >= tM15):
                tMakespan = tM6
            elif (tM7 >= tM1 and tM7 >= tM2 and tM7 >= tM3 and tM7 >= tM4 and
                  tM7 >= tM5 and tM7 >= tM6 and tM7 >= tM8 and tM7 >= tM9 and tM7 >= tM10
                  and tM7 >= tM11 and tM7 >= tM12 and tM7 >= tM13 and tM7 >= tM14 and tM7 >= tM15):
                tMakespan = tM7
            elif (tM8 >= tM1 and tM8 >= tM2 and tM8 >= tM3 and tM8 >= tM4 and
                  tM8 >= tM5 and tM8 >= tM6 and tM8 >= tM7 and tM8 >= tM9 and tM8 >= tM10
                  and tM8 >= tM11 and tM8 >= tM12 and tM8 >= tM13 and tM8 >= tM14 and tM8 >= tM15):
                tMakespan = tM8
            elif (tM9 >= tM1 and tM9 >= tM2 and tM9 >= tM3 and tM9 >= tM4 and
                  tM9 >= tM5 and tM9 >= tM6 and tM9 >= tM7 and tM9 >= tM8 and tM9 >= tM10
                  and tM9 >= tM11 and tM9 >= tM12 and tM9 >= tM13 and tM9 >= tM14 and tM9 >= tM15):
                tMakespan = tM9
            elif (tM10 >= tM1 and tM10 >= tM2 and tM10 >= tM3 and tM10 >= tM4 and
                  tM10 >= tM5 and tM10 >= tM6 and tM10 >= tM7 and tM10 >= tM8 and tM10 >= tM9
                  and tM10 >= tM11 and tM10 >= tM12 and tM10 >= tM13 and tM10 >= tM14 and tM10 >= tM15):
                tMakespan = tM10
            elif (tM11 >= tM1 and tM11 >= tM2 and tM11 >= tM3 and tM11 >= tM4 and
                  tM11 >= tM5 and tM11 >= tM6 and tM11 >= tM7 and tM11 >= tM8 and tM11 >= tM9
                  and tM11 >= tM10 and tM11 >= tM12 and tM11 >= tM13 and tM11 >= tM14 and tM11 >= tM15):
                tMakespan = tM11
            elif (tM12 >= tM1 and tM12 >= tM2 and tM12 >= tM3 and tM12 >= tM4 and
                  tM12 >= tM5 and tM12 >= tM6 and tM12 >= tM7 and tM12 >= tM8 and tM12 >= tM9
                  and tM12 >= tM11 and tM12 >= tM10 and tM12 >= tM13 and tM12 >= tM14 and tM12 >= tM15):
                tMakespan = tM12
            elif (tM13 >= tM1 and tM13 >= tM2 and tM13 >= tM3 and tM13 >= tM4 and
                  tM13 >= tM5 and tM13 >= tM6 and tM13 >= tM7 and tM13 >= tM8 and tM13 >= tM9
                  and tM13 >= tM11 and tM13 >= tM12 and tM13 >= tM10 and tM13 >= tM14 and tM13 >= tM15):
                tMakespan = tM13
            elif (tM14 >= tM1 and tM14 >= tM2 and tM14 >= tM3 and tM14 >= tM4 and
                  tM14 >= tM5 and tM14 >= tM6 and tM14 >= tM7 and tM14 >= tM8 and tM14 >= tM9
                  and tM14 >= tM11 and tM14 >= tM12 and tM14 >= tM13 and tM14 >= tM10 and tM14 >= tM15):
                tMakespan = tM14
            elif (tM15 >= tM1 and tM15 >= tM2 and tM15 >= tM3 and tM15 >= tM4 and
                  tM15 >= tM5 and tM15 >= tM6 and tM15 >= tM7 and tM15 >= tM8 and tM15 >= tM9
                  and tM15 >= tM11 and tM15 >= tM12 and tM15 >= tM13 and tM15 >= tM14 and tM15 >= tM10):
                tMakespan = tM15

            ### Weighted Completion Time
            temp = Machine_Job_table
            temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
            temp['WCP'] = temp['P_cumsum'] * temp['W']

            sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()
            sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
            sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]
            sumWCP3 = sumWCP[sumWCP['MACHINEID'] == 3]
            sumWCP4 = sumWCP[sumWCP['MACHINEID'] == 4]
            sumWCP5 = sumWCP[sumWCP['MACHINEID'] == 5]
            sumWCP6 = sumWCP[sumWCP['MACHINEID'] == 6]
            sumWCP7 = sumWCP[sumWCP['MACHINEID'] == 7]
            sumWCP8 = sumWCP[sumWCP['MACHINEID'] == 8]
            sumWCP9 = sumWCP[sumWCP['MACHINEID'] == 9]
            sumWCP10 = sumWCP[sumWCP['MACHINEID'] == 10]
            sumWCP11 = sumWCP[sumWCP['MACHINEID'] == 11]
            sumWCP12 = sumWCP[sumWCP['MACHINEID'] == 12]
            sumWCP13 = sumWCP[sumWCP['MACHINEID'] == 13]
            sumWCP14 = sumWCP[sumWCP['MACHINEID'] == 14]
            sumWCP15 = sumWCP[sumWCP['MACHINEID'] == 15]

            if (sumWCP1.empty):
                wc1 = 0
            else:
                wc1 = sumWCP1.iloc[0, 8]
            if (sumWCP2.empty):
                wc2 = 0
            else:
                wc2 = sumWCP2.iloc[0, 8]
            if (sumWCP3.empty):
                wc3 = 0
            else:
                wc3 = sumWCP3.iloc[0, 8]
            if (sumWCP4.empty):
                wc4 = 0
            else:
                wc4 = sumWCP4.iloc[0, 8]
            if (sumWCP5.empty):
                wc5 = 0
            else:
                wc5 = sumWCP5.iloc[0, 8]
            if (sumWCP6.empty):
                wc6 = 0
            else:
                wc6 = sumWCP6.iloc[0, 8]
            if (sumWCP7.empty):
                wc7 = 0
            else:
                wc7 = sumWCP7.iloc[0, 8]
            if (sumWCP8.empty):
                wc8 = 0
            else:
                wc8 = sumWCP8.iloc[0, 8]
            if (sumWCP9.empty):
                wc9 = 0
            else:
                wc9 = sumWCP9.iloc[0, 8]
            if (sumWCP10.empty):
                wc10 = 0
            else:
                wc10 = sumWCP10.iloc[0, 8]
            if (sumWCP11.empty):
                wc11 = 0
            else:
                wc11 = sumWCP11.iloc[0, 8]
            if (sumWCP12.empty):
                wc12 = 0
            else:
                wc12 = sumWCP12.iloc[0, 8]
            if (sumWCP13.empty):
                wc13 = 0
            else:
                wc13 = sumWCP13.iloc[0, 8]
            if (sumWCP14.empty):
                wc14 = 0
            else:
                wc14 = sumWCP14.iloc[0, 8]
            if (sumWCP15.empty):
                wc15 = 0
            else:
                wc15 = sumWCP15.iloc[0, 8]
            wc = wc1 + wc2 + wc3 + wc4 + wc5 + wc6 + wc7 + wc8 + wc9 + wc10 + wc11 + wc12 + wc13 + wc14 + wc15

            if (priorityMachine == 1):
                awc1 = wc1 + aM1 * assigned_job_original[1]
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1 + tM1 * target_job_original[1]
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif (priorityMachine == 2):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2 + aM2 * assigned_job_original[1]
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2 + tM2 * target_job_original[1]
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif (priorityMachine == 3):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3 + aM3 * assigned_job_original[1]
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3 + tM3 * target_job_original[1]
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif (priorityMachine == 4):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4 + aM4 * assigned_job_original[1]
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4 + tM4 * target_job_original[1]
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif (priorityMachine == 5):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5 + aM5 * assigned_job_original[1]
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5 + tM5 * target_job_original[1]
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif (priorityMachine == 6):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6 + aM6 * assigned_job_original[1]
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6 + tM6 * target_job_original[1]
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif (priorityMachine == 7):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7 + aM7 * assigned_job_original[1]
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7 + tM7 * target_job_original[1]
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif (priorityMachine == 8):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8 + aM8 * assigned_job_original[1]
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8 + tM8 * target_job_original[1]
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif (priorityMachine == 9):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9 + aM9 * assigned_job_original[1]
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9 + tM9 * target_job_original[1]
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif (priorityMachine == 10):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10 + aM10 * assigned_job_original[1]
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10 + tM10 * target_job_original[1]
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif (priorityMachine == 11):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11 + aM11 * assigned_job_original[1]
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11 + tM11 * target_job_original[1]
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif(priorityMachine == 12):  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12 + aM12 * assigned_job_original[1]
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12 + tM12 * target_job_original[1]
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15
            elif(priorityMachine == 13):
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13 + aM13 * assigned_job_original[1]
                awc14 = wc14
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13 + tM13 * target_job_original[1]
                twc14 = wc14
                twc15 = wc15
            elif (priorityMachine == 14):
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14 + aM14 * assigned_job_original[1]
                awc15 = wc15
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14 + tM14 * target_job_original[1]
                twc15 = wc15
            elif (priorityMachine == 15):
                awc1 = wc1
                awc2 = wc2
                awc3 = wc3
                awc4 = wc4
                awc5 = wc5
                awc6 = wc6
                awc7 = wc7
                awc8 = wc8
                awc9 = wc9
                awc10 = wc10
                awc11 = wc11
                awc12 = wc12
                awc13 = wc13
                awc14 = wc14
                awc15 = wc15 + aM15 * assigned_job_original[1]
                twc1 = wc1
                twc2 = wc2
                twc3 = wc3
                twc4 = wc4
                twc5 = wc5
                twc6 = wc6
                twc7 = wc7
                twc8 = wc8
                twc9 = wc9
                twc10 = wc10
                twc11 = wc11
                twc12 = wc12
                twc13 = wc13
                twc14 = wc14
                twc15 = wc15 + tM15 * target_job_original[1]

            awc = awc1 + awc2 + awc3 + awc4 + awc5 + awc6 + awc7 + awc8 + awc9 + awc10 + awc11 + awc12 + awc13 + awc14 + awc15
            twc = twc1 + twc2 + twc3 + twc4 + twc5 + twc6 + twc7 + twc8 + twc9 + twc10 + twc11 + twc12 + twc13 + twc14 + twc15
        '''고치기'''
        machine_now = pd.Series(
            [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, wc1, wc2, wc3, wc4, wc5, wc6, wc7, wc8, wc9, wc10,
             wc11, wc12, wc13, wc14, wc15, wc,
             Makespan, SkT_M1, SkT_M2, SkT_M3, SkT_M4, SkT_M5, SkT_M6, SkT_M7, SkT_M8, SkT_M9, SkT_M10,
             SkT_M11, SkT_M12, SkT_M13, SkT_M14, SkT_M15],
            index=['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15',
                   "wc1", "wc2", 'wc3', 'wc4', 'wc5',
                   'wc6', 'wc7', 'wc8', 'wc9', 'wc10', 'wc11', 'wc12', 'wc13', 'wc14', 'wc15', "wc", "Makespan",
                   "Slack_M1", "Slack_M2", "Slack_M3", "Slack_M4", "Slack_M5",
                   "Slack_M6", "Slack_M7", "Slack_M8", "Slack_M9", "Slack_M10", 'Slack_M11', 'Slack_M12', 'Slack_M13', 'Slack_M14', 'Slack_M15'])

        compare_Attr_A = pd.Series([aM1, aM2, aM3, aM4, aM5, aM6, aM7, aM8, aM9, aM10,
                                    aM11, aM12, aM13, aM14, aM15, aMakespan,
                                    awc1, awc2, awc3, awc4, awc5, awc6, awc7, awc8, awc9, awc10,
                                    awc11, awc12, awc13, awc14, awc15, awc],
                                   index=['assign_M1', 'assign_M2', 'assign_M3', 'assign_M4', 'assign_M5',
                                          'assign_M6', 'assign_M7', 'assign_M8', 'assign_M9', 'assign_M10',
                                          'assign_M11', 'assign_M12', 'assign_M13', 'assign_M14', 'assign_M15',
                                          'assign_Makespan',
                                          'assign_WC1', 'assign_WC2', 'assign_WC3', 'assign_WC4', 'assign_WC5',
                                          'assign_WC6', 'assign_WC7', 'assign_WC8', 'assign_WC9', 'assign_WC10',
                                          'assign_WC11', 'assign_WC12', 'assign_WC13', 'assign_WC14', 'assign_WC15',
                                          "assign_WC"])
        compare_Attr_B = pd.Series([tM1, tM2, tM3, tM4, tM5, tM6, tM7, tM8, tM9, tM10, tM11, tM12, tM13, tM14, tM15, tMakespan,
                                    twc1, twc2, twc3, twc4, twc5, twc6, twc7, twc8, twc9, twc10,
                                    twc11, twc12, twc13, twc14, twc15, twc],
                                   index=['target_M1', 'target_M2', 'target_M3', 'target_M4', 'target_M5',
                                          'target_M6', 'target_M7', 'target_M8', 'target_M9', 'target_M10',
                                          'target_M11', 'target_M12', 'target_M13', 'target_M14', 'target_M15',
                                          'target_Makespan',
                                          'target_WC1', 'target_WC2', 'target_WC3', 'target_WC4', 'target_WC5',
                                          'target_WC6', 'target_WC7', 'target_WC8', 'target_WC9', 'target_WC10',
                                          'target_WC11', 'target_WC12', 'target_WC13', 'target_WC14', 'target_WC15',
                                          'target_WC'])

        compare_Attr = pd.Series([dMakespan], index=["dMakespan"])
        ''''''
        compare_AB_Diff = compare_Attr_A.reset_index(drop=True) - compare_Attr_B.reset_index(drop=True)
        compare_AB_Diff.index = ["M1_AB_Diff", "M2_AB_Diff", "M3_AB_Diff", "M4_AB_Diff", "M5_AB_Diff",
                                      "M6_AB_Diff", "M7_AB_Diff", "M8_AB_Diff", "M9_AB_Diff", "M10_AB_Diff",
                                      "M11_AB_Diff", "M12_AB_Diff", "M13_AB_Diff", "M14_AB_Diff", "M15_AB_Diff",
                                      "Makespan_AB_Diff",
                                      "wc1_Diff", "wc2_Diff", "wc3_Diff", "wc4_Diff", "wc5_Diff",
                                      "wc6_Diff", "wc7_Diff", "wc8_Diff", "wc9_Diff", "wc10_Diff",
                                      "wc11_Diff", "wc12_Diff", "wc13_Diff", "wc14_Diff", "wc15_Diff", "wc_Diff"]

        if (flag == 0):
            KPIInfo_Attr = machine_now.append(compare_Attr_A)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr_B)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            insertAttributes = jobA_Attr.append(jobB_Attr)
            insertAttributes = insertAttributes.append(KPIInfo_Attr)
        elif (flag == 1):
            KPIInfo_Attr = machine_now.append(compare_Attr)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_AB_Diff)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]

            insertAttributes = jobAB_Diff.append(KPIInfo_Attr)
        elif (flag == 2):
            KPIInfo_Attr = machine_now.append(compare_Attr)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            compare_Attr_AB_Flag = copy.deepcopy(compare_AB_Diff)
            compare_Attr_AB_Flag = Set_Value_By_ComparisionFlag(compare_Attr_AB_Flag)
            compare_Attr_AB_Flag.index = ["M1_AB_Flag", "M2_AB_Flag", "M3_AB_Flag", "M4_AB_Flag", "M5_AB_Flag",
                                      "M6_AB_Flag", "M7_AB_Flag", "M8_AB_Flag", "M9_AB_Flag", "M10_AB_Flag",
                                      "M11_AB_Flag", "M12_AB_Flag", "M13_AB_Flag", "M14_AB_Flag", "M15_AB_Flag",
                                      "Makespan_AB_Flag",
                                      "wc1_Flag", "wc2_Flag", "wc3_Flag", "wc4_Flag", "wc5_Flag",
                                      "wc6_Flag", "wc7_Flag", "wc8_Flag", "wc9_Flag", "wc10_Flag",
                                      "wc11_Flag", "wc12_Flag", "wc13_Flag", "wc14_Flag", "wc15_Flag", "wc_Flag"]

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Flag = copy.deepcopy(jobAB_Diff)
            jobAB_Flag = Set_Value_By_ComparisionFlag(jobAB_Flag)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]
            jobAB_Flag.index = ["P_AB_Flag", "W_AB_Flag"]

            insertAttributes = jobAB_Flag.append(KPIInfo_Attr)
            insertAttributes = insertAttributes.append(compare_Attr_AB_Flag)
        elif (flag == 3):
            KPIInfo_Attr = machine_now.append(compare_Attr_A)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr_B)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_Attr)
            KPIInfo_Attr = KPIInfo_Attr.append(compare_AB_Diff)

            KPIInfo_Attr = KPIInfo_Attr.to_numpy().reshape((1, -1))
            # KPIInfo_Attr = scaler.transform(KPIInfo_Attr)
            KPIInfo_Attr = KPIInfo_Attr.flatten()
            KPIInfo_Attr = pd.Series(KPIInfo_Attr)

            compare_Attr_AB_Flag = copy.deepcopy(compare_AB_Diff)
            compare_Attr_AB_Flag = Set_Value_By_ComparisionFlag(compare_Attr_AB_Flag)
            compare_Attr_AB_Flag.index = ["M1_AB_Flag", "M2_AB_Flag", "M3_AB_Flag", "M4_AB_Flag", "M5_AB_Flag",
                                      "M6_AB_Flag", "M7_AB_Flag", "M8_AB_Flag", "M9_AB_Flag", "M10_AB_Flag",
                                      "M11_AB_Flag", "M12_AB_Flag", "M13_AB_Flag", "M14_AB_Flag", "M15_AB_Flag",
                                      "Makespan_AB_Flag",
                                      "wc1_Flag", "wc2_Flag", "wc3_Flag", "wc4_Flag", "wc5_Flag",
                                      "wc6_Flag", "wc7_Flag", "wc8_Flag", "wc9_Flag", "wc10_Flag",
                                      "wc11_Flag", "wc12_Flag", "wc13_Flag", "wc14_Flag", "wc15_Flag", "wc_Flag"]

            jobAB_Diff = jobA_Attr.reset_index(drop=True) - jobB_Attr.reset_index(drop=True)
            jobAB_Flag = copy.deepcopy(jobAB_Diff)
            jobAB_Flag = Set_Value_By_ComparisionFlag(jobAB_Flag)
            jobAB_Diff.index = ["P_AB_Diff", "W_AB_Diff"]
            jobAB_Flag.index = ["P_AB_Flag", "W_AB_Flag"]

            insertAttributes = jobA_Attr.append(jobB_Attr)
            insertAttributes = insertAttributes.append(jobAB_Diff)
            insertAttributes = insertAttributes.append(jobAB_Flag)
            insertAttributes = insertAttributes.append(KPIInfo_Attr)
            insertAttributes = insertAttributes.append(compare_Attr_AB_Flag)

    #####정규화#####

    return insertAttributes

def SelectDispatchingJob(originalST, processedST, model, flag, Machine_Job_table, priorityMachine, machineNum): #####

    processedST['RESULT_pred'] = 0
    # processedST = np.array(processedST)
    # originalST = np.array(originalST) #####

    # competing_set = processedST[:, 1:input_shape+1]
    # competing_set_original = originalST[:, 1:input_shape+1] #####

    datas = GernerateAssigneStateAttrSetForTest(processedST, originalST, Machine_Job_table, priorityMachine, machineNum)

    datas = np.array(datas, dtype=np.float32)

    if (str(type(model))=='<class \'lambdamart.LambdaMART\'>'):
        result = model.predict(datas)
        maxIndex = np.argmax(result)
    elif (str(type(model))=='<class \'xgboost.core.Booster\'>'):
        test_dmatrix = DMatrix(datas)
        result = model.predict(test_dmatrix)
        maxIndex = np.argmax(result)
    elif (str(type(model)) == '<class \'lightgbm.sklearn.LGBMRanker\'>'):
        # test_dmatrix = DMatrix(datas)
        result = model.predict(datas)
        maxIndex = np.argmax(result)
    #Direct Ranker
    elif (str(type(model))=='<class \'DirectRanker.directRanker\'>'):

        # datas 구조 변환
        prediction = model.predict_proba(datas)
        sort_idx = np.argsort(np.concatenate(prediction))
    else:
        result = model.predict_result(datas)
        maxIndex = np.argmax(result.array)

    processedST = pd.DataFrame(processedST,
                      columns=["JOBID",
                               "P",
                               "W",
                               "PW",
                               "SCHEDULE_SEQ",
                               "RESULT_pred"])

    JOBID = processedST.iloc[maxIndex][0]

    return JOBID

def CalcPerformanceIndex(Machine_Job_table, machineNum):

    totalMakespan = None
    totalWeightedCompletionTime = None

    ## totalMakespan
    sumResult = Machine_Job_table.groupby(['MACHINEID'], as_index=False).sum()

    if(machineNum==2):
        M1_P = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
        M2_P = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]

        if (M1_P > M2_P):
            totalMakespan = M1_P
        else:
            totalMakespan = M2_P

        ## totalWeightedCompletionTime

        Machine_Job_table['P_cumsum'] = Machine_Job_table.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)

        Machine_Job_table['WCP'] = Machine_Job_table['P_cumsum'] * Machine_Job_table['W']

        sumResult = Machine_Job_table.sum()

        totalWeightedCompletionTime = sumResult['WCP']
    elif(machineNum==5):
        M1_P = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
        M2_P = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]
        M3_P = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]
        M4_P = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]
        M5_P = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]

        if (M1_P >= M2_P and M1_P >= M3_P and M1_P >= M4_P and M1_P >= M5_P):
            totalMakespan = M1_P
        elif(M2_P >= M1_P and M2_P >= M3_P and M2_P >= M4_P and M2_P >= M5_P):
            totalMakespan = M2_P
        elif (M3_P >= M1_P and M3_P >= M2_P and M3_P >= M4_P and M3_P >= M5_P):
            totalMakespan = M3_P
        elif (M4_P >= M1_P and M4_P >= M2_P and M4_P >= M3_P and M4_P >= M5_P):
            totalMakespan = M4_P
        elif (M5_P >= M1_P and M5_P >= M2_P and M5_P >= M3_P and M5_P >= M4_P):
            totalMakespan = M5_P

        ## totalWeightedCompletionTime

        Machine_Job_table['P_cumsum'] = Machine_Job_table.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)

        Machine_Job_table['WCP'] = Machine_Job_table['P_cumsum'] * Machine_Job_table['W']

        sumResult = Machine_Job_table.sum()

        totalWeightedCompletionTime = sumResult['WCP']
    elif (machineNum == 10):
        M1_P = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
        M2_P = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]
        M3_P = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]
        M4_P = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]
        M5_P = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]
        M6_P = sumResult[sumResult['MACHINEID'] == 6].P.iloc[0]
        M7_P = sumResult[sumResult['MACHINEID'] == 7].P.iloc[0]
        M8_P = sumResult[sumResult['MACHINEID'] == 8].P.iloc[0]
        M9_P = sumResult[sumResult['MACHINEID'] == 9].P.iloc[0]
        M10_P = sumResult[sumResult['MACHINEID'] == 10].P.iloc[0]

        if (M1_P >= M2_P and M1_P >= M3_P and M1_P >= M4_P and M1_P >= M5_P
            and M1_P >= M6_P and M1_P >= M7_P and M1_P >= M8_P and M1_P >= M9_P and M1_P >= M10_P):
            totalMakespan = M1_P
        elif (M2_P >= M1_P and M2_P >= M3_P and M2_P >= M4_P and M2_P >= M5_P
            and M2_P >= M6_P and M2_P >= M7_P and M2_P >= M8_P and M2_P >= M9_P and M2_P >= M10_P):
            totalMakespan = M2_P
        elif (M3_P >= M1_P and M3_P >= M2_P and M3_P >= M4_P and M3_P >= M5_P
            and M3_P >= M6_P and M3_P >= M7_P and M3_P >= M8_P and M3_P >= M9_P and M3_P >= M10_P):
            totalMakespan = M3_P
        elif (M4_P >= M1_P and M4_P >= M2_P and M4_P >= M3_P and M4_P >= M5_P
            and M4_P >= M6_P and M4_P >= M7_P and M4_P >= M8_P and M4_P >= M9_P and M4_P >= M10_P):
            totalMakespan = M4_P
        elif (M5_P >= M1_P and M5_P >= M2_P and M5_P >= M3_P and M5_P >= M4_P
            and M5_P >= M6_P and M5_P >= M7_P and M5_P >= M8_P and M5_P >= M9_P and M5_P >= M10_P):
            totalMakespan = M5_P
        elif (M6_P >= M1_P and M6_P >= M2_P and M6_P >= M3_P and M6_P >= M4_P
            and M6_P >= M5_P and M6_P >= M7_P and M6_P >= M8_P and M6_P >= M9_P and M6_P >= M10_P):
            totalMakespan = M6_P
        elif (M7_P >= M1_P and M7_P >= M2_P and M7_P >= M3_P and M7_P >= M4_P
            and M7_P >= M5_P and M7_P >= M6_P and M7_P >= M8_P and M7_P >= M9_P and M7_P >= M10_P):
            totalMakespan = M7_P
        elif (M8_P >= M1_P and M8_P >= M2_P and M8_P >= M3_P and M8_P >= M4_P
            and M8_P >= M5_P and M8_P >= M6_P and M8_P >= M7_P and M8_P >= M9_P and M8_P >= M10_P):
            totalMakespan = M8_P
        elif (M9_P >= M1_P and M9_P >= M2_P and M9_P >= M3_P and M9_P >= M4_P
            and M9_P >= M5_P and M9_P >= M6_P and M9_P >= M7_P and M9_P >= M8_P and M9_P >= M10_P):
            totalMakespan = M9_P
        elif (M10_P >= M1_P and M10_P >= M2_P and M10_P >= M3_P and M10_P >= M4_P
            and M10_P >= M5_P and M10_P >= M6_P and M10_P >= M7_P and M10_P >= M8_P and M9_P >= M9_P):
            totalMakespan = M10_P

        ## totalWeightedCompletionTime

        Machine_Job_table['P_cumsum'] = Machine_Job_table.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)

        Machine_Job_table['WCP'] = Machine_Job_table['P_cumsum'] * Machine_Job_table['W']

        sumResult = Machine_Job_table.sum()

        totalWeightedCompletionTime = sumResult['WCP']
    elif (machineNum == 15):
        M1_P = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
        M2_P = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]
        M3_P = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]
        M4_P = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]
        M5_P = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]
        M6_P = sumResult[sumResult['MACHINEID'] == 6].P.iloc[0]
        M7_P = sumResult[sumResult['MACHINEID'] == 7].P.iloc[0]
        M8_P = sumResult[sumResult['MACHINEID'] == 8].P.iloc[0]
        M9_P = sumResult[sumResult['MACHINEID'] == 9].P.iloc[0]
        M10_P = sumResult[sumResult['MACHINEID'] == 10].P.iloc[0]
        M11_P = sumResult[sumResult['MACHINEID'] == 11].P.iloc[0]
        M12_P = sumResult[sumResult['MACHINEID'] == 12].P.iloc[0]
        M13_P = sumResult[sumResult['MACHINEID'] == 13].P.iloc[0]
        M14_P = sumResult[sumResult['MACHINEID'] == 14].P.iloc[0]
        M15_P = sumResult[sumResult['MACHINEID'] == 15].P.iloc[0]

        if (M1_P >= M2_P and M1_P >= M3_P and M1_P >= M4_P and M1_P >= M5_P
            and M1_P >= M6_P and M1_P >= M7_P and M1_P >= M8_P and M1_P >= M9_P and M1_P >= M10_P
            and M1_P >= M11_P and M1_P >= M12_P and M1_P >= M13_P and M1_P >= M14_P and M1_P >= M15_P):
            totalMakespan = M1_P
        elif (M2_P >= M1_P and M2_P >= M3_P and M2_P >= M4_P and M2_P >= M5_P
            and M2_P >= M6_P and M2_P >= M7_P and M2_P >= M8_P and M2_P >= M9_P and M2_P >= M10_P
            and M2_P >= M11_P and M2_P >= M12_P and M2_P >= M13_P and M2_P >= M14_P and M2_P >= M15_P):
            totalMakespan = M2_P
        elif (M3_P >= M1_P and M3_P >= M2_P and M3_P >= M4_P and M3_P >= M5_P
            and M3_P >= M6_P and M3_P >= M7_P and M3_P >= M8_P and M3_P >= M9_P and M3_P >= M10_P
            and M3_P >= M11_P and M3_P >= M12_P and M3_P >= M13_P and M3_P >= M14_P and M3_P >= M15_P):
            totalMakespan = M3_P
        elif (M4_P >= M1_P and M4_P >= M2_P and M4_P >= M3_P and M4_P >= M5_P
            and M4_P >= M6_P and M4_P >= M7_P and M4_P >= M8_P and M4_P >= M9_P and M4_P >= M10_P
            and M4_P >= M11_P and M4_P >= M12_P and M4_P >= M13_P and M4_P >= M14_P and M4_P >= M15_P):
            totalMakespan = M4_P
        elif (M5_P >= M1_P and M5_P >= M2_P and M5_P >= M3_P and M5_P >= M4_P
            and M5_P >= M6_P and M5_P >= M7_P and M5_P >= M8_P and M5_P >= M9_P and M5_P >= M10_P
            and M5_P >= M11_P and M5_P >= M12_P and M5_P >= M13_P and M5_P >= M14_P and M5_P >= M15_P):
            totalMakespan = M5_P
        elif (M6_P >= M1_P and M6_P >= M2_P and M6_P >= M3_P and M6_P >= M4_P
            and M6_P >= M5_P and M6_P >= M7_P and M6_P >= M8_P and M6_P >= M9_P and M6_P >= M10_P
            and M6_P >= M11_P and M6_P >= M12_P and M6_P >= M13_P and M6_P >= M14_P and M6_P >= M15_P):
            totalMakespan = M6_P
        elif (M7_P >= M1_P and M7_P >= M2_P and M7_P >= M3_P and M7_P >= M4_P
            and M7_P >= M5_P and M7_P >= M6_P and M7_P >= M8_P and M7_P >= M9_P and M7_P >= M10_P
            and M7_P >= M11_P and M7_P >= M12_P and M7_P >= M13_P and M7_P >= M14_P and M7_P >= M15_P):
            totalMakespan = M7_P
        elif (M8_P >= M1_P and M8_P >= M2_P and M8_P >= M3_P and M8_P >= M4_P
            and M8_P >= M5_P and M8_P >= M6_P and M8_P >= M7_P and M8_P >= M9_P and M8_P >= M10_P
            and M8_P >= M11_P and M8_P >= M12_P and M8_P >= M13_P and M8_P >= M14_P and M8_P >= M15_P):
            totalMakespan = M8_P
        elif (M9_P >= M1_P and M9_P >= M2_P and M9_P >= M3_P and M9_P >= M4_P
            and M9_P >= M5_P and M9_P >= M6_P and M9_P >= M7_P and M9_P >= M8_P and M9_P >= M10_P
            and M9_P >= M11_P and M9_P >= M12_P and M9_P >= M13_P and M9_P >= M14_P and M9_P >= M15_P):
            totalMakespan = M9_P
        elif (M10_P >= M1_P and M10_P >= M2_P and M10_P >= M3_P and M10_P >= M4_P
            and M10_P >= M5_P and M10_P >= M6_P and M10_P >= M7_P and M10_P >= M8_P and M10_P >= M9_P
            and M10_P >= M11_P and M10_P >= M12_P and M10_P >= M13_P and M10_P >= M14_P and M10_P >= M15_P):
            totalMakespan = M10_P
        elif (M11_P >= M1_P and M11_P >= M2_P and M11_P >= M3_P and M11_P >= M4_P
            and M11_P >= M5_P and M11_P >= M6_P and M11_P >= M7_P and M11_P >= M8_P and M11_P >= M9_P
            and M11_P >= M10_P and M11_P >= M12_P and M11_P >= M13_P and M11_P >= M14_P and M11_P >= M15_P):
            totalMakespan = M11_P
        elif (M12_P >= M1_P and M12_P >= M2_P and M12_P >= M3_P and M12_P >= M4_P
            and M12_P >= M5_P and M12_P >= M6_P and M12_P >= M7_P and M12_P >= M8_P and M12_P >= M9_P
            and M12_P >= M11_P and M12_P >= M10_P and M12_P >= M13_P and M12_P >= M14_P and M12_P >= M15_P):
            totalMakespan = M12_P
        elif (M13_P >= M1_P and M13_P >= M2_P and M13_P >= M3_P and M13_P >= M4_P
            and M13_P >= M5_P and M13_P >= M6_P and M13_P >= M7_P and M13_P >= M8_P and M13_P >= M9_P
            and M13_P >= M11_P and M13_P >= M12_P and M13_P >= M10_P and M13_P >= M14_P and M13_P >= M15_P):
            totalMakespan = M13_P
        elif (M14_P >= M1_P and M14_P >= M2_P and M14_P >= M3_P and M14_P >= M4_P
            and M14_P >= M5_P and M14_P >= M6_P and M14_P >= M7_P and M14_P >= M8_P and M14_P >= M9_P
            and M14_P >= M11_P and M14_P >= M12_P and M14_P >= M13_P and M14_P >= M10_P and M14_P >= M15_P):
            totalMakespan = M14_P
        elif (M15_P >= M1_P and M15_P >= M2_P and M15_P >= M3_P and M15_P >= M4_P
            and M15_P >= M5_P and M15_P >= M6_P and M15_P >= M7_P and M15_P >= M8_P and M15_P >= M9_P
            and M15_P >= M11_P and M15_P >= M12_P and M15_P >= M13_P and M15_P >= M14_P and M15_P >= M10_P):
            totalMakespan = M15_P

        ## totalWeightedCompletionTime

        Machine_Job_table['P_cumsum'] = Machine_Job_table.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)

        Machine_Job_table['WCP'] = Machine_Job_table['P_cumsum'] * Machine_Job_table['W']

        sumResult = Machine_Job_table.sum()

        totalWeightedCompletionTime = sumResult['WCP']


    return totalMakespan, totalWeightedCompletionTime

def CheckMachinePriority(Machine_Job_table, machineNum):

    priorityMachine = 0

    sumResult = Machine_Job_table.groupby(['MACHINEID'], as_index=False).sum()

    if(machineNum==2):
        if (sumResult[sumResult['MACHINEID'] == 1].empty):
            M1_P = 0
        else:
            M1_P = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 2].empty):
            M2_P = 0
        else:
            M2_P = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]

        temp = Machine_Job_table

        temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)

        temp['WCP'] = temp['P_cumsum'] * temp['W']

        # P_cumsum = Machine_Job_table.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
        #
        # # W_cumsum = Machine_Job_table.groupby('MACHINEID')['W'].transform(pd.Series.cumsum)
        #
        # WCP = P_cumsum * Machine_Job_table['W']

        sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()

        sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
        sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]

        if(sumWCP1.empty):
            a = 0
        else:
            a = sumWCP1.iloc[0, 8]
        if(sumWCP2.empty):
            b = 0
        else:
            b = sumWCP2.iloc[0, 8]

        '''목적식에 따라 다르게'''
        # if (M1_P==0 and M2_P==0) or (M1_P + a <= M2_P + b) :
        if (M1_P == 0 and M2_P == 0) or (M1_P <= M2_P):
            priorityMachine = 1
        else:
            priorityMachine = 2
    elif(machineNum==5):
        if (sumResult[sumResult['MACHINEID'] == 1].empty):
            M1_P = 0
        else:
            M1_P = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 2].empty):
            M2_P = 0
        else:
            M2_P = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 3].empty):
            M3_P = 0
        else:
            M3_P = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 4].empty):
            M4_P = 0
        else:
            M4_P = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 5].empty):
            M5_P = 0
        else:
            M5_P = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]

        temp = Machine_Job_table

        temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)

        temp['WCP'] = temp['P_cumsum'] * temp['W']

        # P_cumsum = Machine_Job_table.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
        #
        # # W_cumsum = Machine_Job_table.groupby('MACHINEID')['W'].transform(pd.Series.cumsum)
        #
        # WCP = P_cumsum * Machine_Job_table['W']

        sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()

        sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
        sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]
        sumWCP3 = sumWCP[sumWCP['MACHINEID'] == 3]
        sumWCP4 = sumWCP[sumWCP['MACHINEID'] == 4]
        sumWCP5 = sumWCP[sumWCP['MACHINEID'] == 5]

        if (sumWCP1.empty):
            a = 0
        else:
            a = sumWCP1.iloc[0, 8]
        if (sumWCP2.empty):
            b = 0
        else:
            b = sumWCP2.iloc[0, 8]
        if (sumWCP3.empty):
            b = 0
        else:
            b = sumWCP3.iloc[0, 8]
        if (sumWCP4.empty):
            b = 0
        else:
            b = sumWCP4.iloc[0, 8]
        if (sumWCP5.empty):
            b = 0
        else:
            b = sumWCP5.iloc[0, 8]

        '''목적식에 따라 다르게'''
        # if (M1_P==0 and M2_P==0) or (M1_P + a <= M2_P + b) :
        if (M1_P == 0 and M2_P == 0 and M3_P == 0 and M4_P == 0 and M5_P == 0) or (
                M1_P <= M2_P and M1_P <= M3_P and M1_P <= M4_P and M1_P <= M5_P):
            priorityMachine = 1
        elif (M2_P <= M1_P and M2_P <= M3_P and M2_P <= M4_P and M2_P <= M5_P):
            priorityMachine = 2
        elif (M3_P <= M1_P and M3_P <= M2_P and M3_P <= M4_P and M3_P <= M5_P):
            priorityMachine = 3
        elif (M4_P <= M1_P and M4_P <= M2_P and M4_P <= M3_P and M4_P <= M5_P):
            priorityMachine = 4
        elif (M5_P <= M1_P and M5_P <= M2_P and M5_P <= M3_P and M5_P <= M4_P):
            priorityMachine = 5
    elif (machineNum == 10):
        if (sumResult[sumResult['MACHINEID'] == 1].empty):
            M1_P = 0
        else:
            M1_P = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 2].empty):
            M2_P = 0
        else:
            M2_P = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 3].empty):
            M3_P = 0
        else:
            M3_P = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 4].empty):
            M4_P = 0
        else:
            M4_P = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 5].empty):
            M5_P = 0
        else:
            M5_P = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 6].empty):
            M6_P = 0
        else:
            M6_P = sumResult[sumResult['MACHINEID'] == 6].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 7].empty):
            M7_P = 0
        else:
            M7_P = sumResult[sumResult['MACHINEID'] == 7].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 8].empty):
            M8_P = 0
        else:
            M8_P = sumResult[sumResult['MACHINEID'] == 8].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 9].empty):
            M9_P = 0
        else:
            M9_P = sumResult[sumResult['MACHINEID'] == 9].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 10].empty):
            M10_P = 0
        else:
            M10_P = sumResult[sumResult['MACHINEID'] == 10].P.iloc[0]

        temp = Machine_Job_table

        temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)

        temp['WCP'] = temp['P_cumsum'] * temp['W']

        # P_cumsum = Machine_Job_table.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
        #
        # # W_cumsum = Machine_Job_table.groupby('MACHINEID')['W'].transform(pd.Series.cumsum)
        #
        # WCP = P_cumsum * Machine_Job_table['W']

        sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()

        sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
        sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]
        sumWCP3 = sumWCP[sumWCP['MACHINEID'] == 3]
        sumWCP4 = sumWCP[sumWCP['MACHINEID'] == 4]
        sumWCP5 = sumWCP[sumWCP['MACHINEID'] == 5]
        sumWCP6 = sumWCP[sumWCP['MACHINEID'] == 6]
        sumWCP7 = sumWCP[sumWCP['MACHINEID'] == 7]
        sumWCP8 = sumWCP[sumWCP['MACHINEID'] == 8]
        sumWCP9 = sumWCP[sumWCP['MACHINEID'] == 9]
        sumWCP10 = sumWCP[sumWCP['MACHINEID'] == 10]

        if(sumWCP1.empty):
            a = 0
        else:
            a = sumWCP1.iloc[0, 8]
        if(sumWCP2.empty):
            b = 0
        else:
            b = sumWCP2.iloc[0, 8]
        if (sumWCP3.empty):
            b = 0
        else:
            b = sumWCP3.iloc[0, 8]
        if (sumWCP4.empty):
            b = 0
        else:
            b = sumWCP4.iloc[0, 8]
        if (sumWCP5.empty):
            b = 0
        else:
            b = sumWCP5.iloc[0, 8]
        if (sumWCP6.empty):
            a = 0
        else:
            a = sumWCP6.iloc[0, 8]
        if (sumWCP7.empty):
            b = 0
        else:
            b = sumWCP7.iloc[0, 8]
        if (sumWCP8.empty):
            b = 0
        else:
            b = sumWCP8.iloc[0, 8]
        if (sumWCP9.empty):
            b = 0
        else:
            b = sumWCP9.iloc[0, 8]
        if (sumWCP10.empty):
            b = 0
        else:
            b = sumWCP10.iloc[0, 8]

        if (M1_P == 0 and M2_P == 0 and M3_P == 0 and M4_P == 0 and M5_P == 0 and M6_P == 0 and M7_P == 0 and M8_P == 0 and M9_P == 0 and M10_P == 0)\
                or (M1_P <= M2_P and M1_P <= M3_P and M1_P <= M4_P and M1_P <= M5_P and
            M1_P <= M6_P and M1_P <= M7_P and M1_P <= M8_P and M1_P <= M9_P and M1_P <= M10_P):
            priorityMachine = 1
        elif(M2_P <= M1_P and M2_P <= M3_P and M2_P <= M4_P and M2_P <= M5_P and
            M2_P <= M6_P and M2_P <= M7_P and M2_P <= M8_P and M2_P <= M9_P and M2_P <= M10_P):
            priorityMachine = 2
        elif (M3_P <= M1_P and M3_P <= M2_P and M3_P <= M4_P and M3_P <= M5_P and
            M3_P <= M6_P and M3_P <= M7_P and M3_P <= M8_P and M3_P <= M9_P and M3_P <= M10_P):
            priorityMachine = 3
        elif (M4_P <= M1_P and M4_P <= M2_P and M4_P <= M3_P and M4_P <= M5_P and
            M4_P <= M6_P and M4_P <= M7_P and M4_P <= M8_P and M4_P <= M9_P and M4_P <= M10_P):
            priorityMachine = 4
        elif (M5_P <= M1_P and M5_P <= M2_P and M5_P <= M3_P and M5_P <= M4_P and
            M5_P <= M6_P and M5_P <= M7_P and M5_P <= M8_P and M5_P <= M9_P and M5_P <= M10_P):
            priorityMachine = 5
        elif (M6_P <= M1_P and M6_P <= M2_P and M6_P <= M3_P and M6_P <= M4_P and
            M6_P <= M5_P and M6_P <= M7_P and M6_P <= M8_P and M6_P <= M9_P and M6_P <= M10_P):
            priorityMachine = 6
        elif (M7_P <= M1_P and M7_P <= M2_P and M7_P <= M3_P and M7_P <= M4_P and
            M7_P <= M5_P and M7_P <= M6_P and M7_P <= M8_P and M7_P <= M9_P and M7_P <= M10_P):
            priorityMachine = 7
        elif (M8_P <= M1_P and M8_P <= M2_P and M8_P <= M3_P and M8_P <= M4_P and
            M8_P <= M5_P and M8_P <= M6_P and M8_P <= M7_P and M8_P <= M9_P and M8_P <= M10_P):
            priorityMachine = 8
        elif (M9_P <= M1_P and M9_P <= M2_P and M9_P <= M3_P and M9_P <= M4_P and
            M9_P <= M5_P and M9_P <= M6_P and M9_P <= M7_P and M9_P <= M8_P and M9_P <= M10_P):
            priorityMachine = 9
        elif (M10_P <= M1_P and M10_P <= M2_P and M10_P <= M3_P and M10_P <= M4_P and
            M10_P <= M5_P and M10_P <= M6_P and M10_P <= M7_P and M10_P <= M8_P and M10_P <= M9_P):
            priorityMachine = 10
    elif (machineNum == 15):
        if (sumResult[sumResult['MACHINEID'] == 1].empty):
            M1_P = 0
        else:
            M1_P = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 2].empty):
            M2_P = 0
        else:
            M2_P = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 3].empty):
            M3_P = 0
        else:
            M3_P = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 4].empty):
            M4_P = 0
        else:
            M4_P = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 5].empty):
            M5_P = 0
        else:
            M5_P = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 6].empty):
            M6_P = 0
        else:
            M6_P = sumResult[sumResult['MACHINEID'] == 6].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 7].empty):
            M7_P = 0
        else:
            M7_P = sumResult[sumResult['MACHINEID'] == 7].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 8].empty):
            M8_P = 0
        else:
            M8_P = sumResult[sumResult['MACHINEID'] == 8].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 9].empty):
            M9_P = 0
        else:
            M9_P = sumResult[sumResult['MACHINEID'] == 9].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 10].empty):
            M10_P = 0
        else:
            M10_P = sumResult[sumResult['MACHINEID'] == 10].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 11].empty):
            M11_P = 0
        else:
            M11_P = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 12].empty):
            M12_P = 0
        else:
            M12_P = sumResult[sumResult['MACHINEID'] == 12].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 13].empty):
            M13_P = 0
        else:
            M13_P = sumResult[sumResult['MACHINEID'] == 13].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 14].empty):
            M14_P = 0
        else:
            M14_P = sumResult[sumResult['MACHINEID'] == 14].P.iloc[0]

        if (sumResult[sumResult['MACHINEID'] == 15].empty):
            M15_P = 0
        else:
            M15_P = sumResult[sumResult['MACHINEID'] == 15].P.iloc[0]

        temp = Machine_Job_table

        temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)

        temp['WCP'] = temp['P_cumsum'] * temp['W']

        # P_cumsum = Machine_Job_table.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
        #
        # # W_cumsum = Machine_Job_table.groupby('MACHINEID')['W'].transform(pd.Series.cumsum)
        #
        # WCP = P_cumsum * Machine_Job_table['W']

        sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()

        sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
        sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]
        sumWCP3 = sumWCP[sumWCP['MACHINEID'] == 3]
        sumWCP4 = sumWCP[sumWCP['MACHINEID'] == 4]
        sumWCP5 = sumWCP[sumWCP['MACHINEID'] == 5]
        sumWCP6 = sumWCP[sumWCP['MACHINEID'] == 6]
        sumWCP7 = sumWCP[sumWCP['MACHINEID'] == 7]
        sumWCP8 = sumWCP[sumWCP['MACHINEID'] == 8]
        sumWCP9 = sumWCP[sumWCP['MACHINEID'] == 9]
        sumWCP10 = sumWCP[sumWCP['MACHINEID'] == 10]
        sumWCP11 = sumWCP[sumWCP['MACHINEID'] == 11]
        sumWCP12 = sumWCP[sumWCP['MACHINEID'] == 12]
        sumWCP13 = sumWCP[sumWCP['MACHINEID'] == 13]
        sumWCP14 = sumWCP[sumWCP['MACHINEID'] == 14]
        sumWCP15 = sumWCP[sumWCP['MACHINEID'] == 15]

        # if(sumWCP1.empty):
        #     a = 0
        # else:
        #     a = sumWCP1.iloc[0, 8]
        # if(sumWCP2.empty):
        #     b = 0
        # else:
        #     b = sumWCP2.iloc[0, 8]
        # if (sumWCP3.empty):
        #     b = 0
        # else:
        #     b = sumWCP3.iloc[0, 8]
        # if (sumWCP4.empty):
        #     b = 0
        # else:
        #     b = sumWCP4.iloc[0, 8]
        # if (sumWCP5.empty):
        #     b = 0
        # else:
        #     b = sumWCP5.iloc[0, 8]
        # if (sumWCP6.empty):
        #     a = 0
        # else:
        #     a = sumWCP6.iloc[0, 8]
        # if (sumWCP7.empty):
        #     b = 0
        # else:
        #     b = sumWCP7.iloc[0, 8]
        # if (sumWCP8.empty):
        #     b = 0
        # else:
        #     b = sumWCP8.iloc[0, 8]
        # if (sumWCP9.empty):
        #     b = 0
        # else:
        #     b = sumWCP9.iloc[0, 8]
        # if (sumWCP10.empty):
        #     b = 0
        # else:
        #     b = sumWCP10.iloc[0, 8]

        if (M1_P == 0 and M2_P == 0 and M3_P == 0 and M4_P == 0 and M5_P == 0 and M6_P == 0 and M7_P == 0 and M8_P == 0 and M9_P == 0 and M10_P == 0)\
                or (M1_P <= M2_P and M1_P <= M3_P and M1_P <= M4_P and M1_P <= M5_P and
            M1_P <= M6_P and M1_P <= M7_P and M1_P <= M8_P and M1_P <= M9_P and M1_P <= M10_P
            and M1_P <= M11_P and M1_P <= M12_P and M1_P <= M13_P and M1_P <= M14_P and M1_P <= M15_P):
            priorityMachine = 1
        elif(M2_P <= M1_P and M2_P <= M3_P and M2_P <= M4_P and M2_P <= M5_P and
            M2_P <= M6_P and M2_P <= M7_P and M2_P <= M8_P and M2_P <= M9_P and M2_P <= M10_P
            and M2_P <= M11_P and M2_P <= M12_P and M2_P <= M13_P and M2_P <= M14_P and M2_P <= M15_P):
            priorityMachine = 2
        elif (M3_P <= M1_P and M3_P <= M2_P and M3_P <= M4_P and M3_P <= M5_P and
            M3_P <= M6_P and M3_P <= M7_P and M3_P <= M8_P and M3_P <= M9_P and M3_P <= M10_P
            and M3_P <= M11_P and M3_P <= M12_P and M3_P <= M13_P and M3_P <= M14_P and M3_P <= M15_P):
            priorityMachine = 3
        elif (M4_P <= M1_P and M4_P <= M2_P and M4_P <= M3_P and M4_P <= M5_P and
            M4_P <= M6_P and M4_P <= M7_P and M4_P <= M8_P and M4_P <= M9_P and M4_P <= M10_P
            and M4_P <= M11_P and M4_P <= M12_P and M4_P <= M13_P and M4_P <= M14_P and M4_P <= M15_P):
            priorityMachine = 4
        elif (M5_P <= M1_P and M5_P <= M2_P and M5_P <= M3_P and M5_P <= M4_P and
            M5_P <= M6_P and M5_P <= M7_P and M5_P <= M8_P and M5_P <= M9_P and M5_P <= M10_P
            and M5_P <= M11_P and M5_P <= M12_P and M5_P <= M13_P and M5_P <= M14_P and M5_P <= M15_P):
            priorityMachine = 5
        elif (M6_P <= M1_P and M6_P <= M2_P and M6_P <= M3_P and M6_P <= M4_P and
            M6_P <= M5_P and M6_P <= M7_P and M6_P <= M8_P and M6_P <= M9_P and M6_P <= M10_P
            and M6_P <= M11_P and M6_P <= M12_P and M6_P <= M13_P and M6_P <= M14_P and M6_P <= M15_P):
            priorityMachine = 6
        elif (M7_P <= M1_P and M7_P <= M2_P and M7_P <= M3_P and M7_P <= M4_P and
            M7_P <= M5_P and M7_P <= M6_P and M7_P <= M8_P and M7_P <= M9_P and M7_P <= M10_P
            and M7_P <= M11_P and M7_P <= M12_P and M7_P <= M13_P and M7_P <= M14_P and M7_P <= M15_P):
            priorityMachine = 7
        elif (M8_P <= M1_P and M8_P <= M2_P and M8_P <= M3_P and M8_P <= M4_P and
            M8_P <= M5_P and M8_P <= M6_P and M8_P <= M7_P and M8_P <= M9_P and M8_P <= M10_P
            and M8_P <= M11_P and M8_P <= M12_P and M8_P <= M13_P and M8_P <= M14_P and M8_P <= M15_P):
            priorityMachine = 8
        elif (M9_P <= M1_P and M9_P <= M2_P and M9_P <= M3_P and M9_P <= M4_P and
            M9_P <= M5_P and M9_P <= M6_P and M9_P <= M7_P and M9_P <= M8_P and M9_P <= M10_P
            and M9_P <= M11_P and M9_P <= M12_P and M9_P <= M13_P and M9_P <= M14_P and M9_P <= M15_P):
            priorityMachine = 9
        elif (M10_P <= M1_P and M10_P <= M2_P and M10_P <= M3_P and M10_P <= M4_P and
            M10_P <= M5_P and M10_P <= M6_P and M10_P <= M7_P and M10_P <= M8_P and M10_P <= M9_P
            and M10_P <= M11_P and M10_P <= M12_P and M10_P <= M13_P and M10_P <= M14_P and M10_P <= M15_P):
            priorityMachine = 10
        elif (M11_P <= M1_P and M11_P <= M2_P and M11_P <= M3_P and M11_P <= M4_P and
            M11_P <= M5_P and M11_P <= M6_P and M11_P <= M7_P and M11_P <= M8_P and M11_P <= M9_P
            and M11_P <= M10_P and M11_P <= M12_P and M11_P <= M13_P and M11_P <= M14_P and M11_P <= M15_P):
            priorityMachine = 11
        elif (M12_P <= M1_P and M12_P <= M2_P and M12_P <= M3_P and M12_P <= M4_P and
            M12_P <= M5_P and M12_P <= M6_P and M12_P <= M7_P and M12_P <= M8_P and M12_P <= M9_P
            and M12_P <= M11_P and M12_P <= M10_P and M12_P <= M13_P and M12_P <= M14_P and M12_P <= M15_P):
            priorityMachine = 12
        elif (M13_P <= M1_P and M13_P <= M2_P and M13_P <= M3_P and M13_P <= M4_P and
            M13_P <= M5_P and M13_P <= M6_P and M13_P <= M7_P and M13_P <= M8_P and M13_P <= M9_P
            and M13_P <= M11_P and M13_P <= M12_P and M13_P <= M10_P and M13_P <= M14_P and M13_P <= M15_P):
            priorityMachine = 13
        elif (M14_P <= M1_P and M14_P <= M2_P and M14_P <= M3_P and M14_P <= M4_P and
            M14_P <= M5_P and M14_P <= M6_P and M14_P <= M7_P and M14_P <= M8_P and M14_P <= M9_P
            and M14_P <= M11_P and M14_P <= M12_P and M14_P <= M13_P and M14_P <= M10_P and M14_P <= M15_P):
            priorityMachine = 14
        elif (M15_P <= M1_P and M15_P <= M2_P and M15_P <= M3_P and M15_P <= M4_P and
            M15_P <= M5_P and M15_P <= M6_P and M15_P <= M7_P and M15_P <= M8_P and M15_P <= M9_P
            and M15_P <= M11_P and M15_P <= M12_P and M15_P <= M13_P and M15_P <= M14_P and M15_P <= M10_P):
            priorityMachine = 15

    return priorityMachine


def CreateColumns(machineNum, index=None):
    if (machineNum == 2):
        datas = pd.DataFrame(columns=["P", "W", "PW",
                                      'M1', 'M2',
                                      'wc1', 'wc2',
                                      'wc', 'Makespan',
                                      "Slack_M1", "Slack_M2",
                                      'assign_M1', 'assign_M2', 'assign_Makespan', 'assign_WC1', 'assign_WC2',
                                      'assign_WC',
                                      'dMakespan'], index= index)

    elif (machineNum == 5):
        datas = pd.DataFrame(columns=["P", "W", "PW",
                                      'M1', 'M2', 'M3', 'M4', 'M5', 'wc1', 'wc2', 'wc3', 'wc4', 'wc5', 'wc',
                                      'Makespan',
                                      "Slack_M1", "Slack_M2", "Slack_M3", "Slack_M4", "Slack_M5",
                                      'assign_M1', 'assign_M2', 'assign_M3', 'assign_M4', 'assign_M5',
                                      'assign_Makespan',
                                      'assign_WC1', 'assign_WC2', 'assign_WC3', 'assign_WC4', 'assign_WC5', "assign_WC",
                                      'dMakespan'], index= index)

    elif (machineNum == 10):
        datas = pd.DataFrame(columns=["P", "W", "PW",
                                      'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10',
                                      'wc1', 'wc2', 'wc3', 'wc4',
                                      'wc5', 'wc6', 'wc7', 'wc8', 'wc9', 'wc10', 'wc',
                                      'Makespan',
                                      "Slack_M1", "Slack_M2", "Slack_M3", "Slack_M4", "Slack_M5",
                                      "Slack_M6", "Slack_M7", "Slack_M8", "Slack_M9", "Slack_M10",
                                      'assign_M1', 'assign_M2', 'assign_M3', 'assign_M4', 'assign_M5',
                                      'assign_M6', 'assign_M7', 'assign_M8', 'assign_M9', 'assign_M10',
                                      'assign_Makespan',
                                      'assign_WC1', 'assign_WC2', 'assign_WC3', 'assign_WC4', 'assign_WC5',
                                      'assign_WC6', 'assign_WC7', 'assign_WC8', 'assign_WC9', 'assign_WC10',
                                      "assign_WC",
                                      'dMakespan'], index= index)

    elif (machineNum == 15):
        datas = pd.DataFrame(columns=["P", "W", "PW",
                                      # "P_B", "W_B",
                                      'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10',
                                      'M11', 'M12', 'M13', 'M14', 'M15',
                                      'wc1', 'wc2', 'wc3', 'wc4', 'wc5', 'wc6', 'wc7', 'wc8', 'wc9', 'wc10',
                                      'wc11', 'wc12', 'wc13', 'wc14', 'wc15', 'wc',
                                      'Makespan',
                                      "Slack_M1", "Slack_M2", "Slack_M3", "Slack_M4", "Slack_M5",
                                      "Slack_M6", "Slack_M7", "Slack_M8", "Slack_M9", "Slack_M10",
                                      "Slack_M11", "Slack_M12", "Slack_M13", "Slack_M14", "Slack_M15",
                                      'assign_M1', 'assign_M2', 'assign_M3', 'assign_M4', 'assign_M5',
                                      'assign_M6', 'assign_M7', 'assign_M8', 'assign_M9', 'assign_M10',
                                      'assign_M11', 'assign_M12', 'assign_M13', 'assign_M14', 'assign_M15',
                                      'assign_Makespan',
                                      'assign_WC1', 'assign_WC2', 'assign_WC3', 'assign_WC4', 'assign_WC5',
                                      'assign_WC6', 'assign_WC7', 'assign_WC8', 'assign_WC9', 'assign_WC10',
                                      'assign_WC11', 'assign_WC12', 'assign_WC13', 'assign_WC14', 'assign_WC15',
                                      "assign_WC",
                                      'dMakespan'], index= index)

    return datas


def GernerateAssigneStateAttrSetForTest(assigned_job, assigned_job_original, Machine_Job_table, priorityMachine, machineNum):

    new_assigned_job = CreateColumns(machineNum, assigned_job.index)

    if (machineNum == 2):

        # SET job and KPIinfo Attr
        new_assigned_job[np.intersect1d(assigned_job.columns, list(new_assigned_job.columns))] = assigned_job[np.intersect1d(assigned_job.columns, list(new_assigned_job.columns))]
        new_assigned_job = new_assigned_job.fillna(0)

        if (not Machine_Job_table.empty):
            ### Makespan
            sumResult = Machine_Job_table.groupby(['MACHINEID'], as_index=False).sum()

            if (sumResult[sumResult['MACHINEID'] == 1].empty):
                M1 = 0
            else:
                M1 = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
                # M1 = aM1
            if (sumResult[sumResult['MACHINEID'] == 2].empty):
                M2 = 0
            else:
                M2 = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]


            if (M1 <= M2):
                Makespan = M2
            else:
                Makespan = M1

            SkT_M1 = Makespan - M1
            SkT_M2 = Makespan - M2

            if (priorityMachine == 1):
                aM1 = M1 + assigned_job_original.loc[:, "P"]
                aM2 = M2

                aMakespan = aM1
                aMakespan[aM2>=aMakespan] = aM2
                dMakespan = aM1 - Makespan

            else:  # machine이 2인 경우
                aM1 = M1
                aM2 = M2 + assigned_job_original.loc[:, "P"]

                aMakespan = aM2
                aMakespan[aM1 <= aMakespan] = aM1
                dMakespan = aM2 - Makespan

            ### Weighted Completion Time
            temp = Machine_Job_table
            temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
            temp['WCP'] = temp['P_cumsum'] * temp['W']

            sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()
            sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
            sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]

            if (sumWCP1.empty):
                wc1 = 0
            else:
                wc1 = sumWCP1.iloc[0]["WCP"]
            if (sumWCP2.empty):
                wc2 = 0
            else:
                wc2 = sumWCP2.iloc[0]["WCP"]
            wc = wc1 + wc2

            if (priorityMachine == 1):
                awc1 = wc1 + aM1 * assigned_job_original.loc[:, "W"]
                awc2 = wc2
            else:  # machine이 2인 경우
                awc1 = wc1
                awc2 = wc2 + aM2 * assigned_job_original.loc[:, "W"]
            awc = awc1 + awc2

            '''Machine Info'''
            new_assigned_job["M1"] = M1
            new_assigned_job["M2"] = M2
            new_assigned_job["wc1"] = wc1
            new_assigned_job["wc2"] = wc2
            new_assigned_job["wc"] = wc
            new_assigned_job["Makespan"] = Makespan

            '''SlackTime'''
            new_assigned_job["Slack_M1"] = SkT_M1
            new_assigned_job["Slack_M2"] = SkT_M2
            # machine_now = pd.Series([slack_1, slack_2], index=["Slack_M1", "Slack_M2"])

            '''Machine별 KPI Info'''
            new_assigned_job.loc[:, "assign_WC1"] = awc1
            new_assigned_job.loc[:, "assign_WC2"] = awc2

            new_assigned_job.loc[:, "assign_M1"] = aM1
            new_assigned_job.loc[:, "assign_M2"] = aM2

            '''종합 KPI Info'''
            new_assigned_job.loc[:, "assign_Makespan"] = aMakespan
            new_assigned_job.loc[:, "assign_WC"] = awc
            new_assigned_job.loc[:, "dMakespan"] = dMakespan

    elif (machineNum == 5):

        # SET job and KPIinfo Attr
        new_assigned_job[np.intersect1d(assigned_job.columns, list(new_assigned_job.columns))] = assigned_job[np.intersect1d(assigned_job.columns, list(new_assigned_job.columns))]
        new_assigned_job = new_assigned_job.fillna(0)

        if (not Machine_Job_table.empty):
            ### Makespan
            sumResult = Machine_Job_table.groupby(['MACHINEID'], as_index=False).sum()

            if (sumResult[sumResult['MACHINEID'] == 1].empty):
                M1 = 0
            else:
                M1 = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 2].empty):
                M2 = 0
            else:
                M2 = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 3].empty):
                M3 = 0
            else:
                M3 = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 4].empty):
                M4 = 0
            else:
                M4 = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 5].empty):
                M5 = 0
            else:
                M5 = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 5].empty):
                M5 = 0
            else:
                M5 = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]




            Makespan = np.max([M1, M2, M3, M4, M5])

            SkT_M1 = Makespan - M1
            SkT_M2 = Makespan - M2
            SkT_M3 = Makespan - M3
            SkT_M4 = Makespan - M4
            SkT_M5 = Makespan - M5


            aM1 = M1
            aM2 = M2
            aM3 = M3
            aM4 = M4
            aM5 = M5

            if (priorityMachine == 1):
                aM1 = M1 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 2):
                aM2 = M2 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 3):
                aM3 = M3 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 4):
                aM4 = M4 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 5):
                aM5 = M5 + assigned_job_original.loc[:, "P"]


            ### Weighted Completion Time
            temp = Machine_Job_table
            temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
            temp['WCP'] = temp['P_cumsum'] * temp['W']

            sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()
            sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
            sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]
            sumWCP3 = sumWCP[sumWCP['MACHINEID'] == 3]
            sumWCP4 = sumWCP[sumWCP['MACHINEID'] == 4]
            sumWCP5 = sumWCP[sumWCP['MACHINEID'] == 5]

            if (sumWCP1.empty):
                wc1 = 0
            else:
                wc1 = sumWCP1.iloc[0]["WCP"]
            if (sumWCP2.empty):
                wc2 = 0
            else:
                wc2 = sumWCP2.iloc[0]["WCP"]
            if (sumWCP3.empty):
                wc3 = 0
            else:
                wc3 = sumWCP3.iloc[0]["WCP"]
            if (sumWCP4.empty):
                wc4 = 0
            else:
                wc4 = sumWCP4.iloc[0]["WCP"]
            if (sumWCP5.empty):
                wc5 = 0
            else:
                wc5 = sumWCP5.iloc[0]["WCP"]


            wc = wc1 + wc2 + wc3 + wc4 + wc5

            awc1 = wc1
            awc2 = wc2
            awc3 = wc3
            awc4 = wc4
            awc5 = wc5


            if (priorityMachine == 1):
                awc1 = wc1 + aM1 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 2):
                awc2 = wc2 + aM2 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 3):
                awc3 = wc3 + aM3 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 4):
                awc4 = wc4 + aM4 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 5):
                awc5 = wc5 + aM5 * assigned_job_original.loc[:, "W"]


            awc = awc1 + awc2 + awc3 + awc4 + awc5
            '''Machine Info'''
            new_assigned_job["M1"] = M1
            new_assigned_job["M2"] = M2
            new_assigned_job["M3"] = M3
            new_assigned_job["M4"] = M4
            new_assigned_job["M5"] = M5


            new_assigned_job["wc1"] = wc1
            new_assigned_job["wc2"] = wc2
            new_assigned_job["wc3"] = wc3
            new_assigned_job["wc4"] = wc4
            new_assigned_job["wc5"] = wc5
            new_assigned_job["wc"] = wc
            new_assigned_job["Makespan"] = Makespan

            '''SlackTime'''
            new_assigned_job["Slack_M1"] = SkT_M1
            new_assigned_job["Slack_M2"] = SkT_M2
            new_assigned_job["Slack_M3"] = SkT_M3
            new_assigned_job["Slack_M4"] = SkT_M4
            new_assigned_job["Slack_M5"] = SkT_M5

            '''Machine별 KPI Info'''

            new_assigned_job.loc[:, "assign_M1"] = aM1
            new_assigned_job.loc[:, "assign_M2"] = aM2
            new_assigned_job.loc[:, "assign_M3"] = aM3
            new_assigned_job.loc[:, "assign_M4"] = aM4
            new_assigned_job.loc[:, "assign_M5"] = aM5

            new_assigned_job.loc[:, "assign_WC1"] = awc1
            new_assigned_job.loc[:, "assign_WC2"] = awc2
            new_assigned_job.loc[:, "assign_WC3"] = awc3
            new_assigned_job.loc[:, "assign_WC4"] = awc4
            new_assigned_job.loc[:, "assign_WC5"] = awc5

            '''종합 KPI Info'''
            new_assigned_job.loc[:, "assign_Makespan"] = pd.concat([new_assigned_job.loc[:, "Makespan"]
                                                                   , new_assigned_job.loc[:, "assign_M1"]
                                                                   , new_assigned_job.loc[:, "assign_M2"]
                                                                   , new_assigned_job.loc[:, "assign_M3"]
                                                                   , new_assigned_job.loc[:, "assign_M4"]
                                                                   , new_assigned_job.loc[:, "assign_M5"]], axis=1).max(axis=1)
            new_assigned_job.loc[:, "assign_WC"] = new_assigned_job.loc[:, "assign_WC1"] + new_assigned_job.loc[:, "assign_WC2"] +new_assigned_job.loc[:, "assign_WC3"] + new_assigned_job.loc[:, "assign_WC4"] + new_assigned_job.loc[:, "assign_WC5"]
            new_assigned_job.loc[:, "dMakespan"] = new_assigned_job.loc[:, "assign_Makespan"] - new_assigned_job.loc[:, "Makespan"]

    elif (machineNum == 10):

        # SET job and KPIinfo Attr
        new_assigned_job[np.intersect1d(assigned_job.columns, list(new_assigned_job.columns))] = assigned_job[np.intersect1d(assigned_job.columns, list(new_assigned_job.columns))]
        new_assigned_job = new_assigned_job.fillna(0)

        if (not Machine_Job_table.empty):
            ### Makespan
            sumResult = Machine_Job_table.groupby(['MACHINEID'], as_index=False).sum()

            if (sumResult[sumResult['MACHINEID'] == 1].empty):
                M1 = 0
            else:
                M1 = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 2].empty):
                M2 = 0
            else:
                M2 = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 3].empty):
                M3 = 0
            else:
                M3 = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 4].empty):
                M4 = 0
            else:
                M4 = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 5].empty):
                M5 = 0
            else:
                M5 = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 5].empty):
                M5 = 0
            else:
                M5 = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 6].empty):
                M6 = 0
            else:
                M6 = sumResult[sumResult['MACHINEID'] == 6].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 7].empty):
                M7 = 0
            else:
                M7 = sumResult[sumResult['MACHINEID'] == 7].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 8].empty):
                M8 = 0
            else:
                M8 = sumResult[sumResult['MACHINEID'] == 8].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 9].empty):
                M9 = 0
            else:
                M9 = sumResult[sumResult['MACHINEID'] == 9].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 10].empty):
                M10 = 0
            else:
                M10 = sumResult[sumResult['MACHINEID'] == 10].P.iloc[0]



            Makespan = np.max([M1, M2, M3, M4, M5, M6, M7, M8, M9, M10])

            SkT_M1 = Makespan - M1
            SkT_M2 = Makespan - M2
            SkT_M3 = Makespan - M3
            SkT_M4 = Makespan - M4
            SkT_M5 = Makespan - M5
            SkT_M6 = Makespan - M6
            SkT_M7 = Makespan - M7
            SkT_M8 = Makespan - M8
            SkT_M9 = Makespan - M9
            SkT_M10 = Makespan - M10


            aM1 = M1
            aM2 = M2
            aM3 = M3
            aM4 = M4
            aM5 = M5
            aM6 = M6
            aM7 = M7
            aM8 = M8
            aM9 = M9
            aM10 = M10

            if (priorityMachine == 1):
                aM1 = M1 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 2):
                aM2 = M2 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 3):
                aM3 = M3 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 4):
                aM4 = M4 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 5):
                aM5 = M5 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 6):
                aM6 = M6 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 7):
                aM7 = M7 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 8):
                aM8 = M8 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 9):
                aM9 = M9 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 10):
                aM10 = M10 + assigned_job_original.loc[:, "P"]

            ### Weighted Completion Time
            temp = Machine_Job_table
            temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
            temp['WCP'] = temp['P_cumsum'] * temp['W']

            sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()
            sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
            sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]
            sumWCP3 = sumWCP[sumWCP['MACHINEID'] == 3]
            sumWCP4 = sumWCP[sumWCP['MACHINEID'] == 4]
            sumWCP5 = sumWCP[sumWCP['MACHINEID'] == 5]
            sumWCP6 = sumWCP[sumWCP['MACHINEID'] == 6]
            sumWCP7 = sumWCP[sumWCP['MACHINEID'] == 7]
            sumWCP8 = sumWCP[sumWCP['MACHINEID'] == 8]
            sumWCP9 = sumWCP[sumWCP['MACHINEID'] == 9]
            sumWCP10 = sumWCP[sumWCP['MACHINEID'] == 10]

            if (sumWCP1.empty):
                wc1 = 0
            else:
                wc1 = sumWCP1.iloc[0]["WCP"]
            if (sumWCP2.empty):
                wc2 = 0
            else:
                wc2 = sumWCP2.iloc[0]["WCP"]
            if (sumWCP3.empty):
                wc3 = 0
            else:
                wc3 = sumWCP3.iloc[0]["WCP"]
            if (sumWCP4.empty):
                wc4 = 0
            else:
                wc4 = sumWCP4.iloc[0]["WCP"]
            if (sumWCP5.empty):
                wc5 = 0
            else:
                wc5 = sumWCP5.iloc[0]["WCP"]
            if (sumWCP6.empty):
                wc6 = 0
            else:
                wc6 = sumWCP6.iloc[0]["WCP"]
            if (sumWCP7.empty):
                wc7 = 0
            else:
                wc7 = sumWCP7.iloc[0]["WCP"]
            if (sumWCP8.empty):
                wc8 = 0
            else:
                wc8 = sumWCP8.iloc[0]["WCP"]
            if (sumWCP9.empty):
                wc9 = 0
            else:
                wc9 = sumWCP9.iloc[0]["WCP"]
            if (sumWCP10.empty):
                wc10 = 0
            else:
                wc10 = sumWCP10.iloc[0]["WCP"]

            wc = wc1 + wc2 + wc3 + wc4 + wc5 + wc6 + wc7 + wc8 + wc9 + wc10

            awc1 = wc1
            awc2 = wc2
            awc3 = wc3
            awc4 = wc4
            awc5 = wc5
            awc6 = wc6
            awc7 = wc7
            awc8 = wc8
            awc9 = wc9
            awc10 = wc10

            if (priorityMachine == 1):
                awc1 = wc1 + aM1 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 2):
                awc2 = wc2 + aM2 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 3):
                awc3 = wc3 + aM3 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 4):
                awc4 = wc4 + aM4 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 5):
                awc5 = wc5 + aM5 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 6):
                awc6 = wc6 + aM6 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 7):
                awc7 = wc7 + aM7 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 8):
                awc8 = wc8 + aM8 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 9):
                awc9 = wc9 + aM9 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 10):
                awc10 = wc10 + aM10 * assigned_job_original.loc[:, "W"]

            awc = awc1 + awc2 + awc3 + awc4 + awc5 + awc6 + awc7 + awc8 + awc9 + awc10
            '''Machine Info'''
            new_assigned_job["M1"] = M1
            new_assigned_job["M2"] = M2
            new_assigned_job["M3"] = M3
            new_assigned_job["M4"] = M4
            new_assigned_job["M5"] = M5
            new_assigned_job["M6"] = M6
            new_assigned_job["M7"] = M7
            new_assigned_job["M8"] = M8
            new_assigned_job["M9"] = M9
            new_assigned_job["M10"] = M10

            new_assigned_job["wc1"] = wc1
            new_assigned_job["wc2"] = wc2
            new_assigned_job["wc3"] = wc3
            new_assigned_job["wc4"] = wc4
            new_assigned_job["wc5"] = wc5
            new_assigned_job["wc6"] = wc6
            new_assigned_job["wc7"] = wc7
            new_assigned_job["wc8"] = wc8
            new_assigned_job["wc9"] = wc9
            new_assigned_job["wc10"] = wc10
            new_assigned_job["wc"] = wc
            new_assigned_job["Makespan"] = Makespan

            '''SlackTime'''
            new_assigned_job["Slack_M1"] = SkT_M1
            new_assigned_job["Slack_M2"] = SkT_M2
            new_assigned_job["Slack_M3"] = SkT_M3
            new_assigned_job["Slack_M4"] = SkT_M4
            new_assigned_job["Slack_M5"] = SkT_M5
            new_assigned_job["Slack_M6"] = SkT_M6
            new_assigned_job["Slack_M7"] = SkT_M7
            new_assigned_job["Slack_M8"] = SkT_M8
            new_assigned_job["Slack_M9"] = SkT_M9
            new_assigned_job["Slack_M10"] = SkT_M10

            '''Machine별 KPI Info'''

            new_assigned_job.loc[:, "assign_M1"] = aM1
            new_assigned_job.loc[:, "assign_M2"] = aM2
            new_assigned_job.loc[:, "assign_M3"] = aM3
            new_assigned_job.loc[:, "assign_M4"] = aM4
            new_assigned_job.loc[:, "assign_M5"] = aM5
            new_assigned_job.loc[:, "assign_M6"] = aM6
            new_assigned_job.loc[:, "assign_M7"] = aM7
            new_assigned_job.loc[:, "assign_M8"] = aM8
            new_assigned_job.loc[:, "assign_M9"] = aM9
            new_assigned_job.loc[:, "assign_M10"] = aM10

            new_assigned_job.loc[:, "assign_WC1"] = awc1
            new_assigned_job.loc[:, "assign_WC2"] = awc2
            new_assigned_job.loc[:, "assign_WC3"] = awc3
            new_assigned_job.loc[:, "assign_WC4"] = awc4
            new_assigned_job.loc[:, "assign_WC5"] = awc5
            new_assigned_job.loc[:, "assign_WC6"] = awc6
            new_assigned_job.loc[:, "assign_WC7"] = awc7
            new_assigned_job.loc[:, "assign_WC8"] = awc8
            new_assigned_job.loc[:, "assign_WC9"] = awc9
            new_assigned_job.loc[:, "assign_WC10"] = awc10

            '''종합 KPI Info'''
            new_assigned_job.loc[:, "assign_Makespan"] = pd.concat([new_assigned_job.loc[:, "Makespan"]
                                                                   , new_assigned_job.loc[:, "assign_M1"]
                                                                   , new_assigned_job.loc[:, "assign_M2"]
                                                                   , new_assigned_job.loc[:, "assign_M3"]
                                                                   , new_assigned_job.loc[:, "assign_M4"]
                                                                   , new_assigned_job.loc[:, "assign_M5"]
                                                                   , new_assigned_job.loc[:, "assign_M6"]
                                                                   , new_assigned_job.loc[:, "assign_M7"]
                                                                   , new_assigned_job.loc[:, "assign_M8"]
                                                                   , new_assigned_job.loc[:, "assign_M9"]
                                                                   , new_assigned_job.loc[:, "assign_M10"]], axis=1).max(axis=1)
            new_assigned_job.loc[:, "assign_WC"] = new_assigned_job.loc[:, "assign_WC1"] + new_assigned_job.loc[:, "assign_WC2"] +new_assigned_job.loc[:, "assign_WC3"] + new_assigned_job.loc[:, "assign_WC4"] + new_assigned_job.loc[:, "assign_WC5"] \
                                            + new_assigned_job.loc[:, "assign_WC6"] + new_assigned_job.loc[:, "assign_WC7"] +new_assigned_job.loc[:, "assign_WC8"] + new_assigned_job.loc[:, "assign_WC9"] + new_assigned_job.loc[:, "assign_WC10"]
            new_assigned_job.loc[:, "dMakespan"] = new_assigned_job.loc[:, "assign_Makespan"] - new_assigned_job.loc[:, "Makespan"]

    elif (machineNum == 15):

        # SET job and KPIinfo Attr
        new_assigned_job[np.intersect1d(assigned_job.columns, list(new_assigned_job.columns))] = assigned_job[np.intersect1d(assigned_job.columns, list(new_assigned_job.columns))]
        new_assigned_job = new_assigned_job.fillna(0)

        if (not Machine_Job_table.empty):
            ### Makespan
            sumResult = Machine_Job_table.groupby(['MACHINEID'], as_index=False).sum()

            if (sumResult[sumResult['MACHINEID'] == 1].empty):
                M1 = 0
            else:
                M1 = sumResult[sumResult['MACHINEID'] == 1].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 2].empty):
                M2 = 0
            else:
                M2 = sumResult[sumResult['MACHINEID'] == 2].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 3].empty):
                M3 = 0
            else:
                M3 = sumResult[sumResult['MACHINEID'] == 3].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 4].empty):
                M4 = 0
            else:
                M4 = sumResult[sumResult['MACHINEID'] == 4].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 5].empty):
                M5 = 0
            else:
                M5 = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 5].empty):
                M5 = 0
            else:
                M5 = sumResult[sumResult['MACHINEID'] == 5].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 6].empty):
                M6 = 0
            else:
                M6 = sumResult[sumResult['MACHINEID'] == 6].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 7].empty):
                M7 = 0
            else:
                M7 = sumResult[sumResult['MACHINEID'] == 7].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 8].empty):
                M8 = 0
            else:
                M8 = sumResult[sumResult['MACHINEID'] == 8].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 9].empty):
                M9 = 0
            else:
                M9 = sumResult[sumResult['MACHINEID'] == 9].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 10].empty):
                M10 = 0
            else:
                M10 = sumResult[sumResult['MACHINEID'] == 10].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 11].empty):
                M11 = 0
            else:
                M11 = sumResult[sumResult['MACHINEID'] == 11].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 12].empty):
                M12 = 0
            else:
                M12 = sumResult[sumResult['MACHINEID'] == 12].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 13].empty):
                M13 = 0
            else:
                M13 = sumResult[sumResult['MACHINEID'] == 13].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 14].empty):
                M14 = 0
            else:
                M14 = sumResult[sumResult['MACHINEID'] == 14].P.iloc[0]
            if (sumResult[sumResult['MACHINEID'] == 15].empty):
                M15 = 0
            else:
                M15 = sumResult[sumResult['MACHINEID'] == 15].P.iloc[0]


            Makespan = np.max([M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15])

            SkT_M1 = Makespan - M1
            SkT_M2 = Makespan - M2
            SkT_M3 = Makespan - M3
            SkT_M4 = Makespan - M4
            SkT_M5 = Makespan - M5
            SkT_M6 = Makespan - M6
            SkT_M7 = Makespan - M7
            SkT_M8 = Makespan - M8
            SkT_M9 = Makespan - M9
            SkT_M10 = Makespan - M10
            SkT_M11 = Makespan - M11
            SkT_M12 = Makespan - M12
            SkT_M13 = Makespan - M13
            SkT_M14 = Makespan - M14
            SkT_M15 = Makespan - M15


            aM1 = M1
            aM2 = M2
            aM3 = M3
            aM4 = M4
            aM5 = M5
            aM6 = M6
            aM7 = M7
            aM8 = M8
            aM9 = M9
            aM10 = M10
            aM11 = M11
            aM12 = M12
            aM13 = M13
            aM14 = M14
            aM15 = M15

            if (priorityMachine == 1):
                aM1 = M1 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 2):
                aM2 = M2 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 3):
                aM3 = M3 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 4):
                aM4 = M4 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 5):
                aM5 = M5 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 6):
                aM6 = M6 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 7):
                aM7 = M7 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 8):
                aM8 = M8 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 9):
                aM9 = M9 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 10):
                aM10 = M10 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 11):
                aM11 = M11 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 12):
                aM12 = M12 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 13):
                aM13 = M13 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 14):
                aM14 = M14 + assigned_job_original.loc[:, "P"]
            elif (priorityMachine == 15):
                aM15 = M15 + assigned_job_original.loc[:, "P"]

            ### Weighted Completion Time
            temp = Machine_Job_table
            temp['P_cumsum'] = temp.groupby('MACHINEID')['P'].transform(pd.Series.cumsum)
            temp['WCP'] = temp['P_cumsum'] * temp['W']

            sumWCP = temp.groupby(['MACHINEID'], as_index=False).sum()
            sumWCP1 = sumWCP[sumWCP['MACHINEID'] == 1]
            sumWCP2 = sumWCP[sumWCP['MACHINEID'] == 2]
            sumWCP3 = sumWCP[sumWCP['MACHINEID'] == 3]
            sumWCP4 = sumWCP[sumWCP['MACHINEID'] == 4]
            sumWCP5 = sumWCP[sumWCP['MACHINEID'] == 5]
            sumWCP6 = sumWCP[sumWCP['MACHINEID'] == 6]
            sumWCP7 = sumWCP[sumWCP['MACHINEID'] == 7]
            sumWCP8 = sumWCP[sumWCP['MACHINEID'] == 8]
            sumWCP9 = sumWCP[sumWCP['MACHINEID'] == 9]
            sumWCP10 = sumWCP[sumWCP['MACHINEID'] == 10]
            sumWCP11 = sumWCP[sumWCP['MACHINEID'] == 11]
            sumWCP12 = sumWCP[sumWCP['MACHINEID'] == 12]
            sumWCP13 = sumWCP[sumWCP['MACHINEID'] == 13]
            sumWCP14 = sumWCP[sumWCP['MACHINEID'] == 14]
            sumWCP15 = sumWCP[sumWCP['MACHINEID'] == 15]

            if (sumWCP1.empty):
                wc1 = 0
            else:
                wc1 = sumWCP1.iloc[0]["WCP"]
            if (sumWCP2.empty):
                wc2 = 0
            else:
                wc2 = sumWCP2.iloc[0]["WCP"]
            if (sumWCP3.empty):
                wc3 = 0
            else:
                wc3 = sumWCP3.iloc[0]["WCP"]
            if (sumWCP4.empty):
                wc4 = 0
            else:
                wc4 = sumWCP4.iloc[0]["WCP"]
            if (sumWCP5.empty):
                wc5 = 0
            else:
                wc5 = sumWCP5.iloc[0]["WCP"]
            if (sumWCP6.empty):
                wc6 = 0
            else:
                wc6 = sumWCP6.iloc[0]["WCP"]
            if (sumWCP7.empty):
                wc7 = 0
            else:
                wc7 = sumWCP7.iloc[0]["WCP"]
            if (sumWCP8.empty):
                wc8 = 0
            else:
                wc8 = sumWCP8.iloc[0]["WCP"]
            if (sumWCP9.empty):
                wc9 = 0
            else:
                wc9 = sumWCP9.iloc[0]["WCP"]
            if (sumWCP10.empty):
                wc10 = 0
            else:
                wc10 = sumWCP10.iloc[0]["WCP"]
            if (sumWCP11.empty):
                wc11 = 0
            else:
                wc11 = sumWCP11.iloc[0]["WCP"]
            if (sumWCP12.empty):
                wc12 = 0
            else:
                wc12 = sumWCP12.iloc[0]["WCP"]
            if (sumWCP13.empty):
                wc13 = 0
            else:
                wc13 = sumWCP13.iloc[0]["WCP"]
            if (sumWCP14.empty):
                wc14 = 0
            else:
                wc14 = sumWCP14.iloc[0]["WCP"]
            if (sumWCP15.empty):
                wc15 = 0
            else:
                wc15 = sumWCP15.iloc[0]["WCP"]

            wc = wc1 + wc2 + wc3 + wc4 + wc5 + wc6 + wc7 + wc8 + wc9 + wc10 + wc11 + wc12 + wc13 + wc14 + wc15

            awc1 = wc1
            awc2 = wc2
            awc3 = wc3
            awc4 = wc4
            awc5 = wc5
            awc6 = wc6
            awc7 = wc7
            awc8 = wc8
            awc9 = wc9
            awc10 = wc10
            awc11 = wc11
            awc12 = wc12
            awc13 = wc13
            awc14 = wc14
            awc15 = wc15

            if (priorityMachine == 1):
                awc1 = wc1 + aM1 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 2):
                awc2 = wc2 + aM2 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 3):
                awc3 = wc3 + aM3 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 4):
                awc4 = wc4 + aM4 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 5):
                awc5 = wc5 + aM5 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 6):
                awc6 = wc6 + aM6 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 7):
                awc7 = wc7 + aM7 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 8):
                awc8 = wc8 + aM8 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 9):
                awc9 = wc9 + aM9 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 10):
                awc10 = wc10 + aM10 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 11):
                awc11 = wc11 + aM11 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 12):
                awc12 = wc12 + aM12 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 13):
                awc13 = wc13 + aM13 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 14):
                awc14 = wc14 + aM14 * assigned_job_original.loc[:, "W"]
            elif (priorityMachine == 15):
                awc15 = wc15 + aM15 * assigned_job_original.loc[:, "W"]

            awc = awc1 + awc2 + awc3 + awc4 + awc5 + awc6 + awc7 + awc8 + awc9 + awc10 + awc11 + awc12 + awc13 + awc14 + awc15

            '''Machine Info'''
            new_assigned_job["M1"] = M1
            new_assigned_job["M2"] = M2
            new_assigned_job["M3"] = M3
            new_assigned_job["M4"] = M4
            new_assigned_job["M5"] = M5
            new_assigned_job["M6"] = M6
            new_assigned_job["M7"] = M7
            new_assigned_job["M8"] = M8
            new_assigned_job["M9"] = M9
            new_assigned_job["M10"] = M10
            new_assigned_job["M11"] = M11
            new_assigned_job["M12"] = M12
            new_assigned_job["M13"] = M13
            new_assigned_job["M14"] = M14
            new_assigned_job["M15"] = M15

            new_assigned_job["wc1"] = wc1
            new_assigned_job["wc2"] = wc2
            new_assigned_job["wc3"] = wc3
            new_assigned_job["wc4"] = wc4
            new_assigned_job["wc5"] = wc5
            new_assigned_job["wc6"] = wc6
            new_assigned_job["wc7"] = wc7
            new_assigned_job["wc8"] = wc8
            new_assigned_job["wc9"] = wc9
            new_assigned_job["wc10"] = wc10
            new_assigned_job["wc11"] = wc11
            new_assigned_job["wc12"] = wc12
            new_assigned_job["wc13"] = wc13
            new_assigned_job["wc14"] = wc14
            new_assigned_job["wc15"] = wc15
            new_assigned_job["wc"] = wc
            new_assigned_job["Makespan"] = Makespan

            '''SlackTime'''
            new_assigned_job["Slack_M1"] = SkT_M1
            new_assigned_job["Slack_M2"] = SkT_M2
            new_assigned_job["Slack_M3"] = SkT_M3
            new_assigned_job["Slack_M4"] = SkT_M4
            new_assigned_job["Slack_M5"] = SkT_M5
            new_assigned_job["Slack_M6"] = SkT_M6
            new_assigned_job["Slack_M7"] = SkT_M7
            new_assigned_job["Slack_M8"] = SkT_M8
            new_assigned_job["Slack_M9"] = SkT_M9
            new_assigned_job["Slack_M10"] = SkT_M10
            new_assigned_job["Slack_M11"] = SkT_M11
            new_assigned_job["Slack_M12"] = SkT_M12
            new_assigned_job["Slack_M13"] = SkT_M13
            new_assigned_job["Slack_M14"] = SkT_M14
            new_assigned_job["Slack_M15"] = SkT_M15

            '''Machine별 KPI Info'''

            new_assigned_job.loc[:, "assign_M1"] = aM1
            new_assigned_job.loc[:, "assign_M2"] = aM2
            new_assigned_job.loc[:, "assign_M3"] = aM3
            new_assigned_job.loc[:, "assign_M4"] = aM4
            new_assigned_job.loc[:, "assign_M5"] = aM5
            new_assigned_job.loc[:, "assign_M6"] = aM6
            new_assigned_job.loc[:, "assign_M7"] = aM7
            new_assigned_job.loc[:, "assign_M8"] = aM8
            new_assigned_job.loc[:, "assign_M9"] = aM9
            new_assigned_job.loc[:, "assign_M10"] = aM10
            new_assigned_job.loc[:, "assign_M11"] = aM11
            new_assigned_job.loc[:, "assign_M12"] = aM12
            new_assigned_job.loc[:, "assign_M13"] = aM13
            new_assigned_job.loc[:, "assign_M14"] = aM14
            new_assigned_job.loc[:, "assign_M15"] = aM15

            new_assigned_job.loc[:, "assign_WC1"] = awc1
            new_assigned_job.loc[:, "assign_WC2"] = awc2
            new_assigned_job.loc[:, "assign_WC3"] = awc3
            new_assigned_job.loc[:, "assign_WC4"] = awc4
            new_assigned_job.loc[:, "assign_WC5"] = awc5
            new_assigned_job.loc[:, "assign_WC6"] = awc6
            new_assigned_job.loc[:, "assign_WC7"] = awc7
            new_assigned_job.loc[:, "assign_WC8"] = awc8
            new_assigned_job.loc[:, "assign_WC9"] = awc9
            new_assigned_job.loc[:, "assign_WC10"] = awc10
            new_assigned_job.loc[:, "assign_WC11"] = awc11
            new_assigned_job.loc[:, "assign_WC12"] = awc12
            new_assigned_job.loc[:, "assign_WC13"] = awc13
            new_assigned_job.loc[:, "assign_WC14"] = awc14
            new_assigned_job.loc[:, "assign_WC15"] = awc15

            '''종합 KPI Info'''
            new_assigned_job.loc[:, "assign_Makespan"] = pd.concat([new_assigned_job.loc[:, "Makespan"]
                                                                   , new_assigned_job.loc[:, "assign_M1"]
                                                                   , new_assigned_job.loc[:, "assign_M2"]
                                                                   , new_assigned_job.loc[:, "assign_M3"]
                                                                   , new_assigned_job.loc[:, "assign_M4"]
                                                                   , new_assigned_job.loc[:, "assign_M5"]
                                                                   , new_assigned_job.loc[:, "assign_M6"]
                                                                   , new_assigned_job.loc[:, "assign_M7"]
                                                                   , new_assigned_job.loc[:, "assign_M8"]
                                                                   , new_assigned_job.loc[:, "assign_M9"]
                                                                   , new_assigned_job.loc[:, "assign_M10"]
                                                                   , new_assigned_job.loc[:, "assign_M11"]
                                                                   , new_assigned_job.loc[:, "assign_M12"]
                                                                   , new_assigned_job.loc[:, "assign_M13"]
                                                                   , new_assigned_job.loc[:, "assign_M14"]
                                                                   , new_assigned_job.loc[:, "assign_M15"]], axis=1).max(axis=1)
            new_assigned_job.loc[:, "assign_WC"] = new_assigned_job.loc[:, "assign_WC1"] + new_assigned_job.loc[:, "assign_WC2"] +new_assigned_job.loc[:, "assign_WC3"] + new_assigned_job.loc[:, "assign_WC4"] + new_assigned_job.loc[:, "assign_WC5"] \
                                            + new_assigned_job.loc[:, "assign_WC6"] + new_assigned_job.loc[:, "assign_WC7"] +new_assigned_job.loc[:, "assign_WC8"] + new_assigned_job.loc[:, "assign_WC9"] + new_assigned_job.loc[:, "assign_WC10"] \
                                            + new_assigned_job.loc[:, "assign_WC11"] + new_assigned_job.loc[:, "assign_WC12"] +new_assigned_job.loc[:, "assign_WC13"] + new_assigned_job.loc[:, "assign_WC14"] + new_assigned_job.loc[:, "assign_WC15"]
            new_assigned_job.loc[:, "dMakespan"] = new_assigned_job.loc[:, "assign_Makespan"] - new_assigned_job.loc[:, "Makespan"]

    return new_assigned_job


def test_pred(model, originalST, processedST, modelName, flag, f, machineNum, jobNum,relevanceType):

    # originalST = originalST.drop(['Unnamed: 0'], axis=1)
    # Machine_Job_Table Columns
    col_names = ['MACHINEID']
    col_names = col_names + list(originalST.columns)

    # result df Columns
    # result_df_col_names = ['ModelType'
    #                         , 'ModelName'
    #                         , 'SCHEDULE_SEQ'
    #                         , 'TotalMakespan'
    #                         , 'TotalWeightedCompletionTime'
    #                         , 'ObjectValue'
    #                         , 'Time']

    result_df_col_names = ['MachineNum'
                            , 'JobNum'
                            , 'ProblemType'
                            , 'ModelName'
                            , 'SCHEDULE_SEQ'
                            , 'TotalMakespan'
                            , 'TotalWeightedCompletionTime'
                            , 'ObjectValue'
                            , 'Time']

    result_summury_df = pd.DataFrame(columns=result_df_col_names)

    schedule_Index_List = processedST['SCHEDULE_SEQ'].drop_duplicates()

    if (f == 0):
        ProblemType = "A"
    else:
        ProblemType = "O"

    print("===================================================================================")
    print("===================================================================================")
    print("Machine Num : " + str(machineNum) + " Job Num : " + str(jobNum) + " Problem Type : " + ProblemType + " Model : " + modelName)
    print("===================================================================================")
    print("===================================================================================")

    for i in schedule_Index_List:
        print("++++++++++++++++++++++++++++++++++")
        print("Schedule Number : " + str(i))
        print("++++++++++++++++++++++++++++++++++")

        time_start = time.time()

        originalST_run = originalST[originalST['SCHEDULE_SEQ'] == i]
        processedST_run = processedST[processedST['SCHEDULE_SEQ'] == i]

        Machine_Job_table = pd.DataFrame(columns=col_names)

        result_df = pd.DataFrame(columns=result_df_col_names)

        while (not processedST_run.empty):

            ## Machine Allocation
            priorityMachine = CheckMachinePriority(Machine_Job_table, machineNum)
            machineIndex = pd.Series([priorityMachine], index=['MACHINEID'])

            ## Dispatching Job Selection
            SelectedJOBID = SelectDispatchingJob(originalST_run, processedST_run, model, flag, Machine_Job_table, priorityMachine, machineNum)

            selectedJOBFilter = (originalST_run['JOBID'] == SelectedJOBID)
            selectedJOBAttributes = originalST_run[selectedJOBFilter].iloc[0]
            insertJobAttributes = machineIndex.append(selectedJOBAttributes)
            Machine_Job_table = Machine_Job_table.append(insertJobAttributes, ignore_index=True)

            table_filtered = (processedST_run['JOBID'] == SelectedJOBID)
            processedST_run = processedST_run.drop(processedST_run[table_filtered].index)
            processedST_run = processedST_run.reset_index(drop=True)

            table_filtered = (originalST_run['JOBID'] == SelectedJOBID)
            originalST_run = originalST_run.drop(originalST_run[table_filtered].index)
            originalST_run = originalST_run.reset_index(drop=True)

        ## Calculate Performance Index
        totalMakespan, totalWeightedCompletionTime = CalcPerformanceIndex(Machine_Job_table, machineNum)

        time_end = time.time() - time_start

        row = pd.Series([machineNum
                         , jobNum
                         , ProblemType
                         , modelName
                         , i
                         , totalMakespan
                         , totalWeightedCompletionTime
                         , totalMakespan + totalWeightedCompletionTime
                         , time_end], index=['MachineNum'
                                                , 'JobNum'
                                                , 'ProblemType'
                                                , 'ModelName'
                                                , 'SCHEDULE_SEQ'
                                                , 'TotalMakespan'
                                                , 'TotalWeightedCompletionTime'
                                                , 'ObjectValue'
                                                , 'Time'])

        result_df = result_df.append(row, ignore_index=True)
        result_summury_df = result_summury_df.append(row, ignore_index=True)

        now = datetime.datetime.now()
        year = '{:02d}'.format(now.year)
        month = '{:02d}'.format(now.month)
        day = '{:02d}'.format(now.day)
        timeStr = year + month + day

        if relevanceType==1:
            fileName = "Comparison_result_"+timeStr+".csv"
            result_df.to_csv(fileName, mode='a', header=(not os.path.exists(fileName)), index=False)
            fileName = "Machine_Job_table_" + timeStr + ".csv"
            Machine_Job_table.to_csv(fileName, mode='a', header=(not os.path.exists(fileName)), index=False)
        else :
            fileName = "Comparison_result_relType" + str(relevanceType) + timeStr + ".csv"
            result_df.to_csv(fileName, mode='a', header=(not os.path.exists(fileName)), index=False)
            fileName = "Machine_Job_table_relType" + str(relevanceType) + timeStr + ".csv"
            Machine_Job_table.to_csv(fileName, mode='a', header=(not os.path.exists(fileName)), index=False)



    fileName = None
    if relevanceType == 1:
        fileName = "Comparison_result_summary_" + timeStr + ".csv"
    else :
        fileName = "Comparison_result_summary_relType" + str(relevanceType) + timeStr + ".csv"

    result_summary_df = result_summury_df.groupby(['MachineNum', 'JobNum', 'ProblemType', 'ModelName']).mean()
    result_summary_df.to_csv(fileName, mode='a', header=(not os.path.exists(fileName)))

#################################################
#################################################
## Logic
#################################################
#################################################

#################################################
##Data 전처리
#################################################

flag = 0

# Import excel file

# '''MIP 목적식 분류'''
# mod = 0 # 다중
# # mod = 1 # Cmaxl
# # mod = 2 # wc
#
# '''MIP 모델링 분류'''
# # f = 0 # 원래 / 근사최적해
# f = 1 # newMIP / 최적해
#
# # machineNum = 2
# # machineNum = 5
# # machineNum = 10
# machineNum = 15
#
# # jobNum = 10
# # jobNum = 20
# # jobNum = 25
# # jobNum = 50
# jobNum = 75
# # jobNum = 100
# # jobNum = 150
#
# # LTRModel = "listnet"
# # LTRModel = "listmle"
# LTRModel = "listpl"

# LTRModel = ["listnet", "listmle", "listpl", "LambdaMart"]
# LTRModel = ["xgboost"]
# LTRModel = ["directRanker"]
# LTRModel = ["xgboost", "LightGBM"]
LTRModel = ["LightGBM"]

list_ = [[0, 2, 10], [0, 2, 20], [0, 5, 25], [0, 5, 50], [0, 10, 50], [0, 10, 100], [0, 15, 75], [0, 15, 150]
            , [1, 2, 10], [1, 2, 20], [1, 5, 25], [1, 5, 50], [1, 10, 50]
            # , [1, 10, 100]
            , [1, 15, 75]
            # , [1, 15, 150]
         ]

list_ = [[[1, 2, 10]]
         ]

# list_ = [[1, 2, 10]]
# list_ = [[0, 15, 150]]

# relevanceType 1 = 1or0, relevanceType 2 = 5or0, relevanceType 3 = 10or0
relevanceType = 1

result = []
for e in list_:
    e.append(LTRModel)
    result.append(e)
# result.pop()

for f, machineNum, jobNum, LTRModels in result:

    for LTRModel in LTRModels:

        # if (LTRModel != "LambdaMart"):
        #     continue

        fileLocation = "C:\\Users\\Han\\Google 드라이브\\oneDrive Bak\\대학원\\논문\\석사졸업논문\\실험\\ScheduML_data\\"  ## File 위치
        fileLocation = fileLocation + "M" + str(machineNum) + "N" + str(jobNum)

        fileName = None
        if f == 0:
            fileName = '\\2.CPLEX_data\\Approximate\\MIP_다중_머신_Test'  ## xlsx file 명
            if relevanceType == 1:
                modelFileName = '\\3.MIP_A\\3-4.Model\\MIP_A_Dispatching_Model_' + LTRModel
            else:
                modelFileName = '\\3.MIP_A\\3-4.Model\\MIP_A_Dispatching_Model_relType'+str(relevanceType)+LTRModel
        elif f == 1:
            fileName = '\\2.CPLEX_data\\Optimal\\newMIP_다중_머신_Test'  ## xlsx file 명
            if relevanceType == 1:
                modelFileName = '\\4.MIP_O\\4-4.Model\\MIP_O_Dispatching_Model_' + LTRModel
            else:
                modelFileName = '\\4.MIP_O\\4-4.Model\\MIP_O_Dispatching_Model_relType'+str(relevanceType)+LTRModel



        df_test = pd.ExcelFile(fileLocation + fileName +".xlsx")
        # print(df_test.sheet_names)
        df_test = df_test.parse(df_test.sheet_names[0])
        for column in df_test.columns:
            if "Unnamed" in column:
                df_test.drop(column, axis=1, inplace=True)

        originalST, processedST = DataPreprocessingForTesting(df_test)

        if (LTRModel.find("list") != -1):
            modelFileName = modelFileName + "l4u128"
            # Create neural network with chainer and apply our loss function
            # predictor = links.Linear(None, 1)
            #
            # if LTRModel=="listnet":
            #     loss = Ranker(predictor, listnet)
            # elif LTRModel=="listmle":
            #     loss = Ranker(predictor, listmle)
            # elif LTRModel=="listpl":
            #     loss = Ranker(predictor, listpl)

            if LTRModel == "listnet":
                loss = Ranker(listnet)
            elif LTRModel == "listmle":
                loss = Ranker(listmle)
            elif LTRModel == "listpl":
                loss = Ranker(listpl)

            serializers.load_npz(fileLocation+modelFileName+'.model', loss)
            model = loss

        elif LTRModel == "LambdaMart":
            model = LambdaMART()
            model.load(fileLocation+modelFileName+'.model')
        elif LTRModel == "xgboost":
            # model = pickle.load(open(fileLocation + modelFileName + ".pkl", "rb"))
            model = pickle.load(open(fileLocation + modelFileName + "_sklearn_pariwise.pkl", "rb"))

            # model = pickle.load(open(file_name, "rb"))
        elif LTRModel == "LightGBM":
            model = pickle.load(open(fileLocation + modelFileName + ".pkl", "rb"))
        # elif LTRModel == "directRanker":
        #     fileLocation = "C:\\Users\\Han\\Google 드라이브\\oneDrive Bak\\대학원\\논문\\석사졸업논문\\실험\\Learning to Rank 실습\\direct-ranker-master\\"
        #     file_name = fileLocation + "M15N150_MIP_A_directRanker"
        #     model = directRanker()
        #     model = directRanker.load_ranker(file_name)


        test_pred(model, originalST, processedST, LTRModel, flag, f, machineNum, jobNum, relevanceType)
        # test_pred(DNN_model, originalST, processedST, "DNN", flag)
