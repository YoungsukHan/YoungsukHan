import pandas as pd
import numpy as np
import copy

# 출력옵션
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
np.set_printoptions(edgeitems=1000, infstr='inf', linewidth=1000, nanstr='nan', precision=10, suppress=False,
                    threshold=20, formatter=None)


def data_preprocessing(data, machineNum, flag):
    ## flag 0 = comparisonDifference
    ## flag 1 = comparisonFlag
    ## flag 2 = comparisonJobAttr

    schedule_Index_List = data['SCHEDULE_SEQ'].drop_duplicates()

    originalData = copy.deepcopy(data)

    # max_schedule = int(data['SCHEDULE_SEQ'].max())
    # min_schedule = int(data['SCHEDULE_SEQ'].min())

    # PW 추가
    PW = data["P"]*data["W"]
    data.insert(3, "PW", PW)

    # 각각의 competing set별로 정규화를 진행
    for i in schedule_Index_List:
        data2 = data[data['SCHEDULE_SEQ'] == i]
        data5 = copy.deepcopy(data2)
        max_loop = int(data5['COMPETE_SEQ'].max())
        for j in range(1, max_loop + 1):
            data3 = data5[data5['COMPETE_SEQ'] == j]
            data4 = copy.deepcopy(data3)
            # 각각 정규화가 필요한 column 정규화

            if data4['P'].min() == data4['P'].max():
                if data4['P'].min() == 0:
                    data4['P'] = 0
                else:
                    data4['P'] /= data4['P'].max()
            else:
                data4['P'] = (data4['P'] - data4['P'].min()) / (data4['P'].max() - data4['P'].min())

            if data4['W'].min() == data4['W'].max():
                if data4['W'].min() == 0:
                    data4['W'] = 0
                else:
                    data4['W'] /= data4['W'].max()
            else:
                data4['W'] = (data4['W'] - data4['W'].min()) / (data4['W'].max() - data4['W'].min())\

            if data4['PW'].min() == data4['PW'].max():
                if data4['PW'].min() == 0:
                    data4['PW'] = 0
                else:
                    data4['PW'] /= data4['PW'].max()
            else:
                data4['PW'] = (data4['PW'] - data4['PW'].min()) / (data4['PW'].max() - data4['PW'].min())\


            data5[data5['COMPETE_SEQ'] == j] = data4
        data[data['SCHEDULE_SEQ'] == i] = data5

    # 전체 Competing set의 LOOP 순서 생성
    data = data.dropna()
    data['loop_total'] = 10000000000000000 * data['SCHEDULE_SEQ'] + 1000000000000 * data['COMPETE_SEQ']
    data['loop_rank'] = data['loop_total'].rank(method='dense')
    data = data.drop(columns=["COMPETE_SEQ", "SCHEDULE_SEQ"])
    data = data.reset_index(drop=True)

    if (machineNum == 2):
        data = pd.DataFrame(data, columns=["P", "W", "PW", 'MACHINE', 'M1', 'M2', "Makespan", "wc1", "wc2", "wc", "RESULT",
                                           "loop_rank"])
    if (machineNum == 5):
        data = pd.DataFrame(data, columns=["P", "W", "PW", 'MACHINE', 'M1', 'M2', "M3", "M4", "M5",
                                           "Makespan",
                                           "wc1", "wc2", "wc3", "wc4", "wc5", "wc", "RESULT",
                                           "loop_rank"])
    if (machineNum == 10):
        data = pd.DataFrame(data,
                            columns=["P", "W", "PW", 'MACHINE', 'M1', 'M2', "M3", "M4", "M5", 'M6', 'M7', "M8", "M9", "M10",
                                     "Makespan",
                                     "wc1", "wc2", "wc3", "wc4", "wc5", "wc6", "wc7", "wc8", "wc9", "wc10",
                                     "wc", "RESULT",
                                     "loop_rank"])
    if (machineNum == 15):
        data = pd.DataFrame(data,
                            columns=["P", "W", "PW", 'MACHINE', 'M1', 'M2', "M3", "M4", "M5", 'M6', 'M7', "M8", "M9", "M10",
                                     "M11", "M12", "M13", "M14", "M15",
                                     "Makespan",
                                     "wc1", "wc2", "wc3", "wc4", "wc5", "wc6", "wc7", "wc8", "wc9", "wc10",
                                     "wc11", "wc12", "wc13", "wc14", "wc15",
                                     "wc", "RESULT",
                                     "loop_rank"])

    return originalData, data


def GernerateAssigneStateAttrSet(datas, assigned_job, assigned_job_original, machineNum):

    # global aM1, aM2, aM3, aM4, aM5, aM6, aM7, aM8, aM9, aM10, aM11, aM12, aM13, aM14, aM15
    # global awc1, awc2, awc3, awc4, awc5, awc6, awc7, awc8, awc9, awc10, awc11, awc12, awc13, awc14, awc15
    # global aMakespan, awc, dMakespan

    new_assigned_job = CreateColumns(machineNum, assigned_job.index)

    if (machineNum == 2):
        # jobA_Attr = pd.Series(assigned_job[0:2], index=["P_A", "W_A"])
        # KPIInfo_Attr = pd.Series(assigned_job[3:9], index=['M1', 'M2', 'Makespan', 'wc1', 'wc2', 'wc'])

        # SET job and KPIinfo Attr
        new_assigned_job.loc[:, new_assigned_job.columns] = assigned_job.loc[:, new_assigned_job.columns]
        new_assigned_job = new_assigned_job.fillna(0)

        '''SlackTime 추가 부분'''
        new_assigned_job["Slack_M1"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M1"]
        new_assigned_job["Slack_M2"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M2"]
        # machine_now = pd.Series([slack_1, slack_2], index=["Slack_M1", "Slack_M2"])

        '''Machine별 KPI Info 추가 부분'''
        new_assigned_job.loc[:, "assign_WC1"] = new_assigned_job.loc[:, "wc1"]
        new_assigned_job.loc[:, "assign_WC2"] = new_assigned_job.loc[:, "wc2"]


        mask = (assigned_job.loc[:, "MACHINE"] == 1)
        new_assigned_job.loc[mask, "assign_M1"] = new_assigned_job.loc[mask, "M1"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC1"] = new_assigned_job.loc[mask, "wc1"] + new_assigned_job.loc[mask, "M1"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 2)
        new_assigned_job.loc[mask, "assign_M2"] = new_assigned_job.loc[mask, "M2"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC2"] = new_assigned_job.loc[mask, "wc2"] + new_assigned_job.loc[mask, "M2"] * assigned_job_original.loc[mask, "W"]

        '''종합 KPI Info 추가 부분'''
        new_assigned_job.loc[:, "assign_Makespan"] = pd.concat([new_assigned_job.loc[:, "Makespan"]
                                                                , new_assigned_job.loc[:, "assign_M1"]
                                                                , new_assigned_job.loc[:, "assign_M2"]], axis=1).max(axis=1)
        new_assigned_job.loc[:, "assign_WC"] = new_assigned_job.loc[:, "assign_WC1"] + new_assigned_job.loc[:, "assign_WC2"]
        new_assigned_job.loc[:, "dMakespan"] = new_assigned_job.loc[:, "assign_Makespan"] - new_assigned_job.loc[:, "Makespan"]

    elif (machineNum == 5):
        new_assigned_job.loc[:, new_assigned_job.columns] = assigned_job.loc[:, new_assigned_job.columns]
        new_assigned_job = new_assigned_job.fillna(0)

        '''SlackTime 추가 부분'''
        new_assigned_job["Slack_M1"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M1"]
        new_assigned_job["Slack_M2"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M2"]
        new_assigned_job["Slack_M3"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M3"]
        new_assigned_job["Slack_M4"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M4"]
        new_assigned_job["Slack_M5"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M5"]

        '''Machine별 KPI Info 추가 부분'''
        new_assigned_job.loc[:, "assign_M1"] = new_assigned_job.loc[:, "M1"]
        new_assigned_job.loc[:, "assign_M2"] = new_assigned_job.loc[:, "M2"]
        new_assigned_job.loc[:, "assign_M3"] = new_assigned_job.loc[:, "M3"]
        new_assigned_job.loc[:, "assign_M4"] = new_assigned_job.loc[:, "M4"]
        new_assigned_job.loc[:, "assign_M5"] = new_assigned_job.loc[:, "M5"]
        
        
        new_assigned_job.loc[:, "assign_WC1"] = new_assigned_job.loc[:, "wc1"]
        new_assigned_job.loc[:, "assign_WC2"] = new_assigned_job.loc[:, "wc2"]
        new_assigned_job.loc[:, "assign_WC3"] = new_assigned_job.loc[:, "wc3"]
        new_assigned_job.loc[:, "assign_WC4"] = new_assigned_job.loc[:, "wc4"]
        new_assigned_job.loc[:, "assign_WC5"] = new_assigned_job.loc[:, "wc5"]


        mask = (assigned_job.loc[:, "MACHINE"] == 1)
        new_assigned_job.loc[mask, "assign_M1"] = new_assigned_job.loc[mask, "M1"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC1"] = new_assigned_job.loc[mask, "wc1"] + new_assigned_job.loc[mask, "M1"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 2)
        new_assigned_job.loc[mask, "assign_M2"] = new_assigned_job.loc[mask, "M2"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC2"] = new_assigned_job.loc[mask, "wc2"] + new_assigned_job.loc[mask, "M2"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 3)
        new_assigned_job.loc[mask, "assign_M3"] = new_assigned_job.loc[mask, "M3"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC3"] = new_assigned_job.loc[mask, "wc3"] + new_assigned_job.loc[mask, "M3"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 4)
        new_assigned_job.loc[mask, "assign_M4"] = new_assigned_job.loc[mask, "M4"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC4"] = new_assigned_job.loc[mask, "wc4"] + new_assigned_job.loc[mask, "M4"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 5)
        new_assigned_job.loc[mask, "assign_M5"] = new_assigned_job.loc[mask, "M5"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC5"] = new_assigned_job.loc[mask, "wc5"] + new_assigned_job.loc[mask, "M5"] * assigned_job_original.loc[mask, "W"]


        '''종합 KPI Info 추가 부분'''
        new_assigned_job.loc[:, "assign_Makespan"] = pd.concat([new_assigned_job.loc[:, "Makespan"]
                                                               , new_assigned_job.loc[:, "assign_M1"]
                                                               , new_assigned_job.loc[:, "assign_M2"]
                                                               , new_assigned_job.loc[:, "assign_M3"]
                                                               , new_assigned_job.loc[:, "assign_M4"]
                                                               , new_assigned_job.loc[:, "assign_M5"]], axis=1).max(axis=1)
        new_assigned_job.loc[:, "assign_WC"] = new_assigned_job.loc[:, "assign_WC1"] + new_assigned_job.loc[:, "assign_WC2"] +new_assigned_job.loc[:, "assign_WC3"] + new_assigned_job.loc[:, "assign_WC4"] + new_assigned_job.loc[:, "assign_WC5"]
        new_assigned_job.loc[:, "dMakespan"] = new_assigned_job.loc[:, "assign_Makespan"] - new_assigned_job.loc[:, "Makespan"]

    elif (machineNum == 10):
        
        new_assigned_job.loc[:, new_assigned_job.columns] = assigned_job.loc[:, new_assigned_job.columns]
        new_assigned_job = new_assigned_job.fillna(0)

        '''SlackTime 추가 부분'''
        new_assigned_job["Slack_M1"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M1"]
        new_assigned_job["Slack_M2"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M2"]
        new_assigned_job["Slack_M3"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M3"]
        new_assigned_job["Slack_M4"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M4"]
        new_assigned_job["Slack_M5"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M5"]
        new_assigned_job["Slack_M6"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M6"]
        new_assigned_job["Slack_M7"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M7"]
        new_assigned_job["Slack_M8"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M8"]
        new_assigned_job["Slack_M9"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M9"]
        new_assigned_job["Slack_M10"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M10"]

        '''Machine별 KPI Info 추가 부분'''

        new_assigned_job.loc[:, "assign_M1"] = new_assigned_job.loc[:, "M1"]
        new_assigned_job.loc[:, "assign_M2"] = new_assigned_job.loc[:, "M2"]
        new_assigned_job.loc[:, "assign_M3"] = new_assigned_job.loc[:, "M3"]
        new_assigned_job.loc[:, "assign_M4"] = new_assigned_job.loc[:, "M4"]
        new_assigned_job.loc[:, "assign_M5"] = new_assigned_job.loc[:, "M5"]
        new_assigned_job.loc[:, "assign_M6"] = new_assigned_job.loc[:, "M6"]
        new_assigned_job.loc[:, "assign_M7"] = new_assigned_job.loc[:, "M7"]
        new_assigned_job.loc[:, "assign_M8"] = new_assigned_job.loc[:, "M8"]
        new_assigned_job.loc[:, "assign_M9"] = new_assigned_job.loc[:, "M9"]
        new_assigned_job.loc[:, "assign_M10"] = new_assigned_job.loc[:, "M10"]
        
        new_assigned_job.loc[:, "assign_WC1"] = new_assigned_job.loc[:, "wc1"]
        new_assigned_job.loc[:, "assign_WC2"] = new_assigned_job.loc[:, "wc2"]
        new_assigned_job.loc[:, "assign_WC3"] = new_assigned_job.loc[:, "wc3"]
        new_assigned_job.loc[:, "assign_WC4"] = new_assigned_job.loc[:, "wc4"]
        new_assigned_job.loc[:, "assign_WC5"] = new_assigned_job.loc[:, "wc5"]
        new_assigned_job.loc[:, "assign_WC6"] = new_assigned_job.loc[:, "wc6"]
        new_assigned_job.loc[:, "assign_WC7"] = new_assigned_job.loc[:, "wc7"]
        new_assigned_job.loc[:, "assign_WC8"] = new_assigned_job.loc[:, "wc8"]
        new_assigned_job.loc[:, "assign_WC9"] = new_assigned_job.loc[:, "wc9"]
        new_assigned_job.loc[:, "assign_WC10"] = new_assigned_job.loc[:, "wc10"]


        mask = (assigned_job.loc[:, "MACHINE"] == 1)
        new_assigned_job.loc[mask, "assign_M1"] = new_assigned_job.loc[mask, "M1"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC1"] = new_assigned_job.loc[mask, "wc1"] + new_assigned_job.loc[mask, "M1"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 2)
        new_assigned_job.loc[mask, "assign_M2"] = new_assigned_job.loc[mask, "M2"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC2"] = new_assigned_job.loc[mask, "wc2"] + new_assigned_job.loc[mask, "M2"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 3)
        new_assigned_job.loc[mask, "assign_M3"] = new_assigned_job.loc[mask, "M3"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC3"] = new_assigned_job.loc[mask, "wc3"] + new_assigned_job.loc[mask, "M3"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 4)
        new_assigned_job.loc[mask, "assign_M4"] = new_assigned_job.loc[mask, "M4"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC4"] = new_assigned_job.loc[mask, "wc4"] + new_assigned_job.loc[mask, "M4"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 5)
        new_assigned_job.loc[mask, "assign_M5"] = new_assigned_job.loc[mask, "M5"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC5"] = new_assigned_job.loc[mask, "wc5"] + new_assigned_job.loc[mask, "M5"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 6)
        new_assigned_job.loc[mask, "assign_M6"] = new_assigned_job.loc[mask, "M6"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC6"] = new_assigned_job.loc[mask, "wc6"] + new_assigned_job.loc[mask, "M6"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 7)
        new_assigned_job.loc[mask, "assign_M7"] = new_assigned_job.loc[mask, "M7"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC7"] = new_assigned_job.loc[mask, "wc7"] + new_assigned_job.loc[mask, "M7"] * assigned_job_original.loc[mask, "W"]
        
        mask = (assigned_job.loc[:, "MACHINE"] == 8)
        new_assigned_job.loc[mask, "assign_M8"] = new_assigned_job.loc[mask, "M8"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC8"] = new_assigned_job.loc[mask, "wc8"] + new_assigned_job.loc[mask, "M8"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 9)
        new_assigned_job.loc[mask, "assign_M9"] = new_assigned_job.loc[mask, "M9"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC9"] = new_assigned_job.loc[mask, "wc9"] + new_assigned_job.loc[mask, "M9"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 10)
        new_assigned_job.loc[mask, "assign_M10"] = new_assigned_job.loc[mask, "M10"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC10"] = new_assigned_job.loc[mask, "wc10"] + new_assigned_job.loc[mask, "M10"] * assigned_job_original.loc[mask, "W"]


        '''종합 KPI Info 추가 부분'''
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
        
        new_assigned_job.loc[:, new_assigned_job.columns] = assigned_job.loc[:, new_assigned_job.columns]
        new_assigned_job = new_assigned_job.fillna(0)

        '''SlackTime 추가 부분'''
        new_assigned_job["Slack_M1"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M1"]
        new_assigned_job["Slack_M2"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M2"]
        new_assigned_job["Slack_M3"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M3"]
        new_assigned_job["Slack_M4"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M4"]
        new_assigned_job["Slack_M5"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M5"]
        new_assigned_job["Slack_M6"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M6"]
        new_assigned_job["Slack_M7"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M7"]
        new_assigned_job["Slack_M8"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M8"]
        new_assigned_job["Slack_M9"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M9"]
        new_assigned_job["Slack_M10"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M10"]
        new_assigned_job["Slack_M11"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M11"]
        new_assigned_job["Slack_M12"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M12"]
        new_assigned_job["Slack_M13"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M13"]
        new_assigned_job["Slack_M14"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M14"]
        new_assigned_job["Slack_M15"] = new_assigned_job.loc[:, "Makespan"] - new_assigned_job.loc[:, "M15"]

        '''Machine별 KPI Info 추가 부분'''
        new_assigned_job.loc[:, "assign_M1"] = new_assigned_job.loc[:, "M1"]
        new_assigned_job.loc[:, "assign_M2"] = new_assigned_job.loc[:, "M2"]
        new_assigned_job.loc[:, "assign_M3"] = new_assigned_job.loc[:, "M3"]
        new_assigned_job.loc[:, "assign_M4"] = new_assigned_job.loc[:, "M4"]
        new_assigned_job.loc[:, "assign_M5"] = new_assigned_job.loc[:, "M5"]
        new_assigned_job.loc[:, "assign_M6"] = new_assigned_job.loc[:, "M6"]
        new_assigned_job.loc[:, "assign_M7"] = new_assigned_job.loc[:, "M7"]
        new_assigned_job.loc[:, "assign_M8"] = new_assigned_job.loc[:, "M8"]
        new_assigned_job.loc[:, "assign_M9"] = new_assigned_job.loc[:, "M9"]
        new_assigned_job.loc[:, "assign_M10"] = new_assigned_job.loc[:, "M10"]
        new_assigned_job.loc[:, "assign_M11"] = new_assigned_job.loc[:, "M11"]
        new_assigned_job.loc[:, "assign_M12"] = new_assigned_job.loc[:, "M12"]
        new_assigned_job.loc[:, "assign_M13"] = new_assigned_job.loc[:, "M13"]
        new_assigned_job.loc[:, "assign_M14"] = new_assigned_job.loc[:, "M14"]
        new_assigned_job.loc[:, "assign_M15"] = new_assigned_job.loc[:, "M15"]
        
        new_assigned_job.loc[:, "assign_WC1"] = new_assigned_job.loc[:, "wc1"]
        new_assigned_job.loc[:, "assign_WC2"] = new_assigned_job.loc[:, "wc2"]
        new_assigned_job.loc[:, "assign_WC3"] = new_assigned_job.loc[:, "wc3"]
        new_assigned_job.loc[:, "assign_WC4"] = new_assigned_job.loc[:, "wc4"]
        new_assigned_job.loc[:, "assign_WC5"] = new_assigned_job.loc[:, "wc5"]
        new_assigned_job.loc[:, "assign_WC6"] = new_assigned_job.loc[:, "wc6"]
        new_assigned_job.loc[:, "assign_WC7"] = new_assigned_job.loc[:, "wc7"]
        new_assigned_job.loc[:, "assign_WC8"] = new_assigned_job.loc[:, "wc8"]
        new_assigned_job.loc[:, "assign_WC9"] = new_assigned_job.loc[:, "wc9"]
        new_assigned_job.loc[:, "assign_WC10"] = new_assigned_job.loc[:, "wc10"]
        new_assigned_job.loc[:, "assign_WC11"] = new_assigned_job.loc[:, "wc11"]
        new_assigned_job.loc[:, "assign_WC12"] = new_assigned_job.loc[:, "wc12"]
        new_assigned_job.loc[:, "assign_WC13"] = new_assigned_job.loc[:, "wc13"]
        new_assigned_job.loc[:, "assign_WC14"] = new_assigned_job.loc[:, "wc14"]
        new_assigned_job.loc[:, "assign_WC15"] = new_assigned_job.loc[:, "wc15"]

        mask = (assigned_job.loc[:, "MACHINE"] == 1)
        new_assigned_job.loc[mask, "assign_M1"] = new_assigned_job.loc[mask, "M1"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC1"] = new_assigned_job.loc[mask, "wc1"] + new_assigned_job.loc[mask, "M1"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 2)
        new_assigned_job.loc[mask, "assign_M2"] = new_assigned_job.loc[mask, "M2"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC2"] = new_assigned_job.loc[mask, "wc2"] + new_assigned_job.loc[mask, "M2"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 3)
        new_assigned_job.loc[mask, "assign_M3"] = new_assigned_job.loc[mask, "M3"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC3"] = new_assigned_job.loc[mask, "wc3"] + new_assigned_job.loc[mask, "M3"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 4)
        new_assigned_job.loc[mask, "assign_M4"] = new_assigned_job.loc[mask, "M4"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC4"] = new_assigned_job.loc[mask, "wc4"] + new_assigned_job.loc[mask, "M4"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 5)
        new_assigned_job.loc[mask, "assign_M5"] = new_assigned_job.loc[mask, "M5"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC5"] = new_assigned_job.loc[mask, "wc5"] + new_assigned_job.loc[mask, "M5"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 6)
        new_assigned_job.loc[mask, "assign_M6"] = new_assigned_job.loc[mask, "M6"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC6"] = new_assigned_job.loc[mask, "wc6"] + new_assigned_job.loc[mask, "M6"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 7)
        new_assigned_job.loc[mask, "assign_M7"] = new_assigned_job.loc[mask, "M7"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC7"] = new_assigned_job.loc[mask, "wc7"] + new_assigned_job.loc[mask, "M7"] * assigned_job_original.loc[mask, "W"]
        
        mask = (assigned_job.loc[:, "MACHINE"] == 8)
        new_assigned_job.loc[mask, "assign_M8"] = new_assigned_job.loc[mask, "M8"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC8"] = new_assigned_job.loc[mask, "wc8"] + new_assigned_job.loc[mask, "M8"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 9)
        new_assigned_job.loc[mask, "assign_M9"] = new_assigned_job.loc[mask, "M9"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC9"] = new_assigned_job.loc[mask, "wc9"] + new_assigned_job.loc[mask, "M9"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 10)
        new_assigned_job.loc[mask, "assign_M10"] = new_assigned_job.loc[mask, "M10"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC10"] = new_assigned_job.loc[mask, "wc10"] + new_assigned_job.loc[mask, "M10"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 11)
        new_assigned_job.loc[mask, "assign_M11"] = new_assigned_job.loc[mask, "M11"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC11"] = new_assigned_job.loc[mask, "wc11"] + new_assigned_job.loc[mask, "M11"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 12)
        new_assigned_job.loc[mask, "assign_M12"] = new_assigned_job.loc[mask, "M12"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC12"] = new_assigned_job.loc[mask, "wc12"] + new_assigned_job.loc[mask, "M12"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 13)
        new_assigned_job.loc[mask, "assign_M13"] = new_assigned_job.loc[mask, "M13"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC13"] = new_assigned_job.loc[mask, "wc13"] + new_assigned_job.loc[mask, "M13"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 14)
        new_assigned_job.loc[mask, "assign_M14"] = new_assigned_job.loc[mask, "M14"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC14"] = new_assigned_job.loc[mask, "wc14"] + new_assigned_job.loc[mask, "M14"] * assigned_job_original.loc[mask, "W"]

        mask = (assigned_job.loc[:, "MACHINE"] == 15)
        new_assigned_job.loc[mask, "assign_M15"] = new_assigned_job.loc[mask, "M15"] + assigned_job_original.loc[mask, "P"]
        new_assigned_job.loc[mask, "assign_WC15"] = new_assigned_job.loc[mask, "wc15"] + new_assigned_job.loc[mask, "M15"] * assigned_job_original.loc[mask, "W"]


        '''종합 KPI Info 추가 부분'''
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

    datas = datas.append(new_assigned_job)

    return datas

def CreateColumns(machineNum, index=None):
    if (machineNum == 2):
        datas = pd.DataFrame(columns=["P", "W", "PW",
                                      'M1', 'M2',
                                      'wc1', 'wc2',
                                      'wc', 'Makespan',
                                      "Slack_M1", "Slack_M2",
                                      'assign_M1', 'assign_M2', 'assign_Makespan', 'assign_WC1', 'assign_WC2',
                                      'assign_WC',
                                      'dMakespan',
                                      "RESULT"], index= index)

    elif (machineNum == 5):
        datas = pd.DataFrame(columns=["P", "W", "PW",
                                      'M1', 'M2', 'M3', 'M4', 'M5', 'wc1', 'wc2', 'wc3', 'wc4', 'wc5', 'wc',
                                      'Makespan',
                                      "Slack_M1", "Slack_M2", "Slack_M3", "Slack_M4", "Slack_M5",
                                      'assign_M1', 'assign_M2', 'assign_M3', 'assign_M4', 'assign_M5',
                                      'assign_Makespan',
                                      'assign_WC1', 'assign_WC2', 'assign_WC3', 'assign_WC4', 'assign_WC5', "assign_WC",
                                      'dMakespan',
                                      "RESULT"], index= index)

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
                                      'dMakespan',
                                      "RESULT"], index= index)

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
                                      'dMakespan',
                                      "RESULT"], index= index)

    return datas


def GeneratedispatchingSet(data, originalData, datas, machineNum):

    min_group = int(data['loop_rank'].min())
    max_group = int(data['loop_rank'].max())

    for i in range(min_group, max_group + 1):
        data_i_group = data[data['loop_rank'] == i]
        data_i_group_original = originalData[data['loop_rank'] == i]

        datas = GernerateAssigneStateAttrSet(datas, data_i_group, data_i_group_original, machineNum)

    datas['loop_rank'] = data['loop_rank']
    cols = list(datas.columns.values)
    cols = cols[:-2]
    cols.insert(0, "loop_rank")
    cols.insert(0, "RESULT")
    datas = datas[cols]


    return datas


# 실제로 할당된 작업(A)와 할당되지 않은 작업(B)의 차이값으로 데이터를 변환
def data_loop_diff(data, originalData, flag, machineNum):
    datas = None
    if (flag == 0):  ##원래 값만

        datas = CreateColumns(machineNum)

        datas = GeneratedispatchingSet(data, originalData, datas, machineNum)

    return datas


def Set_Value_By_ComparisionFlag(data):
    if (type(data) == "dataframe"):
        mask_a = data.loc[:, :] < 0
        mask_b = data.loc[:, :] > 0

        data[mask_a] = -1
        data[mask_b] = 1
    else:
        data[data < 0] = -1
        data[data > 0] = 1

    return data

def ConvertToLTR(data):

    i = 1

    for column in data:
        if(column == "RESULT"):
            continue
        elif(column == "loop_rank"):
            data[column] = "qid:"+data[column].astype(str)
        else:
            data[column] = str(i)+":"+data[column].astype(str)
            i = i + 1

    return data

def save_data(group_data,output_feature,output_group):
    if len(group_data) == 0:
        return

    output_group.write(str(len(group_data))+"\n")
    for data in group_data:
        # only include nonzero features
        feats = [ p for p in data[2:] if float(p.split(':')[1]) != 0.0 ]
        output_feature.write(data[0] + " " + " ".join(feats) + "\n")

############################################################################################
############################################################################################
# Main Logic
############################################################################################
############################################################################################

flag = 0  ## 원래 값만

mode = "TRAIN"
# mode="TEST"

# '''MIP 모델링 분류'''
# # f = 0  # 원래 / 근사최적해
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

list_ = [[0, 2, 10], [0, 2, 20], [0, 5, 25], [0, 5, 50], [0, 10, 50], [0, 10, 100], [0, 15, 75], [0, 15, 150]
                , [1, 2, 10], [1, 2, 20], [1, 5, 25], [1, 5, 50], [1, 10, 50]
                # , [1, 10, 100]
                , [1, 15, 75]
                # , [1, 15, 150]
             ]
# list_ = [
#     # [0, 2, 10], [0, 2, 20], [0, 5, 25], [0, 5, 50], [0, 10, 50], [0, 10, 100], [0, 15, 75], [0, 15, 150]
#                 [1, 2, 10], [1, 2, 20], [1, 5, 25], [1, 5, 50], [1, 10, 50]
#                 # , [1, 10, 100]
#                 , [1, 15, 75]
#                 # , [1, 15, 150]
#              ]

# relevanceType 1 = 1or0, relevanceType 2 = 5or0, relevanceType 3 = 10or0
relevanceType = 1

approach = "PairwiseLTR"

for f, machineNum, jobNum in list_:

    # if(machineNum!=15):
    #     continue
    # else:
    #     print("")

    fileLocation = "C:\\Users\\Han\\Google 드라이브\\oneDrive Bak\\대학원\\논문\\석사졸업논문\\실험\\ScheduML_data\\"  ## File 위치
    fileLocation = fileLocation+"M"+str(machineNum)+"N"+str(jobNum)

    # if f == 0:
    #     fileName = '\\mip-a_dispatch\\MIP-A_Dispatching'  ## xlsx file 명
    # else:
    #     fileName = '\\mip-o_dispatch\\MIP-O_Dispatching'

    if f == 0:
        fileName = '\\3.MIP_A\\3-1.ToDispatching\\MIP_A_Dispatching'  ## xlsx file 명
    else:
        fileName = '\\4.MIP_O\\4-1.ToDispatching\\MIP_O_Dispatching'

    if(approach != "PairwiseLTR"):
        # fileName = 'M2_N50_DataGeneration'  ## xlsx file 명
        data = pd.ExcelFile(fileLocation + fileName +".xlsx")
        data = data.parse(data.sheet_names[0])
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


        ## DataPreprocessing
        originalData, data = data_preprocessing(data, machineNum, flag)

        ## Listwising
        if mode == "TRAIN": data = data_loop_diff(data, originalData, flag, machineNum)

        ## Relevance Setting
        if relevanceType==1 : data.loc[data[data["RESULT"] == 1].index, "RESULT"] = 1
        elif relevanceType==2 : data.loc[data[data["RESULT"] == 1].index, "RESULT"] = 5
        else: data.loc[data[data["RESULT"] == 1].index, "RESULT"] = 10

        ## Converting
        data = ConvertToLTR(data)

        ## Exporting
        if flag == 0:
            if relevanceType == 1:
                data.to_excel(fileLocation + fileName + "_training.xlsx", index=False)
                data.to_csv(fileLocation + fileName + "_training.txt", index=False, header=None, sep=" ")
            else:
                data.to_excel(fileLocation + fileName + "_training_relType"+str(relevanceType)+".xlsx", index=False)
                data.to_csv(fileLocation + fileName + "_training_relType"+str(relevanceType)+".txt", index=False, header=None, sep=" ")
    # PairwiseLTR
    # .txt -> .train .group
    else:
        fi = fileLocation + fileName + "_training.txt"
        output_feature = fileLocation + fileName + "_training.train"
        output_group = fileLocation + fileName + "_training.group"

        fi = open(fi)
        output_feature = open(output_feature, "w")
        output_group = open(output_group, "w")

        group_data = []
        group = ""
        for line in fi:
            if not line:
                break
            if "#" in line:
                line = line[:line.index("#")]
            splits = line.strip().split(" ")
            if splits[1] != group:
                save_data(group_data, output_feature, output_group)
                group_data = []
            group = splits[1]
            group_data.append(splits)

        save_data(group_data, output_feature, output_group)

        fi.close()
        output_feature.close()
        output_group.close()