from pathlib import Path

import numpy as np
import pandas as pd
#import networkx as nx
#import matplotlib.pyplot as plt

from multiprocessing import Process

from tqdm import tqdm

DATA_DIR = Path(r"./orig_data")
MOD_DATA_DIR = Path(r"./mod_data")
df_train = pd.read_csv(MOD_DATA_DIR / "train_A.csv")
df_test = pd.read_csv(DATA_DIR / "test.csv")
#df_train = pd.read_csv(DATA_DIR / "train_small.csv")
#df_test = pd.read_csv(DATA_DIR / "test_small.csv")
df_info = pd.read_csv(DATA_DIR / "info.csv")
#df_test = pd.read_csv(MOD_DATA_DIR / "test_eval.csv")
#df_network = pd.read_csv(DATA_DIR / "network.csv")

df_train_mod = df_train.copy()
df_test_mod = df_test.copy()
#df_train_mod['prev_dalayTime'] = []

def parse_delay_info(infoData):

    prev_data = infoData.iloc[0]
    startTime = infoData.iloc[0]['time']
    endTime = infoData.iloc[0]['time']
    case = infoData.iloc[0]['cse']
    lineName = infoData.iloc[0]['lineName']
    delay_info = []

    for i in range(1, len(infoData)):
        if (prev_data['date'] == infoData.iloc[i]['date']) and (prev_data['lineName'] == infoData.iloc[i]['lineName']) and (prev_data['directionCode1'] == infoData.iloc[i]['directionCode1']):
            endTime = infoData.iloc[i]['time']

        else:
            if (prev_data['date'] != infoData.iloc[i]['date']):
                delay_info.append( {"date":prev_data['date'], "lineName":lineName, "startTime":startTime, "endTime":endTime, "case":case})
            startTime = infoData.iloc[i]['time']
            endTime = infoData.iloc[i]['time']
            lineName = infoData.iloc[i]['lineName']
            case = infoData.iloc[i]['cse']

            prev_data = infoData.iloc[i]

    return sorted(delay_info, key=lambda x: x['date'])

def add_delay_factor(originalData, delay_info):

    delay_case = ["なし"] * len(originalData)
        
    for i in range(len(delay_info)):

        date = delay_info[i]['date']
        lineName = delay_info[i]['lineName']
        startTime = delay_info[i]['startTime']
        endTime = delay_info[i]['endTime']

        case_data = originalData[['id', 'date', 'lineName', 'planArrival']].query('date == @date & lineName == @lineName & planArrival > @startTime & planArrival < @endTime')

        for j in range(len(case_data)):
            id = case_data.iloc[j]['id']
            #print(case_data.index[j])
            delay_case[case_data.index[j]] = delay_info[i]['case']

    #print(df_train.iloc[i])

    return delay_case

def add_prev_DelayTime(originalData):

    prev_trainNo = ''
    prev_dalayTime = []

    for i in tqdm(range(len(originalData))):
    #for i in range(100):
        if (prev_trainNo != originalData.iloc[i]['trainNo']):
            if(np.isnan(originalData.iloc[i]['delayTime'])):
                prev_dalayTime.append(0.0)
            else:
                prev_dalayTime.append(originalData.iloc[i]['delayTime'])
        else:
            if(np.isnan(originalData.iloc[i-1]['delayTime'])):
                #prev_dalayTime.append(prev_dalayTime[i-1])
                prev_dalayTime.append(0.0)
            else:
                prev_dalayTime.append(originalData.iloc[i-1]['delayTime'])
                

        prev_trainNo = originalData.iloc[i]['trainNo']

    #print(df_train.iloc[i])

    return prev_dalayTime

def update_train_csv():

    #global df_info
    #global df_train

    delay_info = parse_delay_info(df_info)

    df_train_mod['prev_dalayTime'] = add_prev_DelayTime(df_train)

    df_train_mod['delay_case'] = add_delay_factor(df_train, delay_info)

    df_train_mod.to_csv(MOD_DATA_DIR / 'train.csv', index=False)

def update_test_csv():

    delay_info = parse_delay_info(df_info)

    df_test_mod['prev_dalayTime'] = add_prev_DelayTime(df_test)

    df_test_mod['delay_case'] = add_delay_factor(df_test, delay_info)

    df_test_mod.to_csv(MOD_DATA_DIR / 'test.csv', index=False)

p1 = Process(target=update_train_csv)
p2 = Process(target=update_test_csv)

p1 = Process(target=update_train_csv)
p1.start()

p2 = Process(target=update_test_csv)
p2.start()

p1.join()
p2.join()
