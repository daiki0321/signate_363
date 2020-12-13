from pathlib import Path

import numpy as np
import pandas as pd
#import networkx as nx
#import matplotlib.pyplot as plt

import threading

from tqdm import tqdm

DATA_DIR = Path(r"./orig_data")
MOD_DATA_DIR = Path(r"./mod_data")
df_train = pd.read_csv(DATA_DIR / "train.csv")
df_test = pd.read_csv(DATA_DIR / "test.csv")

#target が 1のものを削除
drop_index = df_test.index[df_test['target'] == 1]

#条件にマッチしたIndexを削除
df_test = df_test.drop(drop_index)

#delayTime が delayTimeが空のものを削除
drop_index = df_test.index[np.isnan(df_test['delayTime'])]

#条件にマッチしたIndexを削除
df_test = df_test.drop(drop_index)

df_concat = pd.concat([df_train, df_test], sort=False)

df_concat = df_concat.drop("target", axis = 1)

df_concat.to_csv(MOD_DATA_DIR / 'train_A.csv', index=False)


