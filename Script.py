import pandas as pd
import torch
import os
from Utilities.Convert import *


def convert_efc_to_csv(path: str):
    df = pd.read_csv(path, sep='\t')
    df_len = len(df)

    data = []

    for _, row in df.iterrows():
        correction = str(row['Correction'])
        errors = str(row['Errors'][1:-1]).split(', ')

        for word in errors:
            data.append([word, correction])

    return pd.DataFrame(data, columns=['Noised', 'Correct'])


def convert_kaggle_to_csv(path: str):
    f = open(path, "r")
    data = []

    for line in f:
        split = line.split(':')
        correct = split[0]
        noiseds = split[1]
        noiseds = noiseds.split(' ')

        for noised in noiseds:
            if len(noised) > 0:
                data.append([noised.replace('\n', ''), correct])

    return pd.DataFrame(data, columns=['Noised', 'Correct'])


df = pd.read_csv('Data/mispelled_best.csv')

df['Correct'] = df["Correct"].apply(lambda x: ''.join(
    [" " if ord(i) < 32 or ord(i) > 126 else i for i in str(x)]))
df['Noised'] = df["Noised"].apply(lambda x: ''.join(
    [" " if ord(i) < 32 or ord(i) > 126 else i for i in str(x)]))
df.dropna()
df.drop_duplicates()
df.to_csv('Data/mispelled.csv', index=False)
