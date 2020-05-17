import pandas as pd
import torch
import os
from Utilities.Convert import *
from os import listdir
from os.path import isfile, join


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


def convert_sentence_txt_to_list(path: str, num_lines_read: int):
    data = []
    with open(path, 'r', encoding='utf8') as infile:
        count = 0
        lines = []
        for line in infile:
            count += 1
            if count % 7 == 0:
                # each line ends with '\n' which should not be added
                lines.append(line[:-1])
                data.append([lines[2], lines[3], lines[5]])
                count = 0
                lines = []
            else:
                lines.append(line[:-1])

    return data

df = pd.read_csv('Data/Name/FirstNames.csv')
del df['Unnamed: 0']
del df['Unnamed: 0.1']
df.to_csv('Data/Name/Firsts.csv', index=False)

df = pd.read_csv('Data/Name/LastNames.csv')
del df['Unnamed: 0']
df.to_csv('Data/Name/Lasts.csv', index=False)