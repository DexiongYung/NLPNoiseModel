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


directory = 'Data/kaggle'
files = os.listdir(directory)
main = pd.read_csv('Data/mispelled2.csv')

for file_pth in files:
    df = convert_kaggle_to_csv(f'{directory}/{file_pth}')
    main = main.append(df)

main.to_csv('Data/mispelled3.csv', index=False)
