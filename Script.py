import pandas as pd
import torch

df = pd.read_csv('Data/mispelled.csv')

for i in range(len(df)):
    df.iloc[i].Correct = str(df.iloc[i].Correct).capitalize()

df.to_csv('Data/mispelled2.csv')