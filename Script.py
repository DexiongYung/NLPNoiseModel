import pandas as pd
import torch

df = pd.read_csv('Data/mispelled.csv')

for i in range(len(df)):
    sample = int(torch.distributions.Bernoulli(torch.FloatTensor([0.75])).sample())

    if sample == 1:
        df.iloc[i].Correct = str(df.iloc[i].Correct).capitalize()
        df.iloc[i].Noised = str(df.iloc[i].Noised).capitalize()

df.to_csv('Data/mispelled2.csv')