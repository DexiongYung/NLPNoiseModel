import pandas as pd
import torch

df = pd.read_csv('Data/fb_moe.csv')

data = []
for i in range(len(df)):
    sample = int(torch.distributions.Bernoulli(torch.FloatTensor([.8])).sample())

    if sample == 1:
        df.iloc[i].y = str(df.iloc[i].y).capitalize()

        sample_1 = int(torch.distributions.Bernoulli(torch.FloatTensor([.6])).sample())

        if sample_1 == 1:
            df.iloc[i].x = str(df.iloc[i].x).capitalize()

    data.append([df.iloc[i].x, df.iloc[i].y])    

df = pd.DataFrame(data, columns = ['Noised', 'Correct'])     
df.to_csv('Data/fb_moe2.csv')