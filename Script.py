import pandas as pd
import torch

df = pd.read_csv('Data/mispelled.csv')

df.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()

df.to_csv('Data/mispelled.csv', index=False)