import pandas as pd

df = pd.read_csv('Data/fb_moe.csv')
df = df.dropna()
df = df[df.x.str.isalpha()]
df.to_csv('Data/fb_moe2.csv', index=False)
