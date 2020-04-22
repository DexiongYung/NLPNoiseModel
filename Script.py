import pandas as pd
import torch
from Utilities.Convert import *

df = pd.read_csv('Data/mispelled.csv')
data = []

for idx, row in df.iterrows():
    word = row['Correct']
    noised = row['Noised']

    if levenshtein(word, noised) < 5:
        data.append([noised, word])
    
df = pd.DataFrame(data, columns=['Noised', 'Correct'])

df.to_csv('Data/misppeled_2.csv', index=False)

# df = pd.read_csv('Data/test.csv')

# for i in range(len(df)):
#     full_name = df.iloc[i]['name']
#     fn = df.iloc[i]['first']
#     mn = df.iloc[i]['middle']
#     ln = df.iloc[i]['last']

#     noised_fns = test_w_beam([fn])
#     noised_lns = test_w_beam([ln])

#     noised_fn = get_hamming_winner(noised_fns, fn)
#     noised_ln = get_hamming_winner(noised_lns, ln)

#     full_name = full_name.replace(fn, noised_fn)
#     full_name = full_name.replace(ln, noised_ln)

#     if isinstance(mn,str) and len(mn) > 1:
#         noised_mns = test_w_beam([mn])
#         noised_mn = get_hamming_winner(noised_mns, mn)
#         full_name = full_name.replace(mn, noised_mn)

#     df.at[i, 'name'] = full_name

# df.to_csv('Data/noised_test2.csv', index=False)

# df = pd.read_csv('Data/LastNames.csv')

# save_every = 5000

# data = []

# for i in range(len(df)):
#     name = df.iloc[i]['name']
#     noised_names = test_w_beam([name])
#     noised = get_hamming_winner(noised_names, name)
#     data.append([name, noised])

#     if i != 0 and i % save_every == 0:
#         df = pd.DataFrame(data, columns=['name', 'noised'])
#         df.to_csv('Data/NoisedLast.csv', index=False)

# df = pd.DataFrame(data, columns=['name', 'noised'])
# df.to_csv('Data/NoisedLast.csv', index=False)