import pandas as pd
import torch

def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

df = pd.read_csv('Data/NoisedLast.csv')

data =[]

for i in range(len(df)):
    noised = df.iloc[i]['noised']
    name = df.iloc[i]['name']

    distance = hamming_distance(name, noised)
    name_len = len(name)

    if distance < (1/name_len):
        data.append([noised, name])

df = pd.DataFrame(data, columns=['noised', 'name'])

df.to_csv('Data/NoisedLast.csv')

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