import pandas
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from Utilities.Convert import levenshtein, get_levenshtein_w_counts

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', help='Path to csv file with noised and clean columns',
                    nargs='?', default='Data/mispelled_best.csv', type=str)
args = parser.parse_args()
file_path = args.file_path


def show_pie_of_levenshtein(df: pandas.DataFrame, title: str):
    counts_dict = dict()

    for _, row in df.iterrows():
        noised = str(row['Noised'])
        correct = str(row['Correct'])

        distance = levenshtein(noised, correct)

        if distance in counts_dict.keys():
            counts_dict[distance] += 1
        else:
            counts_dict[distance] = 1

    distances = counts_dict.keys()
    counts = counts_dict.values()

    fig, ax = plt.subplots(figsize=(len(counts), 3),
                           subplot_kw=dict(aspect="equal"))

    def func(pct, allvals):
        return "{:.1f}%".format(pct)

    wedges, texts, autotexts = ax.pie(counts, autopct=lambda pct: func(pct, counts),
                                      textprops=dict(color="w"))

    ax.legend(wedges, distances,
              title="Distance",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")

    ax.set_title(title)

    plt.show()


def clean_df_of_rows_above_len(df: pandas.DataFrame, max_length: int = 8):
    data = []
    for _, row in df.iterrows():
        noised = str(row['Noised'])
        correct = str(row['Correct'])
        distance = levenshtein(noised, correct)

        if distance < max_length:
            data.append([noised, correct])

    return pandas.DataFrame(data, columns=['Noised', 'Correct'])

def get_levenshtein_stats(df: pandas.DataFrame):
    counts_dict = dict()
    length = len(df)

    for _, row in df.iterrows():
        noised = str(row['Noised'])
        correct = str(row['Correct'])

        distance = levenshtein(noised, correct)

        if distance in counts_dict.keys():
            counts_dict[distance] += 1
        else:
            counts_dict[distance] = 1
    
    for key in counts_dict.keys():
        counts_dict[key] /= length
    
    return counts_dict

def get_edit_distributions_stats(df: pandas.DataFrame):
    length = len(df)
    ins_total, dels_total, subs_total, total = 0, 0, 0, 0

    for _, row in df.iterrows():
        noised = str(row['Noised'])
        correct = str(row['Correct'])

        ins, dels, subs = get_levenshtein_w_counts(noised, correct)
        ins_total += ins
        dels_total += dels
        subs_total += subs
    
    total = ins_total + dels_total + subs_total

    return ins_total/total, dels_total/total, subs_total/total

df = pandas.read_csv(file_path)
print(get_edit_distributions_stats(df))
