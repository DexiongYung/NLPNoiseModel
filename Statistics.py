import pandas
import string
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


def get_levenshtein_stats(noiseds: list, cleans: list):
    clean_len = len(cleans)
    noise_len = len(noiseds)

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    counts_dict = dict()

    for idx in range(clean_len):
        noised = str(noiseds[idx])
        correct = str(cleans[idx])

        distance = levenshtein(noised, correct)

        if distance in counts_dict.keys():
            counts_dict[distance] += 1
        else:
            counts_dict[distance] = 1

    for key in counts_dict.keys():
        counts_dict[key] = round(counts_dict[key] / clean_len, 4)

    return counts_dict


def get_edit_distributions_stats(noiseds: list, cleans: list):
    clean_len = len(cleans)
    noise_len = len(noiseds)

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    ins_total, dels_total, subs_total, total = 0, 0, 0, 0

    for idx in range(clean_len):
        noised = str(noiseds[idx])
        correct = str(cleans[idx])

        ins, dels, subs = get_levenshtein_w_counts(noised, correct)
        ins_total += ins
        dels_total += dels
        subs_total += subs

    total = ins_total + dels_total + subs_total

    return ins_total/total, dels_total/total, subs_total/total


def get_percent_of_noise_outside_clean(clean: list, noise: list):
    clean_len = len(clean)
    noise_len = len(noise)

    outside_count = 0

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    for idx in range(clean_len):
        noised_word = noise[idx]
        clean_word = clean[idx]

        for char in noised_word:
            if char not in clean_word:
                outside_count += 1

    return float(outside_count/noise_len)


def get_percent_of_digit_noise_outside_clean(clean: list, noise: list):
    clean_len = len(clean)
    noise_len = len(noise)

    outside_count = 0

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    for idx in range(clean_len):
        noised_word = noise[idx]
        clean_word = clean[idx]

        for char in noised_word:
            if char not in clean_word and char in string.digits:
                outside_count += 1

    return float(outside_count/noise_len)


def get_percent_of_punc_noise_outside_clean(clean: list, noise: list):
    clean_len = len(clean)
    noise_len = len(noise)

    outside_count = 0

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    for idx in range(clean_len):
        noised_word = noise[idx]
        clean_word = clean[idx]

        for char in noised_word:
            if char not in clean_word and char in string.punctuation:
                outside_count += 1

    return float(outside_count/noise_len)


def get_percent_of_alpha_noise_outside_clean(clean: list, noise: list):
    clean_len = len(clean)
    noise_len = len(noise)

    outside_count = 0

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    for idx in range(clean_len):
        noised_word = noise[idx]
        clean_word = clean[idx]

        for char in noised_word:
            if char not in clean_word and char in string.ascii_letters:
                outside_count += 1

    return float(outside_count/noise_len)


df = pandas.read_csv(file_path)

get_edit_distributions_stats(df)
