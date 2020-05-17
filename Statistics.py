import pandas
import string
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from Utilities.Distance import *
from Utilities.Plot import *

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


def get_edit_distributions_percents(noiseds: list, cleans: list):
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

    return float(ins_total/total), float(dels_total/total), float(subs_total/total)


def get_percent_of_noise_outside_clean(clean: list, noise: list):
    clean_len = len(clean)
    noise_len = len(noise)

    outside_count = 0

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    for idx in range(clean_len):
        noised_word = str(noise[idx])
        clean_word = str(clean[idx])

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
        noised_word = str(noise[idx])
        clean_word = str(clean[idx])

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
        noised_word = str(noise[idx])
        clean_word = str(clean[idx])

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
        noised_word = str(noise[idx])
        clean_word = str(clean[idx])

        for char in noised_word:
            if char not in clean_word and char in string.ascii_letters:
                outside_count += 1

    return float(outside_count/noise_len)


def get_percent_of_upper_alpha_noise_outside_clean(clean: list, noise: list):
    clean_len = len(clean)
    noise_len = len(noise)

    outside_count = 0

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    for idx in range(clean_len):
        noised_word = str(noise[idx])
        clean_word = str(clean[idx])

        for char in noised_word:
            if char not in clean_word and char in string.ascii_uppercase:
                outside_count += 1

    return float(outside_count/noise_len)


def get_percent_of_lower_alpha_noise_outside_clean(clean: list, noise: list):
    clean_len = len(clean)
    noise_len = len(noise)

    outside_count = 0

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    for idx in range(clean_len):
        noised_word = str(noise[idx])
        clean_word = str(clean[idx])

        for char in noised_word:
            if char not in clean_word and char in string.ascii_lowercase:
                outside_count += 1

    return float(outside_count/noise_len)


def get_percent_of_vowel_noise_outside_clean(clean: list, noise: list):
    clean_len = len(clean)
    noise_len = len(noise)

    outside_count = 0

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    for idx in range(clean_len):
        noised_word = str(noise[idx])
        clean_word = str(clean[idx])

        for char in noised_word:
            if char not in clean_word and char in 'aeiouAEIOU':
                outside_count += 1

    return float(outside_count/noise_len)


def get_percent_of_consonants_noise_outside_clean(clean: list, noise: list):
    clean_len = len(clean)
    noise_len = len(noise)

    outside_count = 0

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    for idx in range(clean_len):
        noised_word = str(noise[idx])
        clean_word = str(clean[idx])

        for char in noised_word:
            if char not in clean_word and char in string.ascii_letters and char not in 'aeiouAEIOU':
                outside_count += 1

    return float(outside_count/noise_len)


def get_points_for_edit_idx_to_clean_length(clean: list, noise: list):
    clean_len = len(clean)
    noise_len = len(noise)
    x, y = [], []

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    for idx in range(clean_len):
        noised_word = str(noise[idx])
        clean_word = str(clean[idx])

        idxes_list = get_indexes_of_edits(clean_word, noised_word)

        x.extend([len(clean_word)] * len(idxes_list))
        y.extend(idxes_list)

    # x representes the list of clean word length, y is the index in the word that had the error
    return x, y


df = pandas.read_csv(file_path)
correct_list = list(df.Correct)
noised_list = list(df.Noised)

x, y = get_points_for_edit_idx_to_clean_length(correct_list, noised_list)
show_scatter_plot(x, y, 'word length', 'error index')
