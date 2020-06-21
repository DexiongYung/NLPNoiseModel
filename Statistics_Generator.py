import pandas
import string
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from Utilities.Distance import *
from Utilities.Plot import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='Path to csv file with noised and clean columns',
                        nargs='?', default='Data/mispelled_pure_noised.csv', type=str)
    args = parser.parse_args()
    file_path = args.file_path

    df = pandas.read_csv(file_path)


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

        ins, dels, subs = get_levenshtein_w_counts(correct, noised)
        ins_total += ins
        dels_total += dels
        subs_total += subs

    total = ins_total + dels_total + subs_total

    return float(ins_total/total), float(dels_total/total), float(subs_total/total)


def get_edit_percents_distribution(noiseds: list, cleans: list):
    clean_len = len(cleans)
    noise_len = len(noiseds)

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    sub_percents = []
    del_percents = []
    ins_percents = []

    for idx in range(clean_len):
        noised = str(noiseds[idx])
        correct = str(cleans[idx])

        ins, dels, subs = get_levenshtein_w_counts(correct, noised)
        total = ins + dels + subs

        del_percents.append(float(dels/total))
        sub_percents.append(float(subs/total))
        ins_percents.append(float(ins/total))

    return ins_percents, sub_percents, del_percents


def get_percent_of_noise_outside_clean(clean: list, noise: list):
    return get_percent(clean, noise, string.printable)


def get_percent_of_digit_noise_outside_clean(clean: list, noise: list):
    return get_percent(clean, noise, string.digits)


def get_percent_of_white_space_outside_clean(clean: list, noise: list):
    return get_percent(clean, noise, string.whitespace)


def get_percent_of_punc_noise_outside_clean(clean: list, noise: list):
    return get_percent(clean, noise, string.punctuation)


def get_percent_of_alpha_noise_outside_clean(clean: list, noise: list):
    return get_percent(clean, noise, string.ascii_letters)


def get_percent_of_upper_alpha_noise_outside_clean(clean: list, noise: list):
    return get_percent(clean, noise, string.ascii_uppercase)


def get_percent_of_lower_alpha_noise_outside_clean(clean: list, noise: list):
    return get_percent(clean, noise, string.ascii_lowercase)


def get_percent_of_upper_vowel_noise_outside_clean(clean: list, noise: list):
    return get_percent(clean, noise, 'AEIOU')


def get_percent_of_lower_vowel_noise_outside_clean(clean: list, noise: list):
    return get_percent(clean, noise, 'aeiou')


def get_percent_of_vowel_noise_outside_clean(clean: list, noise: list):
    return get_percent(clean, noise, 'AEIOUaeiou')


def get_percent_of_consonants_noise_outside_clean(clean: list, noise: list):
    set_of_letters = ''.join(
        [c for c in string.ascii_letters if c not in 'aeiouAEIOU'])

    return get_percent(clean, noise, set_of_letters)


def get_percent_of_upper_consonants_noise(clean: list, noise: list):
    set_of_letters = ''.join(
        [c for c in string.ascii_uppercase if c not in 'AEIOU'])

    return get_percent(clean, noise, set_of_letters)


def get_percent_of_lower_consonants_noise(clean: list, noise: list):
    set_of_letters = ''.join(
        [c for c in string.ascii_lowercase if c not in 'aeiou'])

    return get_percent(clean, noise, set_of_letters)


def get_percent(clean_lst: list, noised_lst: list, exclusion_str: str):
    clean_len = len(clean_lst)
    noise_len = len(noised_lst)

    levenshtein_sum, outside_count = 0, 0

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')
    else:
        length = clean_len

    for idx in range(length):
        noised_word = str(noised_lst[idx])
        clean_word = str(clean_lst[idx])

        levenshtein_sum += levenshtein(clean_word, noised_word)
        outside_count += count_outside_clean_in_noisy_in_set(
            clean_word, noised_word, exclusion_str)

    return float(outside_count/levenshtein_sum)


def count_outside_clean_in_noisy_in_set(clean: str, noisy: str, set_str: str):
    outside_count = 0

    for char in noisy:
        if char not in clean and char in set_str:
            outside_count += 1

    return outside_count


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


def get_stats_of_noised_len_to_clean(clean: list, noise: list):
    '''
    Gets stats on length of noised compared to clean representation. Divided into 3 categories 'larger', 'smaller' and 'equal'
    in length
    '''
    clean_len = len(clean)
    noise_len = len(noise)

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    count_smaller = 0
    count_equal = 0
    count_larger = 0

    for i in range(clean_len):
        clean_word_len = len(str(clean[i]))
        noised_word_len = len(str(noise[i]))

        if noised_word_len > clean_word_len:
            count_larger += 1
        elif noised_word_len < clean_word_len:
            count_smaller += 1
        else:
            count_equal += 1

    return float(count_larger/clean_len), float(count_smaller/clean_len), float(count_equal/clean_len)


def get_percent_of_duplicate_char_noise(clean: list, noise: list):
    clean_len = len(clean)
    noise_len = len(noise)

    duplicated_char_count = 0
    total_edit_count = 0

    if clean_len != noise_len:
        raise Exception('Clean list and noise list are not the same length')

    for i in range(clean_len):
        clean_word = str(clean[i])
        noised_word = str(noise[i])

        total_edit_count += levenshtein(clean_word, noised_word)

        def get_letter_count_dict(word: str):
            letter_count_dict = dict()

            for j in range(len(word)):
                char = word[j]
                letter_count_dict[char] = letter_count_dict[char] + \
                    1 if char in letter_count_dict.keys() else 1

            return letter_count_dict

        clean_word_char_dict = get_letter_count_dict(clean_word)
        noised_word_char_dict = get_letter_count_dict(noised_word)

        for key in clean_word_char_dict.keys():
            noised_key_count = noised_word_char_dict[key] if key in noised_word_char_dict.keys(
            ) else 0
            clean_key_count = clean_word_char_dict[key]

            difference = clean_key_count - noised_key_count

            if difference < 0:
                duplicated_char_count += abs(difference)

    return float(duplicated_char_count/total_edit_count)


for i in range(1,19):
    curr_df = df[df.Correct.str.len() == i]
    correct_list = curr_df.Correct.tolist()
    noised_list = curr_df.Noised.tolist()

    ins, dels, subs = get_edit_distributions_percents(correct_list, noised_list)

    print(f'length = {i}: & {ins} & {dels} & {subs} \\\\')