import torch
import string
from Statistics import *

all_chars = string.printable
white_sps = string.whitespace
punc = string.punctuation
digits = string.digits
upper_vowel_alpha = 'AEIOU'
upper_consonant_alpha = string.ascii_uppercase

for c in upper_vowel_alpha:
    upper_consonant_alpha = upper_consonant_alpha.replace(c, '')

lower_vowel_alpha = 'aeiou'
lower_consonant_alpha = string.ascii_lowercase

for c in lower_vowel_alpha:
    lower_consonant_alpha = lower_consonant_alpha.replace(c, '')

num_chars = len(all_chars)
num_white_spcs = len(white_sps)
num_punc = len(punc)
num_upper_vowel = len(upper_vowel_alpha)
num_upper_consonant = len(upper_consonant_alpha)
num_lower_vowel = len(lower_vowel_alpha)
num_lower_consonant = len(lower_consonant_alpha)
num_digits = len(digits)

dist_template = num_chars * [0.0]

for c in white_sps:
    dist_template[all_chars.index(c)] = (
        1/num_white_spcs) * white_sp_os_word_perc

for c in punc:
    dist_template[all_chars.index(c)] = (1/num_punc) * punc_os_word_perc

for c in digits:
    dist_template[all_chars.index(c)] = (1/num_digits) * digit_os_word_perc


def sample_number_edits(word_len: int):
    '''
        Sample the number of edits for a clean word given the length
    '''
    if word_len > 18:
        raise Exception('No statistics on words greater than length 18')

    categorical = torch.FloatTensor([distance[f'lev_dist_{word_len}']])

    return int(torch.distributions.Categorical(categorical).sample().item()) + 1


def sample_edit_types(word_len: int, num_edits: int):
    '''
        Sample the type of edits given the clean word length and number of edits
    '''
    if word_len > 18:
        raise Exception('No statistics on words greater than length 18')

    categorical = torch.FloatTensor([edit[f'edit_cate_{word_len}']])
    samples = []

    for i in range(num_edits):
        # 0 = insert, 1 = delete, 2 = substitution
        samples.append(
            int(torch.distributions.Categorical(categorical).sample().item()))

    return samples


def sample_substitution_edit(char: str, chars_in_word: list):
    '''
        Sample the letter for a substitution edit
    '''
    categorical = dist_template.copy()
    num_chars_in_word = len(chars_in_word)

    for c in chars_in_word:
        if c is char:
            categorical[all_chars.index(c)] = 0
        else:
            categorical[all_chars.index(c)] = (
                1/(num_chars_in_word - 1)) * char_within_word_perc

    upper_vowels = upper_vowel_alpha
    lower_vowels = lower_vowel_alpha
    upper_consonants = upper_consonant_alpha
    lower_consonants = lower_consonant_alpha

    for c in chars_in_word:
        upper_vowels = upper_vowels.replace(c, '')
        lower_vowels = lower_vowels.replace(c, '')
        upper_consonants = upper_consonants.replace(c, '')
        lower_consonants = lower_consonants.replace(c, '')

    num_upper_vowels = len(upper_vowels)
    num_lower_vowels = len(lower_vowels)
    num_upper_consonants = len(upper_consonants)
    num_lower_consonants = len(lower_consonants)

    for c in upper_vowels:
        categorical[all_chars.index(c)] = (
            1/num_upper_vowels) * upper_vowel_os_word_perc

    for c in lower_vowels:
        categorical[all_chars.index(c)] = (
            1/num_lower_vowels) * lower_vowel_os_word_perc

    for c in upper_consonants:
        categorical[all_chars.index(c)] = (
            1/num_upper_consonants) * upper_consonants_os_word_perc

    for c in lower_consonants:
        categorical[all_chars.index(c)] = (
            1/num_lower_consonants) * lower_consonants_os_word_perc

    index = int(torch.distributions.Categorical(
        torch.FloatTensor(categorical)).sample().item())

    return all_chars[index]


def sample_insertion_edit(char: str, chars_in_word: list):
    '''
        Sample the letter for a insertion edit
    '''
    categorical = dist_template.copy()
    num_chars_in_word = len(chars_in_word)

    for c in chars_in_word:
        if c is char:
            categorical[all_chars.index(c)] = (1/2) * char_within_word_perc
        else:
            categorical[all_chars.index(c)] = (1/2) * (
                1/(num_chars_in_word - 1)) * char_within_word_perc

    upper_vowels = upper_vowel_alpha
    lower_vowels = lower_vowel_alpha
    upper_consonants = upper_consonant_alpha
    lower_consonants = lower_consonant_alpha

    for c in chars_in_word:
        upper_vowels = upper_vowels.replace(c, '')
        lower_vowels = lower_vowels.replace(c, '')
        upper_consonants = upper_consonants.replace(c, '')
        lower_consonants = lower_consonants.replace(c, '')

    num_upper_vowels = len(upper_vowels)
    num_lower_vowels = len(lower_vowels)
    num_upper_consonants = len(upper_consonants)
    num_lower_consonants = len(lower_consonants)

    for c in upper_vowels:
        categorical[all_chars.index(c)] = (
            1/num_upper_vowels) * upper_vowel_os_word_perc

    for c in lower_vowels:
        categorical[all_chars.index(c)] = (
            1/num_lower_vowels) * lower_vowel_os_word_perc

    for c in upper_consonants:
        categorical[all_chars.index(c)] = (
            1/num_upper_consonants) * upper_consonants_os_word_perc

    for c in lower_consonants:
        categorical[all_chars.index(c)] = (
            1/num_lower_consonants) * lower_consonants_os_word_perc

    index = int(torch.distributions.Categorical(
        torch.FloatTensor(categorical)).sample().item())

    return all_chars[index]


for i in range(10):
    print(sample_substitution_edit('a', ['t', 'a', 'c', 'o']))
