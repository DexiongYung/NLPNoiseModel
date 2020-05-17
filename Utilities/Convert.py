import torch
import math
from Constants import DEVICE


def indexTensor(names: list, max_len: int, allowed_chars: list):
    tensor = torch.zeros(max_len, len(names)).type(torch.LongTensor)
    for i, name in enumerate(names):
        for j, letter in enumerate(name):
            index = allowed_chars.index(letter)

            if index < 0:
                raise Exception(
                    f'{names[j][i]} is not a char in {allowed_chars}')

            tensor[j][i] = index
    return tensor


def lengthTensor(names: list):
    tensor = torch.zeros(len(names)).type(torch.LongTensor)
    for i, name in enumerate(names):
        tensor[i] = len(name)

    return tensor


def targetTensor(names: list, max_len: int, allowed_chars: list):
    batch_sz = len(names)
    ret = torch.zeros(max_len, batch_sz).type(torch.LongTensor)
    for i in range(max_len):
        for j in range(batch_sz):
            index = allowed_chars.index(names[j][i])

            if index < 0:
                raise Exception(
                    f'{names[j][i]} is not a char in {allowed_chars}')

            ret[i][j] = index
    return ret