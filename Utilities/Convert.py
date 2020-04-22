import torch
import math
import Constants

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


def targetsTensor(names: list, max_len: int, allowed_chars: list):
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


def top_k_beam_search(hidden: torch.Tensor, k: int = 6, penalty: float = 4.0):
    input = targetsTensor([SOS], 1, CHARACTERS).to(DEVICE)
    output, hidden = decoder.forward(input, hidden)
    output = output.reshape(NUM_CHAR)
    probs = torch.exp(output)
    EOS_idx = CHARACTERS.index(EOS)
    probs[EOS_idx] = 0
    top_k_probs, top_k_idx = torch.topk(probs, k, dim=0)

    top_k = []
    prev_chars = []
    for i in range(len(top_k_idx)):
        prev_char = CHARACTERS[top_k_idx[i].item()]

        if i == 0:
            top_k.append(
                ([prev_char], -math.log(top_k_probs[i].item()) + penalty, hidden))
        else:
            top_k.append(
                ([prev_char], -math.log(top_k_probs[i].item()), hidden))

        prev_chars.append(prev_char)

    while EOS not in prev_chars:
        prev_chars = []
        hypotheses = []

        for name, score, hidden in top_k:
            prev_char = name[-1]
            input = targetsTensor([prev_char], 1, CHARACTERS).to(DEVICE)
            output, hidden = decoder.forward(input, hidden)
            probs = torch.exp(output)
            top_k_probs, top_k_idx = torch.topk(probs, k, dim=2)
            top_k_probs = top_k_probs.reshape(k)
            top_k_idx = top_k_idx.reshape(k)

            for i in range(len(top_k_probs)):
                current_char = CHARACTERS[top_k_idx[i]]
                current_prob = top_k_probs[i]
                name_copy = name.copy()
                name_copy.append(current_char)
                new_score = score + -math.log(current_prob)
                hypotheses.append((name_copy, new_score, hidden))

        hypotheses.sort(key=lambda x: x[1])
        top_k = hypotheses[:k]
        top_k[0] = top_k[0][0], top_k[0][1] + penalty, top_k[0][2]
        prev_chars = [name[-1] for name, probs, hidden in top_k]

    return top_k


def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))


def get_hamming_winner(noised_names: list, name: str):
    distance = math.inf
    winner = ''
    for noised_name in noised_names:
        output = ''
        for char in noised_name:
            if char is not EOS:
                output += char
        output_hash = hashlib.md5(output.encode()).hexdigest()
        name_hash = hashlib.md5(name.encode()).hexdigest()
        curr_distance = hamming_distance(name_hash, output_hash)

        if curr_distance < distance and curr_distance != 0:
            distance = curr_distance
            winner = output
    return winner


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_levenshtein_winner(noised_names: list, name: str):
    distance = math.inf
    winner = ''
    for noised_name in noised_names:
        output = ''
        for char in noised_name:
            if char is not EOS:
                output += char
        curr_distance = levenshtein(output, name)

        if curr_distance < distance and curr_distance != 0:
            distance = curr_distance
            winner = output
    return winner
