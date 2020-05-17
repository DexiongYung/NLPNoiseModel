import torch
import math
from Utilities.Convert import targetTensor
from Constants import *


def top_k_beam_search(decoder, hidden: torch.Tensor, in_vocab: list, out_vocab: list, sos: str, pad: str, eos: str, k: int = 15, penalty: float = 4.0):
    input = targetTensor([sos], 1, in_vocab).to(DEVICE)
    output, hidden = decoder.forward(input, hidden)
    output = output.reshape(len(out_vocab))
    probs = torch.exp(output)
    EOS_idx = out_vocab.index(eos)
    probs[EOS_idx] = 0
    top_k_probs, top_k_idx = torch.topk(probs, k, dim=0)

    top_k = []
    for i in range(len(top_k_idx)):
        prev_char = out_vocab[top_k_idx[i].item()]

        if i == 0:
            top_k.append(
                ([prev_char], -math.log(top_k_probs[i].item()) + penalty, hidden))
        else:
            top_k.append(
                ([prev_char], -math.log(top_k_probs[i].item()), hidden))

    while not is_EOS_in_all_topk(top_k, eos):
        hypotheses = []

        for name, score, hidden in top_k:
            if eos in name:
                hidden_clone = (hidden[0].clone(), hidden[1].clone())
                hypotheses.append((name.copy(), score, hidden_clone))
            else:
                prev_char = name[-1]
                input = targetTensor([prev_char], 1, in_vocab).to(DEVICE)
                output, hidden = decoder.forward(input, hidden)
                hidden_clone = (hidden[0].clone(), hidden[1].clone())
                probs = torch.exp(output)
                top_k_probs, top_k_idx = torch.topk(probs, k, dim=2)
                top_k_probs = top_k_probs.reshape(k)
                top_k_idx = top_k_idx.reshape(k)

                for i in range(len(top_k_probs)):
                    current_char = out_vocab[top_k_idx[i]]
                    current_prob = top_k_probs[i]
                    name_copy = name.copy()
                    name_copy.append(current_char)
                    new_score = score + -math.log(current_prob)

                    hypotheses.append((name_copy, new_score, hidden_clone))

        hypotheses.sort(key=lambda x: x[1])
        top_k = hypotheses[:k]
        top_k[0] = top_k[0][0], top_k[0][1] + penalty, top_k[0][2]

    return top_k


def top_k_beam_search_graph(decoder, hidden: torch.Tensor, in_vocab: list, out_vocab: list, sos: str, pad: str, eos: str, k: int = 6, penalty: float = 4.0):
    input = targetTensor([sos], 1, in_vocab).to(DEVICE)
    output, hidden = decoder.forward(input, hidden)
    output = output.reshape(len(out_vocab))
    probs = torch.exp(output)
    EOS_idx = out_vocab.index(eos)
    probs[EOS_idx] = 0
    top_k_probs, top_k_idx = torch.topk(probs, k, dim=0)

    top_k = []
    prev_chars = []
    for i in range(len(top_k_idx)):
        prev_char = out_vocab[top_k_idx[i].item()]

        if i == 0:
            top_k.append(
                ([prev_char], -math.log(top_k_probs[i].item()) + penalty, hidden))
        else:
            top_k.append(
                ([prev_char], -math.log(top_k_probs[i].item()), hidden))

        prev_chars.append(prev_char)

    while eos not in prev_chars:
        prev_chars = []
        hypotheses = []

        for name, score, hidden in top_k:
            prev_char = name[-1]
            input = targetTensor([prev_char], 1, in_vocab).to(DEVICE)
            output, hidden = decoder.forward(input, hidden)
            probs = torch.exp(output)
            top_k_probs, top_k_idx = torch.topk(probs, k, dim=2)
            top_k_probs = top_k_probs.reshape(k)
            top_k_idx = top_k_idx.reshape(k)

            for i in range(len(top_k_probs)):
                current_char = out_vocab[top_k_idx[i]]
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


def is_EOS_in_all_topk(topk: list, eos: str):
    '''
        Topk contains a tuple of (name, score, hidden state)
    '''
    for name, _, _ in topk:
        if eos not in name:
            return False
    return True
