import argparse
import pandas as pd
import torch
from Utilities.Json import load_json
from Utilities.Convert import *
from Utilities.Distance import *
from Utilities.Search import *
from Model.Seq2Seq import Encoder, Decoder
from Prob_Prog_Model import sample_number_edits
from Statistics import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Config json CONFIG_NAME',
                    nargs='?', default='noise', type=str)
parser.add_argument('--k', help='Number of width for beam search',
                    nargs='?', default=6, type=int)
parser.add_argument('--name', help='Name to test one',
                    nargs='?', default='Dylan', type=str)

args = parser.parse_args()
CONFIG_NAME = args.config
NAME = args.name
K = args.k
config_json = load_json(f'Config/{CONFIG_NAME}.json')
input_sz = config_json['input_sz']
input = config_json['input']
output_sz = config_json['output_sz']
output = config_json['output']
hidden_sz = config_json['hidden_size']
num_layers = config_json['num_layers']
embed_sz = config_json['embed_dim']
SOS = config_json['SOS']
EOS = config_json['EOS']
PAD = config_json['PAD']
PAD_idx = input.index(PAD)

encoder = Encoder(input_sz, hidden_sz, PAD_idx,
                  num_layers, embed_sz).to(DEVICE)
decoder = Decoder(input_sz, hidden_sz, PAD_idx,
                  num_layers, embed_sz).to(DEVICE)

encoder.load_state_dict(torch.load(
    f'Checkpoints/{CONFIG_NAME}_encoder.path.tar')['weights'])
decoder.load_state_dict(torch.load(
    f'Checkpoints/{CONFIG_NAME}_decoder.path.tar')['weights'])

encoder.eval()
decoder.eval()



def test_argmax(x: list):
    name_length = len(x[0])

    src = indexTensor(x, name_length, input).to(DEVICE)
    lng = lengthTensor(x).to(DEVICE)

    hidden = encoder.forward(src, lng)

    name = ''

    lstm_input = targetTensor([SOS], 1, output).to(DEVICE)
    maxed_char = SOS
    for i in range(100):
        decoder_out, hidden = decoder.forward(lstm_input, hidden)
        decoder_out = decoder_out.reshape(output_sz)
        lstm_probs = torch.softmax(decoder_out, dim=0)
        maxes = lstm_probs.max(dim=2)
        maxed_char = output[maxes]

        if maxed_char == EOS:
            break

        name += maxed_char
        lstm_input = targetTensor([maxed_char], 1, output).to(DEVICE)

    return name


def test_sample(x: list):
    name_length = len(x[0])

    src_x = [c for c in x[0]]

    src = indexTensor(x, name_length, input).to(DEVICE)
    lng = lengthTensor(x).to(DEVICE)

    hidden = encoder.forward(src, lng)

    name = ''

    lstm_input = targetTensor([SOS], 1, output).to(DEVICE)
    sampled_char = SOS
    for i in range(100):
        decoder_out, hidden = decoder.forward(lstm_input, hidden)
        decoder_out = decoder_out.reshape(output_sz)
        lstm_probs = torch.softmax(decoder_out, dim=0)
        sample = int(torch.distributions.Categorical(lstm_probs).sample())
        sampled_char = output[sample]

        if sampled_char == EOS:
            break

        name += sampled_char
        lstm_input = targetTensor([sampled_char], 1, output).to(DEVICE)

    return name

def test_w_beam(x: list):
    name_length = len(x[0])

    src_x = [c for c in x[0]]

    src = indexTensor(x, name_length, input).to(DEVICE)
    lng = lengthTensor(x).to(DEVICE)

    hidden = encoder.forward(src, lng)

    return [''.join(c for c in name[:-1]) for name, score, hidden in top_k_beam_search(decoder, hidden, input, output, SOS, PAD, EOS, K)]


def noise_test(in_path: str, out_path: str):
    df = pd.read_csv(in_path)

    for i in range(len(df)):
        full_name = df.iloc[i]['name']
        fn = df.iloc[i]['first']
        mn = df.iloc[i]['middle']
        ln = df.iloc[i]['last']

        mn_exists = is_name(mn)
        probability_tensor = None

        if mn_exists:
            probability_tensor = torch.FloatTensor([1/3, 1/3, 1/3])
        else:
            probability_tensor = torch.FloatTensor([1/2, 1/2, 0])

        sample = int(torch.distributions.Categorical(
            probability_tensor).sample())

        if is_name(fn) and sample == 0:
            noised_fn = get_levenshtein_beam_winner(fn)
            full_name = full_name.replace(fn, noised_fn)

        if is_name(mn) and sample == 2:
            noised_mn = get_levenshtein_beam_winner(mn)
            full_name = full_name.replace(mn, noised_mn)

        if is_name(ln) and sample == 1:
            noised_ln = get_levenshtein_beam_winner(ln)
            full_name = full_name.replace(ln, noised_ln)

        df.at[i, 'name'] = full_name

    df.to_csv(out_path, index=False)


def is_name(name: str):
    return isinstance(name, str) and len(name) > 1


def get_levenshtein_beam_winner(name: str):
    noised = test_w_beam([name])
    noised_strs = [''.join(c for c in name).replace(
        EOS, '') for name in noised]

    return get_levenshtein_winner(noised_strs, name)

def mixture_noising(name: str):
    name_len = len(name)
    noised_name = ''

    # Get uniform char probs of edit
    num_edits = sample_number_edits(name_len)
    char_edit_probs = torch.FloatTensor([float(num_edits/ name_len)])

    # Forward Name through Encoder
    src = indexTensor([name], len(name), input).to(DEVICE)
    lng = lengthTensor([name]).to(DEVICE)
    hidden = encoder.forward(src, lng)

    # Forward SOS through decoder with Encoder hidden state
    lstm_input = targetTensor([SOS], 1, output).to(DEVICE)
    out_probs, hidden = decoder.forward(lstm_input, hidden)

    # Get edit type categorical distribution
    edit_cate_dist = torch.FloatTensor(edit[f'edit_cate_{name_len}'])

    for i in range(name_len):
        # Sample if edit
        is_edit = bool(torch.distributions.Bernoulli(
            char_edit_probs).sample().item())
        curr_char = name[i]

        if is_edit:
            # Sample edit type, 0 = insert, 1 = del, 2 = sub
            edit_type = int(torch.distributions.Categorical(
                edit_cate_dist).sample().item())

            if edit_type is 0:
                # Add current char and push through LSTM
                noised_name = noised_name + curr_char
                lstm_input = targetTensor([curr_char], 1, output).to(DEVICE)
                out_probs, hidden = decoder.forward(lstm_input, hidden)

                # Sample inserted char
                out_probs[0][0][output.index(EOS)] = 0.0
                sampled_char = output[torch.distributions.Categorical(out_probs).sample().item()]
                noised_name += sampled_char

                # Forward inserted char through model
                lstm_input = targetTensor([sampled_char], 1, output).to(DEVICE)
                out_probs, hidden = decoder.forward(lstm_input, hidden)
            elif edit_type is 2:
                # Substitution

                # Zero out EOS and current char
                out_probs[0][0][output.index(EOS)] = 0.0
                out_probs[0][0][output.index(curr_char)] = 0.0

                # Sample sub char
                sampled_char = output[torch.distributions.Categorical(out_probs).sample().item()]
                noised_name += sampled_char
                
                # Forward sample
                lstm_input = targetTensor([sampled_char], 1, output).to(DEVICE)
                out_probs, hidden = decoder.forward(lstm_input, hidden)
        else:
            noised_name += curr_char

            # Forward current character through LSTM
            lstm_input = targetTensor([curr_char], 1, output).to(DEVICE)
            out_probs, hidden = decoder.forward(lstm_input, hidden)

    return noised_name

for i in range(10):
    print(mixture_noising(NAME))