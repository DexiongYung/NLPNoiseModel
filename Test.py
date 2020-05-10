import argparse
import pandas as pd
import torch
from Utilities.Json import load_json
from Utilities.Convert import *
from Model.Seq2Seq import Encoder, Decoder

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Config json CONFIG_NAME',
                    nargs='?', default='noise', type=str)
parser.add_argument('--k', help='Number of width for beam search',
                    nargs='?', default=101, type=int)
parser.add_argument('--name', help='Name to test one',
                    nargs='?', default='Jason', type=str)

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


def test(x: list):
    name_length = len(x[0])

    src_x = [c for c in x[0]]

    src = indexTensor(x, name_length, CHARACTERS).to(DEVICE)
    lng = lengthTensor(x).to(DEVICE)

    hidden = encoder.forward(src, lng)

    name = ''

    lstm_input = targetTensor([SOS], 1, CHARACTERS).to(DEVICE)
    sampled_char = SOS
    for i in range(100):
        decoder_out, hidden = decoder.forward(lstm_input, hidden)
        decoder_out = decoder_out.reshape(NUM_CHARS)
        lstm_probs = torch.softmax(decoder_out, dim=0)
        sample = int(torch.distributions.Categorical(lstm_probs).sample())
        sampled_char = CHARACTERS[sample]

        if sampled_char == EOS:
            break

        name += sampled_char
        lstm_input = targetTensor([sampled_char], 1, CHARACTERS).to(DEVICE)

    return name


def test_w_beam(x: list):
    name_length = len(x[0])

    src_x = [c for c in x[0]]

    src = indexTensor(x, name_length, CHARACTERS).to(DEVICE)
    lng = lengthTensor(x).to(DEVICE)

    hidden = encoder.forward(src, lng)

    return [''.join(c for c in name[:-1]) for name, score, hidden in top_k_beam_search(decoder, hidden, K)]


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
        'EOS', '') for name in noised]

    return get_levenshtein_winner(noised_strs, name)