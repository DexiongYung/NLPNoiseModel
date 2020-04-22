import argparse
import pandas as pd
from Utilities.Json import load_json
from Utilities.Convert import *
from Model.Seq2Seq import Encoder, Decoder

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Config json name',
                    nargs='?', default='noise', type=str)

args = parser.parse_args()
NAME = args.config
config_json = load_json(f'Config/{NAME}.json')
input_sz = config_json['input_sz']
input = config_json['input']
output_sz = config_json['output_sz']
output = config_json['output']
hidden_sz = config_json['hidden_size']
num_layers = config_json['num_layers']
embed_sz = config_json['embed_dim']
EOS = config_json['EOS']
PAD = config_json['PAD']
PAD_idx = input.index(PAD)

encoder = Encoder(input_sz, hidden_sz, PAD_idx, num_layers, embed_sz)
decoder = Decoder(input_sz, hidden_sz, PAD_idx, num_layers, embed_sz)

encoder.load_state_dict(torch.load(
    f'Checkpoints/{NAME}_encoder.path.tar')['weights'])
decoder.load_state_dict(torch.load(
    f'Checkpoints/{NAME}_decoder.path.tar')['weights'])

encoder.eval()
decoder.eval()


def test(x: list):
    name_length = len(x[0])

    src_x = [c for c in x[0]]

    src = indexTensor(x, name_length, CHARACTERS).to(DEVICE)
    lng = lengthTensor(x).to(DEVICE)

    hidden = encoder.forward(src, lng)

    name = ''

    lstm_input = targetsTensor([SOS], 1, CHARACTERS).to(DEVICE)
    sampled_char = SOS
    for i in range(100):
        decoder_out, hidden = decoder.forward(lstm_input, hidden)
        decoder_out = decoder_out.reshape(NUM_CHAR)
        lstm_probs = torch.softmax(decoder_out, dim=0)
        sample = int(torch.distributions.Categorical(lstm_probs).sample())
        sampled_char = CHARACTERS[sample]

        if sampled_char is EOS:
            break

        name += sampled_char
        lstm_input = targetsTensor([sampled_char], 1, CHARACTERS).to(DEVICE)

    return name


def test_w_beam(x: list):
    encoder.eval()
    decoder.eval()

    name_length = len(x[0])

    src_x = [c for c in x[0]]

    src = indexTensor(x, name_length, CHARACTERS).to(DEVICE)
    lng = lengthTensor(x).to(DEVICE)

    hidden = encoder.forward(src, lng)

    return [name for name, score, hidden in top_k_beam_search(decoder, hidden, 10)]


def noise_test(in_path: str, out_path: str):
    df = pd.read_csv(in_path)

    for i in range(len(df)):
        full_name = df.iloc[i]['name']
        fn = df.iloc[i]['first']
        mn = df.iloc[i]['middle']
        ln = df.iloc[i]['last']

        noised_fns = test_w_beam([fn])
        noised_lns = test_w_beam([ln])

        noised_fn = get_levenshtein_winner(noised_fns, fn)
        noised_ln = get_levenshtein_winner(noised_lns, ln)

        full_name = full_name.replace(fn, noised_fn)
        full_name = full_name.replace(ln, noised_ln)

        if isinstance(mn, str) and len(mn) > 1:
            noised_mns = test_w_beam([mn])
            noised_mn = get_levenshtein_winner(noised_mns, mn)
            full_name = full_name.replace(mn, noised_mn)

        df.at[i, 'name'] = full_name

    df.to_csv(out_path, index=False)
