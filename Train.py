import torch
import os
import torch.nn as nn
import argparse
import pandas as pd
from Constants import *
from Utilities.Plot import *
from Utilities.Convert import *
from Utilities.Json import *
from torch.utils.data import DataLoader
from Model.Seq2Seq import Encoder, Decoder
from Dataset.WordDataset import WordDataset

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session',
                    nargs='?', default='noise', type=str)
parser.add_argument('--hidden_size', help='Size of the hidden layer of LSTM',
                    nargs='?', default=256, type=int)
parser.add_argument('--embed_dim', help='Size of embedding dimension',
                    nargs='?', default=32, type=int)
parser.add_argument('--lr', help='Learning rate',
                    nargs='?', default=0.0005, type=float)
parser.add_argument('--num_iter', help='Number of iterations',
                    nargs='?', default=200000, type=int)
parser.add_argument('--num_layers', help='Number of layers',
                    nargs='?', default=5, type=int)
parser.add_argument('--train_file', help='File to train on',
                    nargs='?', default='Data/mispelled.csv', type=str)
parser.add_argument('--column', help='Column header of data',
                    nargs='?', default='name', type=str)
parser.add_argument('--print', help='Print every',
                    nargs='?', default=50, type=int)
parser.add_argument('--batch', help='Batch size',
                    nargs='?', default=512, type=int)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=1, type=int)

# Parse optional args from command line and save the configurations into a JSON file
args = parser.parse_args()
NAME = args.name
ITER = args.num_iter
NUM_LAYERS = args.num_layers
EMBED_DIM = args.embed_dim
LR = args.lr
HIDDEN_SZ = args.hidden_size
TRAIN_FILE = args.train_file
BATCH_SZ = args.batch
COLUMN = args.column
PRINTS = args.print
CLIP = 1


def train(x: list):
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    loss = 0.

    batch_sz = len(x[0])
    src_max_len = len(max(x[1], key=len))
    trg_max_len = len(max(x[0], key=len)) + 1

    src_x = list(
        map(lambda s: [char for char in s] + [PAD] * ((src_max_len - len(s))), x[1]))
    trg_x = list(map(lambda s: [char for char in s] +
                               [EOS] + [PAD] * ((trg_max_len - len(s)) - 1), x[0]))

    src = indexTensor(src_x, src_max_len, CHARACTERS).to(DEVICE)
    trg = targetsTensor(trg_x, trg_max_len, CHARACTERS).to(DEVICE)
    lng = lengthTensor(x[1]).to(DEVICE)

    hidden = encoder.forward(src, lng)

    lstm_input = targetsTensor([SOS] * batch_sz, 1, CHARACTERS).to(DEVICE)
    for i in range(trg.shape[0]):
        lstm_probs, hidden = decoder.forward(lstm_input, hidden)
        loss += criterion(lstm_probs[0], trg[i])
        lstm_input = trg[i].unsqueeze(0)

    loss.backward()
    encoder_opt.step()
    decoder_opt.step()

    return loss.item()


def iter_train(dl: DataLoader, path: str = "Checkpoints/"):
    all_losses = []
    total_loss = 0
    count = 0

    for iter in range(1, ITER + 1):
        for x in dl:
            count = count + 1
            loss = train(x)
            total_loss += loss

            if count % PRINTS == 0:
                all_losses.append(total_loss / PRINTS)
                total_loss = 0
                plot_losses(
                    all_losses, x_label=f"Iteration of Batch Size: {BATCH_SZ}", y_label="NLLosss", filename=NAME)
                torch.save({'weights': encoder.state_dict()},
                           f"{path}{NAME}_encoder.path.tar")
                torch.save({'weights': decoder.state_dict()},
                           f"{path}{NAME}_decoder.path.tar")


def test(x: list):
    encoder.eval()
    decoder.eval()

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

    return [name for name, score, hidden in top_k_beam_search(hidden)]


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


to_save = {
    'session_name': NAME,
    'hidden_size': HIDDEN_SZ,
    'num_layers': NUM_LAYERS,
    'embed_dim': EMBED_DIM,
    'input': CHARACTERS,
    'output': CHARACTERS,
    'input_sz': NUM_CHAR,
    'output_sz': NUM_CHAR,
    'EOS': EOS,
    'SOS': SOS,
    'PAD': PAD,
}

save_json(f'Config/{NAME}.json', to_save)

decoder = Decoder(NUM_CHAR, HIDDEN_SZ, PAD_IDX,
                  NUM_LAYERS, EMBED_DIM).to(DEVICE)
encoder = Encoder(NUM_CHAR, HIDDEN_SZ, PAD_IDX,
                  NUM_LAYERS, EMBED_DIM).to(DEVICE)

if args.continue_training == 1:
    encoder.load_state_dict(torch.load(
        f'Checkpoints/{NAME}_encoder.path.tar')['weights'])
    decoder.load_state_dict(torch.load(
        f'Checkpoints/{NAME}_decoder.path.tar')['weights'])

criterion = nn.NLLLoss(ignore_index=PAD_IDX)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr=LR)
encoder_opt = torch.optim.Adam(encoder.parameters(), lr=LR)

df = pd.read_csv(TRAIN_FILE)
ds = WordDataset(df)
dl = DataLoader(ds, batch_size=BATCH_SZ, shuffle=True)

iter_train(dl)
