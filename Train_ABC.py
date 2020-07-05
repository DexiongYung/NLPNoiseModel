import torch
import os
import hashlib
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
from Statistics_Generator import get_summary_stats_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session',
                    nargs='?', default='ABC', type=str)
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
                    nargs='?', default='Data/mispelled_pure_noised.csv', type=str)
parser.add_argument('--column', help='Column header of data',
                    nargs='?', default='name', type=str)
parser.add_argument('--sample', help='Backprop every n samples',
                    nargs='?', default=50, type=int)
parser.add_argument('--batch', help='Batch size',
                    nargs='?', default=50, type=int)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=True, type=bool)

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
SAMPLE_NUM = args.sample
CLIP = 1

# SOS, PAD, EOS should be single char cause Levenshtein distance seq goes through string and compares letters
SOS = chr(0x00FD)
PAD = chr(0x00FE)
EOS = chr(0x00FF)
CHARACTERS = [c for c in string.printable] + [SOS, PAD, EOS]
NUM_CHARS = len(CHARACTERS)
PAD_IDX = CHARACTERS.index(PAD)


def train(x: list):
    loss = 0.

    batch_sz = len(x[0])
    names = [''] * batch_sz
    src_max_len = len(max(x[1], key=len))
    trg_max_len = len(max(x[0], key=len)) + 1

    src_x = list(
        map(lambda s: [char for char in s] + [PAD] * ((src_max_len - len(s))), x[1]))
    trg_x = list(map(lambda s: [char for char in s] +
                               [EOS] + [PAD] * ((trg_max_len - len(s)) - 1), x[0]))

    src = indexTensor(src_x, src_max_len, CHARACTERS).to(DEVICE)
    trg = targetTensor(trg_x, trg_max_len, CHARACTERS).to(DEVICE)
    lng = lengthTensor(x[1]).to(DEVICE)

    hidden = encoder.forward(src, lng)

    lstm_input = targetTensor([SOS] * batch_sz, 1, CHARACTERS).to(DEVICE)
    for i in range(trg.shape[0]):
        lstm_probs, hidden = decoder.forward(lstm_input, hidden)
        categorical = torch.distributions.Categorical(
            probs=lstm_probs.squeeze().exp())
        sample = categorical.sample()

        for j in range(batch_sz):
            names[j] += CHARACTERS[sample[j].item()]

        loss += criterion(lstm_probs[0], trg[i])
        lstm_input = trg[i].unsqueeze(0)

    return loss, names


def iter_train(dl: DataLoader, path: str = "Checkpoints/"):
    all_losses = []
    loss_list = []
    count = 0

    for iter in range(1, ITER + 1):
        for x in dl:
            count = count + 1
            loss, names = train(x)
            samples_stats = get_summary_stats_tensor(names, x[0])
            distance = torch.dist(samples_stats, obs_stats, p=2).detach()
            loss_list.append(distance * loss)

            if count % SAMPLE_NUM == 0:
                encoder_opt.zero_grad()
                decoder_opt.zero_grad()

                loss = (1/SAMPLE_NUM) * \
                    torch.stack(loss_list).mean()
                loss_list = []
                loss.backward()

                encoder_opt.step()
                decoder_opt.step()

                all_losses.append(loss / (count / SAMPLE_NUM))
                plot_losses(
                    all_losses, x_label=f"Iteration of Batch Size: {BATCH_SZ}", y_label="NLLosss * distance", filename=NAME)
                torch.save({'weights': encoder.state_dict()},
                           f"{path}{NAME}_encoder.path.tar")
                torch.save({'weights': decoder.state_dict()},
                           f"{path}{NAME}_decoder.path.tar")


to_save = {
    'session_name': NAME,
    'hidden_size': HIDDEN_SZ,
    'num_layers': NUM_LAYERS,
    'embed_dim': EMBED_DIM,
    'input': CHARACTERS,
    'output': CHARACTERS,
    'input_sz': NUM_CHARS,
    'output_sz': NUM_CHARS,
    'EOS': EOS,
    'SOS': SOS,
    'PAD': PAD,
}

save_json(f'Config/{NAME}.json', to_save)

decoder = Decoder(NUM_CHARS, HIDDEN_SZ, PAD_IDX,
                  NUM_LAYERS, EMBED_DIM).to(DEVICE)
encoder = Encoder(NUM_CHARS, HIDDEN_SZ, PAD_IDX,
                  NUM_LAYERS, EMBED_DIM).to(DEVICE)

if args.continue_training:
    encoder.load_state_dict(torch.load(
        f'Checkpoints/{NAME}_encoder.path.tar')['weights'])
    decoder.load_state_dict(torch.load(
        f'Checkpoints/{NAME}_decoder.path.tar')['weights'])


criterion = nn.NLLLoss(ignore_index=PAD_IDX)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr=LR)
encoder_opt = torch.optim.Adam(encoder.parameters(), lr=LR)

df = pd.read_csv(TRAIN_FILE)
obs_stats = get_summary_stats_tensor(df.Noised.tolist(), df.Correct.tolist())
ds = WordDataset(df)
dl = DataLoader(ds, batch_size=BATCH_SZ, shuffle=True)

iter_train(dl)
