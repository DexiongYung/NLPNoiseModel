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
from Statistics import *

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
                    nargs='?', default='Data/mispelled.csv', type=str)
parser.add_argument('--column', help='Column header of data',
                    nargs='?', default='name', type=str)
parser.add_argument('--print', help='Print every',
                    nargs='?', default=50, type=int)
parser.add_argument('--batch', help='Batch size',
                    nargs='?', default=256, type=int)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=False, type=bool)

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


CHARACTERS = [c for c in string.printable] + [SOS, PAD, EOS]
NUM_CHARS = len(CHARACTERS)
PAD_IDX = CHARACTERS.index(PAD)


def train(x: list):
    batch_sz = len(x[0])
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
    names = [''] * batch_sz

    for i in range(trg.shape[0]):
        lstm_probs, hidden = decoder.forward(lstm_input, hidden)
        _, indices = lstm_probs.max(2)

        for j in range(batch_sz):
            names[j] += CHARACTERS[indices[0][j].item()]

        lstm_input = trg[i].unsqueeze(0)

    return names


def iter_train(dl: DataLoader, path: str = "Checkpoints/"):
    all_losses = []
    total_loss = 0
    count = 0

    for iter in range(1, ITER + 1):
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()

        for x in dl:
            count = count + 1
            names = train(x)

            if count % PRINTS == 0:
                all_losses.append(total_loss / PRINTS)
                total_loss = 0
                plot_losses(
                    all_losses, x_label=f"Iteration of Batch Size: {BATCH_SZ}", y_label="NLLosss", filename=NAME)
                torch.save({'weights': encoder.state_dict()},
                           f"{path}{NAME}_encoder.path.tar")
                torch.save({'weights': decoder.state_dict()},
                           f"{path}{NAME}_decoder.path.tar")

        encoder_opt.step()
        decoder_opt.step()


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
        f'Checkpoints/ABC/{NAME}_encoder.path.tar')['weights'])
    decoder.load_state_dict(torch.load(
        f'Checkpoints/ABC/{NAME}_decoder.path.tar')['weights'])


criterion = nn.NLLLoss(ignore_index=PAD_IDX)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr=LR)
encoder_opt = torch.optim.Adam(encoder.parameters(), lr=LR)

df = pd.read_csv(TRAIN_FILE)
ds = WordDataset(df)
dl = DataLoader(ds, batch_size=BATCH_SZ, shuffle=True)

iter_train(dl)
