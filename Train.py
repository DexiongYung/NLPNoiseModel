import torch
import os
import torch.nn as nn
import argparse
import pandas as pd
from Constants import *
from Utilities.Plot import *
from Utilities.Convert import *
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
                    nargs='?', default='Data/fb_moe.csv', type=str)
parser.add_argument('--column', help='Column header of data',
                    nargs='?', default='name', type=str)
parser.add_argument('--print', help='Print every',
                    nargs='?', default=50, type=int)
parser.add_argument('--batch', help='Batch size',
                    nargs='?', default=512, type=int)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=0, type=int)

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

    names = [''] * batch_sz

    lstm_input = targetsTensor([SOS] * batch_sz, 1, CHARACTERS).to(DEVICE)
    for i in range(trg.shape[0]):
        lstm_probs, lstm_hidden = decoder.forward(lstm_input, hidden)
        best_index = torch.argmax(lstm_probs, dim=2)

        loss += criterion(lstm_probs[0], trg[i])

        for idx in range(len(names)):
            names[idx] += CHARACTERS[best_index[0][idx].item()]
        
        lstm_input = trg[i].unsqueeze(0)

    loss.backward()
    encoder_opt.step()
    decoder_opt.step()

    return names, loss.item()

def iter_train(dl: DataLoader, path: str = "Checkpoints/"):
    all_losses = []
    total_loss = 0
    count = 0

    for iter in range(1, ITER + 1):
        for x in dl:
            count = count + 1
            name, loss = train(x)
            total_loss += loss

            if count % PRINTS == 0:
                all_losses.append(total_loss / PRINTS)
                total_loss = 0
                plot_losses(
                    all_losses, x_label=f"Iteration of Batch Size: {BATCH_SZ}", y_label="NLLosss", filename=NAME)
                torch.save({'weights': encoder.state_dict()},
                           os.path.join(f"{path}{NAME}_encoder.path.tar"))
                torch.save({'weights': decoder.state_dict()},
                           os.path.join(f"{path}{NAME}_decoder.path.tar"))

decoder = Decoder(NUM_CHAR, HIDDEN_SZ, PAD_IDX, NUM_LAYERS, EMBED_DIM).to(DEVICE)
encoder = Encoder(NUM_CHAR, HIDDEN_SZ, PAD_IDX, NUM_LAYERS, EMBED_DIM).to(DEVICE)

if args.continue_training == 1:
    encoder.load_state_dict(torch.load(f'Checkpoints/{NAME}_encoder.path.tar')['weights'])
    decoder.load_state_dict(torch.load(f'Checkpoints/{NAME}_decoder.path.tar')['weights'])

criterion = nn.NLLLoss(ignore_index=PAD_IDX)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr=LR)
encoder_opt = torch.optim.Adam(encoder.parameters(), lr=LR)

df = pd.read_csv(TRAIN_FILE)
ds = WordDataset(df)
dl = DataLoader(ds, batch_size=BATCH_SZ, shuffle=True)

iter_train(dl)