import torch
import os
import torch.nn as nn
import argparse
import pandas as pd
import string
from Constants import DEVICE
from Utilities.Plot import *
from Utilities.Convert import *
from Utilities.Json import *
from torch.utils.data import DataLoader
from Model.Seq2Seq import Encoder, Decoder
from Dataset.NameDataset import NameDataset
from Statistics_Generator import get_summary_stats_tensor


parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session',
                    nargs='?', default='REINFORCE', type=str)
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
                    nargs='?', default='Data/Name/Firsts.csv', type=str)
parser.add_argument('--obs_file', help='File to observation summary statistics on',
                    nargs='?', default=None, type=str)
parser.add_argument('--column', help='Column header of data',
                    nargs='?', default='name', type=str)
parser.add_argument('--print', help='Print every',
                    nargs='?', default=1, type=int)
parser.add_argument('--num_sample', help='Number of samples for gradient evaluation',
                    nargs='?', default=32, type=int)
parser.add_argument('--mini_batch', help='Mini batch size',
                    nargs='?', default=32, type=int)
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
OBS_FILE = args.obs_file
NUM_SAMPLE = args.num_sample
COLUMN = args.column
PRINTS = args.print
MINI_BATCH_SZ = args.mini_batch
CLIP = 1

# SOS, PAD, EOS should be single char cause Levenshtein distance seq goes through string and compares letters
SOS = chr(0x00FD)
PAD = chr(0x00FE)
EOS = chr(0x00FF)
CHARACTERS = [c for c in string.printable] + [SOS, PAD, EOS]
NUM_CHARS = len(CHARACTERS)
PAD_IDX = CHARACTERS.index(PAD)


def sample_model(x: list):
    log_prob_sum = 0
    padded_len = len(max(x, key=len))

    padded_x = list(map(lambda s: [char for char in s] +
                        [PAD] * (padded_len - len(s)), x))
    src = indexTensor(padded_x, padded_len, CHARACTERS).to(DEVICE)
    lng = lengthTensor(x).to(DEVICE)

    hidden = encoder.forward(src, lng)

    lstm_input = targetTensor([SOS] * MINI_BATCH_SZ, 1, CHARACTERS).to(DEVICE)
    names = [''] * MINI_BATCH_SZ

    # padded_len + 1 since as length of word increases the Levenshtein distance size goes down
    for i in range(padded_len + 1):
        lstm_probs, hidden = decoder.forward(lstm_input, hidden)
        categorical = torch.distributions.Categorical(
            probs=lstm_probs.squeeze().exp())
        sample = categorical.sample()
        log_prob_sum += categorical.log_prob(sample).sum()

        for j in range(MINI_BATCH_SZ):
            names[j] += CHARACTERS[sample[j].item()]

        lstm_input = sample.unsqueeze(0)

    return names, log_prob_sum


def iterate_train(dl: DataLoader, path: str = "Checkpoints/"):
    all_losses = []
    num_model_iterations = 0
    scores_list = []

    for epoch_index in range(1, ITER + 1):
        for batch_index, x in enumerate(dl):
            # Zero gradient in models
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()

            # Generate noised outputs
            generated_names, log_prob_sum = sample_model(x)

            # Split generated names
            noised_list = [name.split(EOS)[0] for name in generated_names]

            # Get summary stats of batch
            sample_stats_sum_tensor = get_summary_stats_tensor(noised_list, x)

            # Score batch
            distance = torch.dist(sample_stats_sum_tensor,
                                   obs_stats_sum_tensor, p=2).detach()
            score = distance * log_prob_sum
            scores_list.append(score)

            if batch_index % NUM_SAMPLE == 0:
                # Multiply be -1 because doing gradient descent 
                reinforce_loss = -1 * torch.mean(torch.FloatTensor(scores_list))
                reinforce_loss.backward()

                encoder_opt.step()
                decoder_opt.step()

                # Save weights and print loss
                print(
                    f"Loss: {reinforce_loss.item() / NUM_SAMPLE}")
                all_losses.append(reinforce_loss.item() / NUM_SAMPLE)
                plot_losses(
                    all_losses,
                    x_label=f"Iteration of # Samples: {NUM_SAMPLE}, Mini Batch Size: {MINI_BATCH_SZ}",
                    y_label="ABC",
                    filename=NAME
                )
                torch.save({'weights': encoder.state_dict()},
                           f"{path}{NAME}_encoder.path.tar")
                torch.save({'weights': decoder.state_dict()},
                           f"{path}{NAME}_decoder.path.tar")

                # Zero out metrics
                scores_list = []


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
else:
    # M Teng suggests using Xavier for weight init
    for name, param in decoder.lstm.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)

    for name, param in encoder.lstm.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)

decoder_opt = torch.optim.Adam(decoder.parameters(), lr=LR)
encoder_opt = torch.optim.Adam(encoder.parameters(), lr=LR)

if OBS_FILE is not None:
    obs_df = pd.read_csv(OBS_FILE)
    obs_stats_sum_tensor = get_summary_stats_tensor(
        list(obs_df['Noised']), list(obs_df['Correct']))
else:
    obs_stats_sum_tensor = torch.FloatTensor([1.8526e-01, 2.6551e-01, 5.4923e-01, 5.4276e-01, 2.2182e-04, 1.5038e-02,
                                              5.1833e-01, 2.9150e-02, 4.8918e-01, 2.2695e-01, 2.9138e-01]).to(DEVICE)

df = pd.read_csv(TRAIN_FILE)
ds = NameDataset(df)

dl = DataLoader(ds, batch_size=MINI_BATCH_SZ, shuffle=True)

iterate_train(dl)
