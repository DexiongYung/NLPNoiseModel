import torch
import string

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables
SOS = 'SOS'
PAD = 'PAD'
EOS = 'EOS'

CHARACTERS = [c for c in string.printable] + [SOS, PAD, EOS]
NUM_CHAR = len(CHARACTERS)
PAD_IDX = CHARACTERS.index(PAD)