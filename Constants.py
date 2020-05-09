import torch
import string

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables
SOS = 'SOS'
PAD = 'PAD'
EOS = 'EOS'

CHARACTERS = [c for c in string.printable] + [SOS, PAD, EOS]
OUT_CHARACTERS = CHARACTERS + [chr(i) for i in range(1000, 1100)]
NUM_IN_CHAR = len(CHARACTERS)
NUM_OUT_CHAR = len
PAD_IDX = CHARACTERS.index(PAD)
