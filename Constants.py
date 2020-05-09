import torch
import string

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables
SOS = 'SOS'
PAD = 'PAD'
EOS = 'EOS'

CHARACTERS = [c for c in string.printable] + \
    [chr(i) for i in range(1000, 1100)] + [chr(i) for i in range(0x0021, 0x02FF)] + ['’', '‘', '€'] + [SOS, PAD, EOS]
NUM_CHARS = len(CHARACTERS)
PAD_IDX = CHARACTERS.index(PAD)
