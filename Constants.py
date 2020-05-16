import torch
import string

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables
SOS = 'SOS'
PAD = 'PAD'
EOS = 'EOS'
