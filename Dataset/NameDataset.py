from pandas import DataFrame
from torch.utils.data import Dataset

class NameDataset(Dataset):
    def __init__(self, df: DataFrame):
        self.data_frame = df.dropna().drop_duplicates()['name']

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        return self.data_frame.iloc[index]
