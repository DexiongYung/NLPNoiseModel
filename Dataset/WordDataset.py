from pandas import DataFrame
from torch.utils.data import Dataset
import numpy


class WordDataset(Dataset):
    def __init__(self, df: DataFrame):
        df['x'] = df['x'].str.encode('ascii', 'ignore').str.decode('ascii')
        df['y'] = df['y'].str.encode('ascii', 'ignore').str.decode('ascii')
        df['x'].replace('', numpy.nan, inplace=True)
        self.data_frame = df.dropna()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        return (self.data_frame.iloc[index][0], self.data_frame.iloc[index][1])