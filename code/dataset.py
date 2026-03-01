import torch
import torch.nn as n
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ForecastDataset(Dataset):
    def __init__(self, 
                 csv_file, 
                 sampling_rate=6, 
                 sequence_length=3,
                 delay=144
                 ):
        self.sampling_rate = sampling_rate
        self.sequence_length = sequence_length
        self.delay = delay
        self.window_span = (sequence_length - 1) * sampling_rate
        self.csv = csv_file
        self.data = pd.read_csv(csv_file)
        self.y = self.data['T (degC)'].to_numpy()

        self.data.drop(
            columns=['T (degC)', 'Date Time'], 
            inplace=True
        ) 

        self.x = self.data.to_numpy()
        
    def __len__(self):
        return len(self.x) - self.window_span - self.delay
    
    def __getitem__(self, index):
        end_of_window = index + self.window_span
        indices = np.arange(index, end_of_window + 1, self.sampling_rate)
        target_index = end_of_window + self.delay
    
        x = self.x[indices]
        y = self.y[target_index]

        return (torch.from_numpy(x), torch.from_numpy(y))