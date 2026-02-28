import torch
import torch.nn as n
from torch.utils.data import Dataset
import pandas as pd

class ForecastDataset(Dataset):
    def __init__(self, csv_file):
        self.csv = csv_file
        self.data = pd.read_csv(csv_file)
        self.y = self.data['T (degC)'].to_numpy()

        self.data.drop(
            columns=['T (degC)', 'Date Time'], 
            inplace=True
        ) 

        self.x = self.data.to_numpy()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.x[index], dtype=torch.float32), 
            torch.tensor(self.y[index], dtype=torch.float32)
        )
    