import torch
import torch.nn as n
import numpy as np
from dataset import ForecastDataset

def timeseries_dataset_from_array(
        x,
        y, 
        sampling_rate,
        sequence_length,
        delay
    ):
    sequences = []
    targets = []
    window_span = (sequence_length - 1) * sampling_rate

    for i in range(len(x) - window_span - delay):
        indices = range(i, i + window_span + 1, sampling_rate)
        window = x[indices] 
    
        target_index = indices[-1] + delay
        target = y[target_index]
        
        sequences.append(window)
        targets.append(target)
        
    return (
            torch.tensor(np.array(sequences)), 
            torch.tensor(np.array(targets))
        )