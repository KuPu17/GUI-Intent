import os
from typing import List
import torch
from torch.utils.data import Dataset


class ClickstreamDataset(Dataset):
    def __init__(self, data_paths: List[str], seq_len: int = 20):
        self.seq_len = seq_len
        self.examples = []
        for p in data_paths:
            if os.path.exists(p):
                # In a real pipeline, load processed parquet and tokenize
                pass

    def __len__(self):
        return max(0, len(self.examples))

    def __getitem__(self, idx):
        # Return a dummy example shape (seq_len, feature_dim)
        x = torch.zeros(self.seq_len, 32)
        y = torch.zeros(1, dtype=torch.long)
        return x, y
