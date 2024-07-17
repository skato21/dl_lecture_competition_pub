import os
import numpy as np
import torch
from typing import Tuple, Dict, Optional
from termcolor import cprint
from glob import glob
from scipy.signal import resample, butter, filtfilt

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_rate: Optional[int] = None, filter_params: Optional[Dict] = None, scaling: bool = True, baseline_correction: bool = True) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.resample_rate = resample_rate
        self.filter_params = filter_params
        self.scaling = scaling
        self.baseline_correction = baseline_correction
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = np.load(X_path)
        
        if self.resample_rate:
            X = resample(X, self.resample_rate, axis=1)
        if self.filter_params:
            b, a = butter(self.filter_params['order'], self.filter_params['cutoff'], btype=self.filter_params['btype'])
            X = filtfilt(b, a, X, axis=1)
        if self.scaling:
            X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
        if self.baseline_correction:
            baseline = X[:, :self.baseline_correction].mean(axis=1, keepdims=True)
            X = X - baseline

        X = torch.from_numpy(X)
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            return X, y, subject_idx
        else:
            return X, subject_idx
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]