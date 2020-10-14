import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchaudio


vocab = "B abcdefghijklmnopqrstuvwxyz'"  # B: blank
char2idx = {char: idx for idx, char in enumerate(vocab)}

def convert_text(text):
    return [char2idx[char] for char in text if char in char2idx]


class LJDataset(Dataset):
    def __init__(self, df, transform=None):
        self.dir = "LJSpeech-1.1/wavs"
        self.filenames = df.index.values
        self.labels = df[2].values
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        text = np.array(convert_text(self.labels[idx].lower()))
        wav, sr = torchaudio.load(os.path.join(self.dir, f'{filename}.wav'))
        wav = wav.squeeze()
        input = self.transform(wav)
        len_input = input.shape[1]
        return input.T, len_input // 2, torch.Tensor(text), len(text)

    def __len__(self):
        return len(self.filenames)