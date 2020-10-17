import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchaudio
import youtokentome as yttm
from config import params
import pandas as pd
import string

PUNCTUATION = string.punctuation + '—–«»−…‑'

def prepare_bpe():
    bpe_path = params["bpe_model"]
    if not os.path.exists(bpe_path):
        df = pd.read_csv("LJSpeech-1.1/metadata.csv", sep='|', quotechar='`', index_col=0, header=None)
        train_data_path = 'bpe_texts.txt'
        with open(train_data_path, "w") as f:
            for i, row in df.iterrows():
                text = row[2].lower().strip().translate(str.maketrans('', '', PUNCTUATION))
                f.write(f"{text}\n")
        yttm.BPE.train(data=train_data_path, vocab_size=params["vocab_size"], model=bpe_path)
        os.system(f'rm {train_data_path}')
    bpe = yttm.BPE(model=bpe_path)
    return bpe


class LJDataset(Dataset):
    def __init__(self, df, transform=None):
        self.dir = "LJSpeech-1.1/wavs"
        self.filenames = df.index.values
        self.labels = df[2].values
        self.transform = transform
        self.bpe = prepare_bpe()

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        text = self.labels[idx].lower().strip().translate(str.maketrans('', '', PUNCTUATION))
        text = np.array(self.bpe.encode(text, dropout_prob=0.2))
        wav, sr = torchaudio.load(os.path.join(self.dir, f'{filename}.wav'))
        wav = wav.squeeze()
        input = self.transform(wav)
        len_input = input.shape[1]
        return input.T, len_input // 2, torch.Tensor(text), len(text)

    def __len__(self):
        return len(self.filenames)
