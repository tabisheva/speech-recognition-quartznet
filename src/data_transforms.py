import torch
from torchvision import transforms
from torch import distributions
import torchaudio
from config import params
import librosa
import numpy as np

mel_spectrogramer = torchaudio.transforms.MelSpectrogram(
    sample_rate=params["sample_rate"],
    n_fft=512,
    win_length=int(20e-3 * 16000),
    hop_length=int(10e-3 * 16000),
    f_min=0,
    f_max=8000,
    n_mels=64,
)


class AddNormalNoise(object):

    def __init__(self):
        self.var = params["noise_variance"]

    def __call__(self, wav):
        noiser = distributions.Normal(0, self.var)
        if np.random.uniform() < 0.5:
            wav += noiser.sample(wav.size())
        return wav.clamp(-1, 1)


class TimeStretch(object):

    def __init__(self):
        self.min_scale = params["min_time_stretch"]
        self.max_scale = params["max_time_stretch"]

    def __call__(self, wav):
        random_stretch = np.random.uniform(self.min_scale, self.max_scale, 1)[0]
        if np.random.uniform() < 0.5:
            wav_stretched = librosa.effects.time_stretch(wav.numpy(), random_stretch)
        else:
            wav_stretched = wav.numpy()
        return torch.from_numpy(wav_stretched)


class PitchShifting(object):
    def __init__(self):
        self.sample_rate = params["sample_rate"]
        self.min_shift = params["min_shift"]
        self.max_shift = params["max_shift"]

    def __call__(self, wav):
        random_shift = np.random.uniform(self.min_shift, self.max_shift, 1)[0]
        if np.random.uniform() < 0.5:
            wav_shifted = librosa.effects.pitch_shift(wav.numpy(), self.sample_rate, random_shift)
        else:
            wav_shifted = wav.numpy()
        return torch.from_numpy(wav_shifted)


class MelSpectrogram(object):
    def __call__(self, wav):
        mel_spectrogram = mel_spectrogramer(wav)
        return mel_spectrogram


class NormalizePerFeature(object):
    """
    Normalize the spectrogram to mean=0, std=1 per channel
    """
    def __call__(self, spec):
        log_mel = torch.log(torch.clamp(spec, min=1e-18))
        mean = torch.mean(log_mel, dim=1, keepdim=True)
        std = torch.std(log_mel, dim=1, keepdim=True) + 1e-5
        log_mel = (log_mel - mean) / std
        return log_mel


transforms = {
    'train': transforms.Compose([
        torchaudio.transforms.Resample(params['original_sample_rate'], params['sample_rate']),
        AddNormalNoise(),
        PitchShifting(),
        TimeStretch(),
        MelSpectrogram(),
        NormalizePerFeature(),
        torchaudio.transforms.TimeMasking(params["time_masking"], True),
    ]),
    'test': transforms.Compose([
        torchaudio.transforms.Resample(params['original_sample_rate'], params['sample_rate']),
        MelSpectrogram(),
        NormalizePerFeature(),
    ]),
}


def collate_fn(batch):
    """
    Stacking sequences of variable lengths in batches with zero-padding in the end of sequences
    :param batch: list of tuples with (inputs with shape (time, channels), inputs_length, targets, targets_length)
    :return: tensor (batch, channels, max_length of inputs) with zero-padded inputs,
             tensor (batch, ) with inputs_lengths,
             tensor (batch, max_length of targets) with zero-padded targets,
             tensor (batch, ) with targets_lengths
    """
    inputs, inputs_length, targets, targets_length = list(zip(*batch))
    input_aligned = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True).permute(0, 2, 1)
    target_aligned = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return input_aligned, torch.Tensor(inputs_length).long(), \
           target_aligned, torch.Tensor(targets_length).long()
