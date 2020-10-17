import torch
import torchaudio
from src.model import QuartzNet
from config import model_config, params
from src.data_transforms import transforms
from src.decoder import CerWer

model = QuartzNet(quartznet_conf=model_config, num_classes=params["vocab_size"], feat_in=params['num_features'])
model.load_state_dict(torch.load(params["model_path"]))

wav_file = params["path_to_file"]
wav, sr = torchaudio.load(wav_file)
wav = wav
input = transforms['test'](wav)
len_input = input.shape[1]
cerwer = CerWer()
output = model(input)
output = output.permute(2, 0, 1)
_, max_probs = torch.max(output, 2)
decoded_words = cerwer.inference(max_probs.T.numpy().squeeze(), len_input)
with open(f"{wav_file}".replace("wav", "txt"), "w") as txt_file:
    for word in decoded_words:
        txt_file.write(" ".join(word) + "\n")
