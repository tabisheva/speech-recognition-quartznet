from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from src.dataset import LJDataset
from src.model import QuartzNet
from config import model_config, params
from src.decoder import CerWer
from src.data_transforms import transforms, collate_fn
import wandb
import numpy as np

df = pd.read_csv("LJSpeech-1.1/metadata.csv", sep='|', quotechar='`', index_col=0, header=None)
train, test = train_test_split(df, test_size=0.2, random_state=10)

train_dataset = LJDataset(train, transform=transforms['train'])
test_dataset = LJDataset(test, transform=transforms['test'])

train_dataloader = DataLoader(train_dataset,
                              batch_size=params["batch_size"],
                              num_workers=params["num_workers"],
                              shuffle=True,
                              collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset,
                             batch_size=params["batch_size"],
                             num_workers=params["num_workers"],
                             collate_fn=collate_fn)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model = QuartzNet(quartznet_conf=model_config, num_classes=params["vocab_size"], feat_in=params['num_features'])
if params["from_pretrained"]:
    model.load_state_dict(torch.load(params["model_path"]))
model.to(device)
criterion = nn.CTCLoss(zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])
num_steps = len(train_dataloader) * params["num_epochs"]
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00001)
cerwer = CerWer()

wandb.init(project=params["wandb_name"], config=params)
wandb.watch(model, log="all", log_freq=1000)

start_epoch = params["start_epoch"] + 1 if params["from_pretrained"] else 1
for epoch in range(start_epoch, params["num_epochs"] + 1):
    train_cer, train_wer, val_wer, val_cer = 0.0, 0.0, 0.0, 0.0
    train_losses = []
    model.train()
    for inputs, inputs_length, targets, targets_length in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        outputs = outputs.permute(2, 0, 1)
        optimizer.zero_grad()
        loss = criterion(outputs.log_softmax(dim=2), targets, inputs_length, targets_length).cpu()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip_grad_norm'])
        optimizer.step()
        lr_scheduler.step()
        train_losses.append(loss.item())
        _, max_probs = torch.max(outputs, 2)
        train_epoch_cer, train_epoch_wer, train_decoded_words, train_target_words = cerwer(max_probs.T.cpu().numpy(),
                                                                                           targets.cpu().numpy(),
                                                                                           inputs_length,
                                                                                           targets_length)
        train_wer += train_epoch_wer
        train_cer += train_epoch_cer

    model.eval()
    with torch.no_grad():
        val_losses = []
        for inputs, inputs_length, targets, targets_length in test_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            outputs = outputs.permute(2, 0, 1)
            loss = criterion(outputs.log_softmax(dim=2), targets, inputs_length, targets_length).cpu()
            val_losses.append(loss.item())
            _, max_probs = torch.max(outputs, 2)
            val_epoch_cer, val_epoch_wer, val_decoded_words, val_target_words = cerwer(max_probs.T.cpu().numpy(),
                                                                                       targets.cpu().numpy(),
                                                                                       inputs_length, targets_length)
            val_wer += val_epoch_wer
            val_cer += val_epoch_cer
    wandb.log({"train_loss": np.mean(train_losses),
               "val_wer": val_wer / len(test_dataset),
               "train_cer": train_cer / len(train_dataset),
               "val_loss": np.mean(val_losses),
               "train_wer": train_wer / len(train_dataset),
               "val_cer": val_cer / len(test_dataset),
               "train_samples": wandb.Table(columns=["Target text", "Predicted text"],
                                            data=[train_target_words, train_decoded_words]),
               "val_samples": wandb.Table(columns=["Target text", "Predicted text"],
                                          data=[val_target_words, val_decoded_words]),
               })

    if (epoch % 10 == 0) and (epoch >= 40):
        torch.save(model.state_dict(), f"model{epoch}.pth")
