from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from dataset import LJDataset, vocab
from model import QuartzNet
from config import model_config, params
from decoder import CerWer
from data_transforms import transforms, collate_fn
import wandb
import youtokentome as yttm

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
model = QuartzNet(quartznet_conf=model_config, num_classes=len(vocab), feat_in=params['num_features'])
model.to(device)
criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])
cerwer = CerWer()

wandb.init(project=params["wandb_name"], config=params)
wandb.watch(model, log="all", log_freq=500)

for epoch in range(1, params["num_epochs"] + 1):
    train_loss, val_loss, train_cer, train_wer, val_wer, val_cer = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
        train_loss += loss.item()
        _, max_probs = torch.max(outputs, 2)
        train_epoch_cer, train_epoch_wer, train_decoded_words, train_target_words = cerwer(max_probs.T.cpu().numpy(),
                                                                                             targets.cpu().numpy(),
                                              inputs_length, targets_length)
        train_wer += train_epoch_wer
        train_cer += train_epoch_cer

    model.eval()
    with torch.no_grad():
        for inputs, inputs_length, targets, targets_length in test_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            outputs = outputs.permute(2, 0, 1)
            loss = criterion(outputs.log_softmax(dim=2), targets, inputs_length, targets_length).cpu()
            val_loss += loss.item()
            _, max_probs = torch.max(outputs, 2)
            val_epoch_cer, val_epoch_wer, val_decoded_words, val_target_words = cerwer(max_probs.T.cpu().numpy(),
                                                                                         targets.cpu().numpy(),
                                                      inputs_length, targets_length)
            val_wer += val_epoch_wer
            val_cer += val_epoch_cer
    wandb.log({"train_loss": train_loss / len(train_dataset),
               "train_wer": train_wer / len(train_dataset),
               "train_cer": train_cer / len(train_dataset),
               "val_loss": val_loss / len(test_dataset),
               "val_wer": val_wer / len(test_dataset),
               "val_cer": val_cer / len(test_dataset),
               "train_samples": wandb.Table(columns=["Target text", "Predicted text"],
                                            data=[train_target_words, train_decoded_words]),
               "val_samples": wandb.Table(columns=["Target text", "Predicted text"],
                                          data=[val_target_words, val_decoded_words]),
               })
    # print(f"""Epoch: {epoch} \ttrain_loss: {train_loss / len(train_dataset):.4} \tval_loss: {val_loss / len(test_dataset):.4}
    #    \ttrain_wer:, {train_wer / len(train_dataset):.4} \tval_wer: {val_wer / len(test_dataset):.4}
    #    \ttrain_cer: {train_cer / len(train_dataset):.4} \tval_cer: {val_cer / len(test_dataset):.4}
    #     \tval_target: {val_target_words} \tval_predicted: {val_decoded_words}\n""")
    if (epoch % 10 == 0) and (epoch >= 40):
        torch.save(model.state_dict(), f"model{epoch}")
