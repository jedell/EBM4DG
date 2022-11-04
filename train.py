#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 02:24:30 2022

@author: levpaciorkowski
"""

# Generic script to train a given model on given data

import argparse
import ebm_simple as archs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

SUPPORTED_MODELS = ['EBM_Simple']


# This dataset is only used for simple smoke testing
class TestingData(torch.utils.data.IterableDataset):
    
    def __init__(self, n):
        self.n = n
    
    def __len__(self):
        return self.n
    
    def __iter__(self):
        for _ in range(self.n):
            x = torch.randn(10)
            target = torch.randint(0,1,(1,))
            yield x, target

def cross_entropy(energies, *args, **kwargs):
    return F.cross_entropy(-1*energies, *args, **kwargs)


# Implement forward pass for one batch
def get_loss_and_correct(model, batch, criterion):
    inputs, targets = batch[0], batch[1].float()
    labels = F.one_hot(targets.argmax(dim=1), num_classes=2).float()
    model_outputs = model.forward(inputs)
    # have hard-coded the number of classes to 2 here
    # also hard-coding accuracy calculation ... this may depend on type of data we use later
    # TODO - might need to change this later
    loss = criterion(model_outputs, labels)
    preds = model_outputs.argmin(dim=1).float()
    num_correct = torch.eq(preds.view(-1), targets)[0].sum()
    return loss, num_correct


# Simple generic training loop
def train(epochs, model, optimizer, criterion, train_dataloader, val_dataloader, train_size, val_size):
    model.train()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    pbar = tqdm(range(epochs))
    for i in pbar:
        total_train_loss, total_val_loss = 0, 0
        total_train_correct, total_val_correct = 0, 0
        for batch in tqdm(train_dataloader, leave=False):
            loss, num_correct = get_loss_and_correct(model, batch, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_correct += num_correct.item()
        with torch.no_grad():
            for batch in val_dataloader:
                loss, num_correct = get_loss_and_correct(model, batch, criterion)
                total_val_loss += loss.item()
                total_val_correct += num_correct.item()
        mean_train_loss = total_train_loss / train_size
        mean_val_loss = total_val_loss / val_size
        train_accuracy = total_train_correct / train_size
        val_accuracy = total_val_correct / val_size
        train_losses.append(mean_train_loss)
        val_losses.append(mean_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        pbar.set_postfix({'train_loss:': mean_train_loss, 'validation_loss': mean_val_loss})
    return train_losses, val_losses, train_accs, val_accs
    


# Parsing command line arguments
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='EBM_Simple', help='EBM_Simple')
    parser.add_argument('--data', type=str, default='', help='path to data directory (data should be in .csv)')
    parser.add_argument('--n_hidden1', type=int, default=100, help='number of features in first hidden layer')
    parser.add_argument('--n_hidden2', type=int, default=50, help='number of features in second hidden layer')
    parser.add_argument('--test_mode', type=int, default=0, help='for testing purposes only')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for optimization algorithm')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size during training')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    return parser.parse_args()

def main(opt):
    if opt.model not in SUPPORTED_MODELS:
        print("Supported models", SUPPORTED_MODELS)
        raise ValueError(opt.model + ' not supported. See list above.')
    if opt.test_mode:
        training_data = TestingData(1000)
        train_dataloader = DataLoader(training_data, batch_size=10)
        val_data = TestingData(100)
        val_dataloader = DataLoader(val_data, batch_size=100)
        model = archs.EBM_Simple(10, 100, 50, 2, test_mode=True)
        optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.01)
        criterion = cross_entropy
        train_losses, val_losses, train_accs, val_accs = train(25, model, optimizer, criterion, train_dataloader, val_dataloader,
                                                               len(training_data), len(val_data))
        print("Training successful")
        print("Train losses:", train_losses)
        print("Validation losses:", val_losses)
        print("Train accuracies:", train_accs)
        print("Validation accuracies:", val_accs)
    else:
        X_train = torch.from_numpy(np.genfromtext(opt.data+"/X_train.csv", delimiter=","))
        y_train = torch.from_numpy(np.genfromtext(opt.data+"/y_train.csv", delimiter=","))
        X_val = torch.from_numpy(np.genfromtext(opt.data+"/X_val.csv", delimiter=","))
        y_val = torch.from_numpy(np.genfromtext(opt.data+"/y_val.csv", delimiter=","))
        input_size = X_train.shape[1]
        output_size = y_train.shape[0]
        if opt.model == 'EBM_Simple': model = archs.EBM_Simple(input_size, opt.n_hidden1, opt.n_hidden2, output_size)



if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

