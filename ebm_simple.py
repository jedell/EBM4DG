#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:54:53 2022

@author: levpaciorkowski
"""

# Implementing a simple energy-based model to explore its domain generalizing capabilities when learning a task

from torch import nn


# This simple model can handle vector inputs only
# The outputs of the network are the model's given energy values for the possible values of y, given the input x
class EBM_Simple(nn.Module):
    
    def __init__(self, input_size, n_hidden1, n_hidden2, output_size):
        super().__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, output_size),
            )
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        # x is of size BxF; B = batch size; F = number of input features
        return self.network(x)





