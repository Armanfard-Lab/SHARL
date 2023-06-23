import torch
import pprint, pickle
import numpy as np
from numpy.random import default_rng
import copy
import torch.nn as nn  
import torch.optim as optim 
from torch.autograd import Variable
from torch.distributions import Bernoulli
import torchvision
import torch.nn.functional as F  
from torch.utils.data import DataLoader 
from tqdm import tqdm

def train(model,train_data ,train_labels,num_epochs,n_batches,device):
  model.train()
  optimizer = optim.Adam(model.parameters(), lr=0.0001)
  criterion = nn.CrossEntropyLoss()
  for epoch in tqdm(range(num_epochs)):
    for i in range(int(np.ceil(train_data.shape[0] / n_batches))):
        local_X, local_y = train_data[i * n_batches:(i + 1) * n_batches, ], train_labels[i * n_batches:(i + 1) * n_batches, ]
        local_X = local_X.to(device)
        local_y = local_y.to(device).long()
        scores = model(local_X)
        loss = criterion(scores, local_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def check_accuracy(model,data,labels,device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        scores = model(data.to(device))
        # print(scores)
        _, predictions = scores.max(1)
        num_correct += (predictions == labels.to(device)).sum()
        # print(predictions)
        num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy  \
              {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()
    return (float(num_correct) / float(num_samples) * 100)