# from modules import *
from re import L
from sklearn import model_selection
import torch.nn as nn
import torch
import numpy as np
from Data_loader.MarketData import *
import modules as model
import higher

class ARM_LL(nn.module):
  def __init__(self, model_loss, model_prediction, labels_type = "market", learning_rate = 1e-5) -> None:
    super().__init__()
    self.model_loss = model_loss
    self.model_prediction = model_prediction

    self.loss = []
    self.data = labels_type
    if(labels_type == "market"):
      for i in range(6):
        self.loss.append(nn.MSELoss())
      for i in range(4):
        self.loss.append(nn.NLLLoss())
    else:
      self.loss.append(nn.NLLLoss(weight = torch.tensor([0.25, 0.75])))
    self.learning_rate = learning_rate
    self.optimizer = torch.optim.Adam(self.model_prediction.parameters(), lr = self.learning_rate)

  def predict(self, X_test, Y_test, train=False):
    self.train()
    with higher.innerloop_ctx(self.model_prediction, self.optimizer, copy_initial_weights=False) as (fmodel, diffopt):
      logits = fmodel.predict(X_test)
      floss = self.model_loss(logits)
      diffopt.step(floss)

      logits = model.predict(X_test)
      loss = 0
      if train:
        if self.data == 'market':
          pass
        else:
          for loss_fn in self.loss:
            loss += loss_fn(logits, Y_test)
        loss.backward()
      return logits, loss  
  
  def learn(self, X_test, Y_test):
    self.train()
    self.optimizer.zero_grad()
    logits, loss = self.predict(X_test, Y_test, True)
    self.optimizer.step()

