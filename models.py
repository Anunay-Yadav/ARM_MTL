# from modules import *
from re import L
from sklearn import model_selection
import torch.nn as nn
import torch
import numpy as np
from Data_loader.MarketData import *
import modules as model
import higher

class ARM_CNP(nn.Module):
    def __init__(self, model_context, model_prediction, num_labels, labels_type = "market", data_type = "flat", learning_rate = 1e-5) -> None:
        super().__init__()
        self.model_context = model_context
        self.model_prediction = model_prediction
        self.data_type = data_type
        self.loss = []
        self.data = labels_type
        self.loss.append(nn.NLLLoss(weight = torch.tensor([0.25, 0.75])))
        self.learning_rate = learning_rate
        params = list(self.model_context.parameters()) + list(self.model_prediction.parameters())
        self.classes = num_labels
        self.optimizer = torch.optim.Adam(params, lr = self.learning_rate)

    def predictStream(self, X_train, X_test):
        context = torch.sum(self.model_context(X_test), 0)
        context = torch.sum(self.model_context(X_train), 0) + context
        context /= (X_test.size(0) + X_train.size(0))

        X = self.addContext(X_test, context)

        return self.model_prediction(X)

    def predict(self, X_test, y):
        context = self.model_context(X_test)
        labels = y.view(y.size(0), 1).expand(-1, context.size(1))
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, context)
        r = res / labels_count.float().unsqueeze(1)
        X = self.addContext(X_test, res)
        return self.model_prediction(X)

    def addContext(self, X_test, context):
        added_X_test = []
        context_padded = torch.unsqueeze(context, 0)
        sample_size = X_test.size(0)
        context_padded_sample = torch.cat((context_padded,)*sample_size, 0)
        final_X = torch.cat((X_test, context_padded_sample), len(X_test.size()) - 1)

        
        return final_X
    def learnStream(self, X_train, Y_train, X_test, Y_test):
        self.train()

        logits = self.predictStream(X_train, X_test)
        loss = 0 
        for loss_fn in self.loss:
          loss += loss_fn(logits, Y_test)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, X_test, Y_test):
        self.train()
        logits = self.predict(X_test)
        loss = 0 
        for loss_fn in self.loss:
          loss += loss_fn(logits, Y_test)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def accuracy_market(self, logits, Y):
        accuracy = []
        for i in range(10):
            if(i < 6):
                accuracy.append(((logits[i][ :, 0] - Y[:, i])*(logits[i][ :, 0] - Y[:, i])).sum().data)
            else:
                preds = np.argmax(logits[i].detach().cpu().numpy(), axis=1)
                acc = np.mean(preds == Y[:, i].detach().cpu().numpy().reshape(-1))
                accuracy.append(acc)
        return accuracy
    def accuracy(self, logits, Y):
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        accuracy = np.mean(preds == Y.detach().cpu().numpy().reshape(-1))
        return accuracy
        
    def accuracy_arc(Net,logits,y_test):
        #Net.eval()
        m = 0
        y_pred = logits
        prob, predicted = torch.max(y_pred, 1)
        correct = 0
        for i in range(0, y_test.size(0), 4):
            cnt = 0
            for j in range(4):
                cnt += (y_test[i + j] == predicted[i + j])
            if(cnt == 4):
                correct += 1
            m += 1
        accuracy = correct/m
        #Net.train()
        return accuracy

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
    if self.data == 'market':
      logits_ret, loss_ret = [], []
      for i in range(10): 
        with higher.innerloop_ctx(self.model_prediction[i], self.optimizer, copy_initial_weights=False) as (fmodel, diffopt):
          logits = fmodel.predict(X_test)
          floss = self.model_loss[i](logits)
          diffopt.step(floss)

          logits = fmodel.predict(X_test)
          if train:
            if i >5:
              loss = self.loss[i](logits, Y_test[:,i].long())
            else:
              loss = self.loss[i](logits[:,0], Y_test[:,i].long())
            loss.backward()
          logits_ret.append(logits)
          loss_ret.append(loss)
      return logits_ret, loss_ret
    else:
      with higher.innerloop_ctx(self.model_prediction, self.optimizer, copy_initial_weights=False) as (fmodel, diffopt):
        logits = fmodel.predict(X_test)
        floss = self.model_loss(logits)
        diffopt.step(floss)

        logits = fmodel.predict(X_test)
        if train:
          loss = self.loss[0](logits, Y_test)
          loss.backward()
      return logits, loss
  
  def learn(self, X_test, Y_test):
    self.train()
    self.optimizer.zero_grad()
    logits, loss = self.predict(X_test, Y_test, True)
    self.optimizer.step()

  def accuracy_market(self, logits, Y):
    accuracy = []
    for i in range(10):
        if(i < 6):
            accuracy.append(((logits[i][ :, 0] - Y[:, i])*(logits[i][ :, 0] - Y[:, i])).sum().data)
        else:
            preds = np.argmax(logits[i].detach().cpu().numpy(), axis=1)
            acc = np.mean(preds == Y[:, i].detach().cpu().numpy().reshape(-1))
            accuracy.append(acc)
    return accuracy
    
  def accuracy(self, logits, Y):
    preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
    accuracy = np.mean(preds == Y.detach().cpu().numpy().reshape(-1))
    return accuracy
      
  def accuracy_arc(Net,logits,y_test):
    #Net.eval()
    m = 0
    y_pred = logits
    prob, predicted = torch.max(y_pred, 1)
    correct = 0
    for i in range(0, y_test.size(0), 4):
        cnt = 0
        for j in range(4):
            cnt += (y_test[i + j] == predicted[i + j])
        if(cnt == 4):
            correct += 1
        m += 1
    accuracy = correct/m
    #Net.train()
    return accuracy

if __name__ == "__main__":
  