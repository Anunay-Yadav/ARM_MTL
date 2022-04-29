from re import L
from sklearn import model_selection
import torch.nn as nn
import torch
import numpy as np
from Data_loader.MarketData import *
from Data_loader.OptionData import *
from Data_loader.FewShotPaddedARC import *
import modules as model

from re import L
from sklearn import model_selection
import torch.nn as nn
import torch
import numpy as np
from Data_loader.MarketData import *
import modules as model
import higher


class ARM_LL(nn.Module):
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
 
    if(self.data == "market"):
        params = list([])
        for i in range(10):
            params = params + list(self.model_prediction[i].parameters()) + list(self.model_loss[i].parameters())
    else:
        params = list(self.model_prediction.parameters()) + list(self.model_loss.parameters())
    self.optimizer = torch.optim.Adam(params, lr = self.learning_rate)
    if(self.data == "market"):
        self.meta_optim = []
        for i in range(10):
            self.meta_optim.append(torch.optim.Adam(self.model_prediction[i].parameters(), lr=self.learning_rate))
    else:
        self.meta_optim = torch.optim.Adam(self.model_prediction.parameters(), lr=self.learning_rate)
  def predict(self, X_test, Y_test = None, train=False):
    self.train()
    if self.data == 'market':
      logits_ret, loss_ret = [], []
      for i in range(10): 
        with higher.innerloop_ctx(self.model_prediction[i], self.meta_optim[i], copy_initial_weights=False) as (fmodel, diffopt):
          logits = fmodel(X_test)
          floss = self.model_loss[i](logits)
          diffopt.step(floss)

          logits = fmodel(X_test)
          loss = 0
          if train:
            if i >5:
              loss = self.loss[i](logits, Y_test[:,i].long())
            else:
              loss = self.loss[i](logits[:,0], Y_test[:,i].float())
            loss.backward()
          logits_ret.append(logits)
          loss_ret.append(loss)
      return logits_ret, loss_ret
    else:
      with higher.innerloop_ctx(self.model_prediction, self.meta_optim, copy_initial_weights=False) as (fmodel, diffopt):
        for i in range(2):
            logits = fmodel(X_test)
            floss = self.model_loss(logits)
            diffopt.step(floss)

        logits = fmodel(X_test)
        loss = 0
        if train:
          loss = self.loss[0](logits, Y_test)
          loss.backward()
      return [logits, loss]
  
  def learn(self, X_test, Y_test):
    self.train()
    self.optimizer.zero_grad()
    logits, loss = self.predict(X_test, Y_test, True)
    self.optimizer.step()
    return loss

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
class ARM_LL_sup(nn.Module):
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
 
    if(self.data == "market"):
        params = list([])
        for i in range(10):
            params = params + list(self.model_prediction[i].parameters()) + list(self.model_loss[i].parameters())
    else:
        params = list(self.model_prediction.parameters()) + list(self.model_loss.parameters())
    self.optimizer = torch.optim.Adam(params, lr = self.learning_rate)
    if(self.data == "market"):
        self.meta_optim = []
        for i in range(10):
            self.meta_optim.append(torch.optim.Adam(self.model_prediction[i].parameters(), lr=self.learning_rate))
    else:
        self.meta_optim = torch.optim.Adam(self.model_prediction.parameters(), lr=self.learning_rate)
  def predict(self, X_train, Y_train, X_test, Y_test = None, train=False):
    self.train()
    if self.data == 'market':
      logits_ret, loss_ret = [], []
      for i in range(10): 
        with higher.innerloop_ctx(self.model_prediction[i], self.meta_optim[i], copy_initial_weights=False) as (fmodel, diffopt):
          logits = fmodel(torch.cat((X_train, torch.unsqueeze(Y_train, dim = 1)), dim = 1))
          floss = self.model_loss[i](logits)
          diffopt.step(floss)

          logits = fmodel(X_test)
          loss = 0
          if train:
            if i >5:
              loss = self.loss[i](logits, Y_test[:,i].long())
            else:
              loss = self.loss[i](logits[:,0], Y_test[:,i].float())
            loss.backward()
          logits_ret.append(logits)
          if train:
            loss_ret.append(loss.item())
      return logits_ret, loss_ret
    else:
    #   print(Y_train)
      with higher.innerloop_ctx(self.model_prediction, self.meta_optim, copy_initial_weights=False) as (fmodel, diffopt):
        for i in range(2):
            logits = fmodel(X_train)
            floss = self.model_loss(torch.cat((logits, torch.unsqueeze(Y_train, dim = 1).float()), dim = 1))
            diffopt.step(floss)

        logits = fmodel(X_test)
        loss = 0
        if train:
          loss = self.loss[0](logits, Y_test)
          loss.backward()
      return [logits, loss]
  
  def learn(self, X_train, Y_train,  X_test, Y_test):
    self.train()
    self.optimizer.zero_grad()
    logits, loss = self.predict(X_train,Y_train, X_test, Y_test, True)
    self.optimizer.step()
    return loss

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
class ARM_CML(nn.Module):
    def __init__(self, model_context, model_prediction, labels_type = "market", data_type = "flat", learning_rate = 1e-5) -> None:
        super().__init__()
        self.model_context = model_context
        self.model_prediction = model_prediction
        self.data_type = data_type
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
        if(labels_type == "market"):
            params = list(self.model_context.parameters())
            for model in model_prediction:
                params = params + list(model.parameters())
        else:
            params = list(self.model_context.parameters()) + list(self.model_prediction.parameters())
        
        self.optimizer = torch.optim.Adam(params, lr = self.learning_rate)

    def predictStream(self, X_train, X_test):
        context = torch.sum(self.model_context(X_test), 0)
        context = torch.sum(self.model_context(X_train), 0) + context
        context /= (X_test.size(0) + X_train.size(0))

        X = self.addContext(X_test, context)

        if(self.data == "market"):
            ret_val = []
            for model in self.model_prediction:
                ret_val.append(model(X))
            return ret_val
        return self.model_prediction(X)

    def predict(self, X_test):
        context = torch.mean(self.model_context(X_test), 0)

        X = self.addContext(X_test, context)
        if(self.data == "market"):
            ret_val = []
            for model in self.model_prediction:
                ret_val.append(model(X))
            return ret_val
        return self.model_prediction(X)

    def addContext(self, X_test, context):
        added_X_test = []
        context_padded = torch.unsqueeze(context, 0)
        sample_size = X_test.size(0)
        context_padded_sample = torch.cat((context_padded,)*sample_size, 0)
        if(self.data == "market" and self.data_type == "time_series"):
            context_padded_sample = torch.unsqueeze(context_padded_sample, 1)
            context_padded_sample = torch.cat((context_padded_sample,)*X_test.size(1), 1)
        final_X = torch.cat((X_test, context_padded_sample), len(X_test.size()) - 1)

        
        return final_X
    def learnStream(self, X_train, Y_train, X_test, Y_test):
        self.train()

        logits = self.predictStream(X_train, X_test)
        loss = 0 
        ret_loss = []
        if(self.data == "market"):
            for ind, loss_fn in enumerate(self.loss):
                if(ind > 5):    
                    # print(logits[ind], Y_test[:, ind])
                    k = loss_fn(logits[ind], Y_test[:, ind].long())
                    loss += k
                    ret_loss.append(k.item())
                else:
                    loss += loss_fn(logits[ind][ :, 0], Y_test[:, ind].float())
        else:
            for loss_fn in self.loss:
                loss += loss_fn(logits, Y_test)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ret_loss

    def learn(self, X_test, Y_test):
        self.train()
        logits = self.predict(X_test)
        loss = 0 
        ret_loss = []
        if(self.data == "market"):
            for ind, loss_fn in enumerate(self.loss):
                if(ind > 5):    
                    k = loss_fn(logits[ind], Y_test[:, ind].long())
                    loss += k
                    ret_loss.append(k.item())
                else:
                    # print(logits[ind][:, 0])
                    # print(Y_test[:, ind])
                    loss += loss_fn(logits[ind][ :, 0], Y_test[:, ind].float())
        else:
            for loss_fn in self.loss:
                loss += loss_fn(logits, Y_test)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return ret_loss

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
class ARM_BN(nn.Module):
    def __init__(self, model_prediction, labels_type = "market", data_type = "flat", learning_rate = 1e-5) -> None:
        super().__init__()
        self.model_prediction = model_prediction
        self.data_type = data_type
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
        if(labels_type == "market"):
            params = []
            for model in model_prediction:
                params = params + list(model.parameters())
        else:
            params = list(self.model_prediction.parameters())
        
        self.optimizer = torch.optim.Adam(params, lr = self.learning_rate)

    def predictStream(self, X_train, X_test):
        X = X_test
        if(self.data == "market"):
            ret_val = []
            for model in self.model_prediction:
                ret_val.append(model(X))
            return ret_val
        return self.model_prediction(X)

    def predict(self, X_test):
        
        X = X_test
        if(self.data == "market"):
            ret_val = []
            for model in self.model_prediction:
                ret_val.append(model(X))
            return ret_val
        return self.model_prediction(X)

    def learnStream(self, X_train, Y_train, X_test, Y_test):
        self.train()

        logits = self.predictStream(X_train, X_test)
        loss = 0 
        if(self.data == "market"):
            for ind, loss_fn in enumerate(self.loss):
                if(ind > 5):    
                    # print(logits[ind], Y_test[:, ind])
                    loss += loss_fn(logits[ind], Y_test[:, ind].long())
                else:
                    loss += loss_fn(logits[ind][ :, 0], Y_test[:, ind].float())
        else:
            for loss_fn in self.loss:
                loss += loss_fn(logits, Y_test)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, X_test, Y_test):
        self.train()
        ret_loss = []
        logits = self.predict(X_test)
        loss = 0 
        if(self.data == "market"):
            for ind, loss_fn in enumerate(self.loss):
                if(ind > 5):    
                    k = loss_fn(logits[ind], Y_test[:, ind].long())
                    loss += k
                    ret_loss.append(k.item())
                else:
                    # print(logits[ind][:, 0])
                    # print(Y_test[:, ind])
                    loss += loss_fn(logits[ind][ :, 0], Y_test[:, ind].float())
        else:
            for loss_fn in self.loss:
                loss += loss_fn(logits, Y_test)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return ret_loss
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
from tqdm import tqdm
def runMarketSeries(algo_type):
    loader = DataLoaderMarket()
    if(algo_type == "ARM_CML"):
        model_context = model.SimpleGRU(input_size=18, hidden_size=32, num_layers=2, output_size=18, task = "regression", lr = 0.00005)
    elif(algo_type == "ARM_LL"):
        model_context = []
    models = []
    for i in range(6):
        if(algo_type == "ARM_BN"):
            model_pred = model.SimpleGRU(input_size=18, hidden_size=32, num_layers=2, output_size=1, task = "regression", lr = 0.00005)
        elif(algo_type == "ARM_CML"):
            model_pred = model.SimpleGRU(input_size=36, hidden_size=32, num_layers=2, output_size=1, task = "regression", lr = 0.00005)
        elif(algo_type == "ARM_LL"):
            model_context.append(model.MLP(dims = [1, 8, 1], task = "regression", norm_reduce = True, lr = 1e-3))
            model_pred = model.SimpleGRU(input_size=18, hidden_size=16, num_layers=1, output_size=1, task = "regression", lr = 0.00005)
        models.append(model_pred)
    for i in range(4):
        if(algo_type == "ARM_BN"):
            model_pred = model.SimpleGRU(input_size=18, hidden_size=64, num_layers=2, output_size=4, task = "classification", lr = 0.00005)
        elif(algo_type == "ARM_CML"):
            model_pred = model.SimpleGRU(input_size=36, hidden_size=64, num_layers=2, output_size=4, task = "classification", lr = 0.00005)
        elif(algo_type == "ARM_LL"):
            model_context.append(model.MLP(dims = [4, 8, 1], task = "regression", norm_reduce = True, lr = 1e-3))
            model_pred = model.SimpleGRU(input_size=18, hidden_size=16, num_layers=1, output_size=4, task = "regression", lr = 0.00005)
        models.append(model_pred)
    if(algo_type == "ARM_BN"):
        arm_model = ARM_BN( model_prediction=models, labels_type="market", data_type="time_series", learning_rate=1e-6)
    elif(algo_type == "ARM_CML"):
        arm_model = ARM_CML( model_context=model_context ,model_prediction=models, labels_type="market", data_type="time_series", learning_rate=1e-6)
    elif(algo_type == "ARM_LL"):
        arm_model = ARM_LL(model_loss = model_context, model_prediction = models, labels_type = "market", learning_rate = 1e-5)

    samples = 50
    for j in range(1000):
        acc = [0]*10
        print("---------------EPOCH {} ----------------------".format( j))
        for i in tqdm(range(samples)):
            train, test = loader.get_task("meta_train", "series")
            arm_model.learn(train[0], train[1])
            train, test = loader.get_task("meta_test", "series")
            accur = arm_model.accuracy_market(arm_model.predict(train[0]), train[1])
            for j1 in range(10):
                acc[j1] += accur[j1]
        for i in range(10):
            if(i < 6):
                print("MSE LOSS: ", acc[i].item()/samples)
            else:
                print("Accuracy: ", acc[i]/samples)
def runMarketFlat(algo_type):
    loader = DataLoaderMarket()
    if(algo_type == "ARM_CML"):
        model_context = model.MLP(dims = [23, 62, 23])
    elif(algo_type == "ARM_LL"):
        model_context = []
    models = []
    for i in range(6):
        if(algo_type == "ARM_BN"):
            model_pred = model.MLP(dims=[23, 32, 1], task="regression")
        elif(algo_type == "ARM_CML"):
            model_pred = model.MLP(dims=[46, 32, 1], task="regression")
        elif(algo_type == "ARM_LL"):
            model_context.append(model.MLP(dims = [1, 8, 1], task = "regression", norm_reduce = True, lr = 1e-3))
            model_pred = model.MLP(dims=[23, 32, 1], task="regression")
        models.append(model_pred)
    for i in range(4):
        if(algo_type == "ARM_BN"):
            model_pred = model.MLP(dims=[23, 32, 4], task="classification")
        elif(algo_type == "ARM_CML"):
            model_pred = model.MLP(dims=[46, 32, 4], task="classification")
        elif(algo_type == "ARM_LL"):
            model_context.append(model.MLP(dims = [4, 8, 1], task = "regression", norm_reduce = True, lr = 1e-3))
            model_pred = model.MLP(dims=[23, 32, 4], task="classification")
        models.append(model_pred)
    if(algo_type == "ARM_BN"):
        arm_model = ARM_BN( model_prediction=models, labels_type="market", data_type="flat", learning_rate=1e-6)
    elif(algo_type == "ARM_CML"):
        arm_model = ARM_CML(model_context=model_context, model_prediction=models, labels_type="market", data_type="flat", learning_rate=1e-6)
    elif(algo_type == "ARM_LL"):
        arm_model = ARM_LL(model_loss = model_context, model_prediction = models, labels_type = "market", learning_rate = 1e-5)

    samples = 100
    loss_1 = []
    loss_2 = []
    for j in range(100):
        acc = [0]*10
        loss_in = []
        print("---------------EPOCH {} ----------------------".format( j))
        for i in range(samples):
            train, test = loader.get_task("meta_train", "flat")
            # print(test[1][:,5])
            loss_list = arm_model.learn(train[0], train[1])
            loss_in.append([sum(loss_list[:6])/6, sum(loss_list[6:])/4])
            if(algo_type == "ARM_CNP"):
                loss_in.append(arm_model.learn(train[0], train[1], test[0], test[1]))
            else:
                loss_in.append(arm_model.learn(train[0], train[1]))
            train, test = loader.get_task("meta_test")
            if(algo_type == "ARM_CNP"):
                accur = arm_model.accuracy_market(arm_model.predict(train[0], train[1], test[0]), test[1])
            elif(algo_type == "ARM_LL"):
                # print(arm_model.predict(train[0], train[1], test[0])[0])
                accur = arm_model.accuracy_market(arm_model.predict(test[0])[0], test[1])
            else:
                accur = arm_model.accuracy_market(arm_model.predict(test[0]), test[1])
            for j1 in range(10):
                acc[j1] += accur[j1]
        # for i in range(10):
        #     if(i < 6):
        #         print("MSE LOSS: ", acc[i].item()/samples)
        #     else:
        #         print("Accuracy: ", acc[i]/samples)
        loss_1.append(sum([i[0] for i in loss_in]) / samples)
        loss_2.append(sum([i[1] for i in loss_in]) / samples)
    return loss_1, loss_2
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

    def predict(self, X_train, y, X_test):
        context = self.model_context(X_train)
        labels = y.view(y.size(0), 1).expand(-1, context.size(1))
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, context)
        r = res / labels_count.float().unsqueeze(1)
        res = res.reshape(1, -1)
        res = res.squeeze()
        if(res.shape[0]%2 != 0):
            res = torch.cat((res, res))
        X = self.addContext(X_test, res)
        
        return self.model_prediction(X)

    def addContext(self, X_test, context):
        added_X_test = []
        # print(context.shape, X_test.shape)
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
        k = loss.item()    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return k

    def learn(self,X_train, Y_train, X_test, Y_test):
        self.train()
        logits = self.predict(X_train, Y_train, X_test)
        loss = 0 
        for loss_fn in self.loss:
          loss += loss_fn(logits, Y_test)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
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
import matplotlib.pyplot as plt
def runOptions(algo_type):
    loader = DataLoaderOptions()
    if(algo_type == "ARM_BN"):
        model_pred = model.MLP(dims=[148, 64, 32, 2])
        arm_model = ARM_BN( model_prediction=model_pred, labels_type="options", data_type="flat", learning_rate=1e-6)
    elif(algo_type == "ARM_CML"):
        model_pred = model.MLP(dims=[296, 64, 32, 2])
        model_context = model.MLP(dims = [148, 124, 148])
        arm_model = ARM_CML(model_context=model_context, model_prediction=model_pred, labels_type="options", data_type="flat", learning_rate=1e-6)
    elif(algo_type == "ARM_LL_sup"):
        model_pred = model.MLP(dims=[148, 64, 32, 2])
        model_context = model.MLP(dims = [3, 8, 1], task = "regression", norm_reduce = True, lr = 1e-3)
        arm_model = ARM_LL_sup(model_loss = model_context, model_prediction = model_pred, labels_type = "options", learning_rate = 1e-5)
    elif(algo_type == "ARM_LL"):
        model_pred = model.MLP(dims=[148, 64, 32, 2])
        model_context = model.MLP(dims = [2, 8, 1], task = "regression", norm_reduce = True, lr = 1e-3)
        arm_model = ARM_LL(model_loss = model_context, model_prediction = model_pred, labels_type = "options", learning_rate = 1e-5)
    elif(algo_type == "ARM_CNP"):
        model_context = model.MLP(dims = [148, 124, 45])
        model_pred = model.MLP(dims=[148 + 90, 64, 32, 2])
        arm_model = ARM_CNP(model_context=model_context, model_prediction=model_pred, num_labels=2,labels_type="options", data_type="flat", learning_rate=1e-6)
    samples = 200
    loss = []
    for j in range(10000):
        acc = 0
        loss_in = []
        for i in range(samples):
            train, test = loader.get_task("meta_train")
            if(algo_type == "ARM_CNP" or algo_type == "ARM_LL_sup"):
                loss_in.append(arm_model.learn(train[0], train[1], test[0], test[1]))
            else:
                loss_in.append(arm_model.learn(train[0], train[1]))
            train, test = loader.get_task("meta_test")
            if(algo_type == "ARM_CNP"):
                acc += arm_model.accuracy(arm_model.predict(train[0], train[1], test[0]), test[1])
            elif(algo_type == "ARM_LL_sup"):
                # print(arm_model.predict(train[0], train[1], test[0])[0])
                acc += arm_model.accuracy(arm_model.predict(train[0], train[1], test[0])[0], test[1])
            elif(algo_type == "ARM_LL"):
                # print(arm_model.predict(train[0], train[1], test[0])[0])
                acc += arm_model.accuracy(arm_model.predict(test[0])[0], test[1])
            else:
                acc += arm_model.accuracy(arm_model.predict(test[0]), test[1])
        if(j%100 == 0):
            print()
        print("\rEpoch:", j ,", Accuracy :", acc/samples, end = " ")
        # loss.append(sum(loss_in) / len(loss_in))
    print("")
    return loss
def runARC(algo_type):
    loader = DataLoader("FewShotPaddedARC")
    if(algo_type == "ARM_BN"):
        model_pred = model.MLP(dims=[200, 64, 32, 2])
        arm_model = ARM_BN( model_prediction=model_pred, labels_type="arc", data_type="flat", learning_rate=1e-6)
    elif(algo_type == "ARM_CML"):
        model_context = model.MLP(dims = [200, 124, 200])
        model_pred = model.MLP(dims=[400, 64, 32, 2])
        arm_model = ARM_CML(model_context=model_context, model_prediction=model_pred, labels_type="arc", data_type="flat", learning_rate=1e-6)
    elif(algo_type == "ARM_LL"):
        model_context = model.MLP(dims = [2, 8, 1], task = "regression", norm_reduce = True, lr = 1e-3)
        model_pred = model.MLP(dims=[200, 64, 32, 2])
        arm_model = ARM_LL(model_loss = model_context, model_prediction = model_pred, labels_type = "arc", learning_rate = 1e-5)
    elif(algo_type == "ARM_CNP"):
        model_context = model.MLP(dims = [200, 124, 45])
        model_pred = model.MLP(dims=[290, 64, 32, 2])
        arm_model = ARM_CNP(model_context=model_context, model_prediction=model_pred, num_labels=2,labels_type="arc", data_type="flat", learning_rate=1e-6)
    samples = 1000
    epoch_accuracy = []
    loss = []
    for j in range(100):
        acc = 0
        for i in range(samples):
            train, test = loader.get_task("meta_train")
            if(algo_type == "ARM_CNP"):
                loss.append(arm_model.learn(train[0], train[1], test[0], test[1]))
            else:
                loss.append(arm_model.learn(train[0], train[1]))
            train, test = loader.get_task("meta_train")
            if(algo_type == "ARM_CNP"):
                acc += arm_model.accuracy_arc(arm_model.predict(train[0], train[1], test[0]), test[1])
            elif(algo_type == "ARM_LL"):
                acc += arm_model.accuracy_arc(arm_model.predict(test[0])[0], test[1])
            else:
                acc += arm_model.accuracy_arc(arm_model.predict(test[0]), test[1])
        if(j%100 == 0):
            epoch_accuracy.append(acc/samples)
            print()
        print("\rEpoch:", j ,", Accuracy :", acc/samples, end = " ")    
    
    print("")
    return loss
if __name__ == "__main__":
    loss = runMarketFlat("ARM_BN")
