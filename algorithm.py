from re import L
from sklearn import model_selection
import torch.nn as nn
import torch
import numpy as np
from Data_loader.MarketData import *
import modules as model
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
        logits = self.predict(X_test)
        loss = 0 
        if(self.data == "market"):
            for ind, loss_fn in enumerate(self.loss):
                if(ind > 5):    
                    loss += loss_fn(logits[ind], Y_test[:, ind].long())
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
    loader = DataLoaderMarket()
    model_context = model.SimpleGRU(input_size=18, hidden_size=32, num_layers=2, output_size=18, task = "regression", lr = 0.00005)
    models = []
    for i in range(6):
        model_pred = model.SimpleGRU(input_size=36, hidden_size=32, num_layers=2, output_size=1, task = "regression", lr = 0.00005)
        models.append(model_pred)
    for i in range(4):
        model_pred = model.SimpleGRU(input_size=36, hidden_size=64, num_layers=2, output_size=4, task = "classification", lr = 0.00005)
        models.append(model_pred)
    arm_model = ARM_CML(model_context=model_context, model_prediction=models, labels_type="market", data_type="time_series", learning_rate=1e-6)
    for j in range(1000):
        acc = [0]*10
        print("---------------EPOCH {} ----------------------".format( j))
        for i in range(50):
            train, test = loader.get_task("meta_train", "series")
            arm_model.learnStream(train[0], train[1], test[0], test[1])
            train, test = loader.get_task("meta_test", "series")
            accur = arm_model.accuracy_market(arm_model.predictStream(train[0], test[0]), test[1])
            for j1 in range(10):
                acc[j1] += accur[j1]
        for i in range(10):
            if(i < 6):
                print("MSE LOSS: ", acc[i].item()/50)
            else:
                print("Accuracy: ", acc[i]/50)
