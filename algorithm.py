from sklearn import model_selection
import torch.nn as nn
import torch
import numpy as np
from Data_loader.FewShotPaddedARC import *
import modules as model
class ARM_CML(nn.Module):
    def __init__(self, model_context, model_prediction, labels_type = "market", learning_rate = 1e-5) -> None:
        super().__init__()
        self.model_context = model_context
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

        params = list(self.model_context.parameters()) + list(self.model_prediction.parameters())
        self.optimizer = torch.optim.Adam(params, lr = self.learning_rate)

    def predictStream(self, X_train, X_test):
        context = self.model_context(X_train[0])
        for i in range(1, len(X_train)):
            context += self.model_context(X_train[i])
        for i in range(X_test):
            context += self.model_context(X_test[i])
        context /= (len(X_test) + len(X_train))

        X = self.addContext(X_test)
        return self.model_prediction(X)

    def predict(self, X_test):
        # X_test = self.convert(X_test)
        # print(len(X_test[0].shape))
        context = torch.mean(self.model_context(X_test), 0)

        X = self.addContext(X_test, context)
        return self.model_prediction(X)

    def addContext(self, X_test, context):
        added_X_test = []
        context_padded = torch.unsqueeze(context, 0)
        sample_size = X_test.size(0)
        context_padded_sample = torch.cat((context_padded,)*sample_size, 0)
        final_X = torch.cat((X_test, context_padded_sample), 1)

        
        return final_X
    def learnStream(self, X_train, Y_train, X_test, Y_test):
        self.train()

        logits = self.predictStream(X_train, X_test)
        loss = 0 
        for loss_fn in self.loss:
            loss += loss_fn(logits, Y_test)
        
        self.optimzer.zero_grad()
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
    
    # def accuracy(self, logits, Y):
    #     preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
    #     accuracy = np.mean(preds == Y.detach().cpu().numpy().reshape(-1))
    #     return accuracy
    def accuracy(Net,logits,y_test):
        #Net.eval()
        m = 0
        y_pred = logits
        prob, predicted = torch.max(y_pred, 1)
        correct = 0;
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
    loader = DataLoader("FewShotPaddedARC")
    model_context = model.MLP(dims = [200, 124, 200])
    model_pred = model.MLP(dims=[400, 64, 32, 2])
    arm_model = ARM_CML(model_context=model_context, model_prediction=model_pred, labels_type="arc", learning_rate=1e-6)
    for j in range(1000):
        acc = 0
        for i in range(1000):
            train, test = loader.get_task("meta_train")
            arm_model.learn(train[0], train[1])
            train, test = loader.get_task("meta_test")
            acc += arm_model.accuracy(arm_model.predict(train[0]), train[1])
        print(acc/100)
