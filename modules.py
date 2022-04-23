import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch import optim

class ContextNet2D(nn.Module):
    def __init__(self, n_in, n_out, n_hid, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(n_in, n_hid, kernel_size, padding)
        self.conv2 = nn.Conv2d(n_hid, n_hid, kernel_size, padding)
        self.conv3 = nn.Conv2d(n_hid, n_out, kernel_size, padding)
        self.bn1 = nn.BatchNorm2d(n_hid)
        self.bn2 = nn.BatchNorm2d(n_hid)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x
    
class MLP(nn.Module):
    def __init__(self,dims=[5,3,2],task='classification',lr=1e-3):
        super(MLP,self).__init__()
        # Neural network layers assigned as attributes of a Module subclass
        # have their parameters registered for training automatically.
        ### INSERT YOUR CODE HERE
        self.linear_layers = nn.ModuleList()
        for i in range(1, len(dims)):
          self.linear_layers.append(nn.Linear(dims[i-1], dims[i]))
        self.relu = nn.ReLU()
        self.logsoft = nn.LogSoftmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
    def forward(self,x):
        ### Insert your code here
        for i in range(len(self.linear_layers)):
          x = self.linear_layers[i](x)
          x = self.relu(x)

        return self.logsoft(x)
        
class CNN(nn.Module):
    def __init__(self, n_in, n_out, n_hid, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(n_in, n_hid, kernel_size, padding)
        self.conv2 = nn.Conv2d(n_hid, n_hid, kernel_size, padding)
        self.conv3 = nn.Conv2d(n_hid, n_hid, kernel_size, padding)
        self.bn1 = nn.BatchNorm2d(n_hid)
        self.bn2 = nn.BatchNorm2d(n_hid)
        self.bn3 = nn.BatchNorm2d(n_hid)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(n_hid, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return x

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,lr):
        # This just calls the base class constructor
        super().__init__()
        # Neural network layers assigned as attributes of a Module subclass
        # have their parameters registered for training automatically.
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers = num_layers, nonlinearity='relu', batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.logsoft = nn.LogSoftmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
    def forward(self, x):
        # torch.nn.RNN also returns its hidden state but we don't use it.
        # While the RNN can also take a hidden state as input, the RNN
        # gets passed a hidden state initialized with zeros by default.
        if self.input_size==1: x=x.unsqueeze(-1)
        ### INSERT YOUR CODE HERE


        x, hidden_states = self.rnn(x)

        x = x[:, -1, :].contiguous().view(-1, self.hidden_size)
        x = self.linear(x)
        x = self.logsoft(x)
        return x

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,lr):
        # This just calls the base class constructor
        super().__init__()
        # Neural network layers assigned as attributes of a Module subclass
        # have their parameters registered for training automatically.
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers = num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.logsoft = nn.LogSoftmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)

    def forward(self, x):
        # torch.nn.RNN also returns its hidden state but we don't use it.
        # While the RNN can also take a hidden state as input, the RNN
        # gets passed a hidden state initialized with zeros by default.
        if self.input_size==1: x=x.unsqueeze(-1)
        ### INSERT YOUR CODE HERE

        x, hidden_states = self.gru(x)

        x = x[:, -1, :].contiguous().view(-1, self.hidden_size)
        x = self.linear(x)
        x = self.logsoft(x)
        return x

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,lr):
        # This just calls the base class constructor
        super().__init__()
        # Neural network layers assigned as attributes of a Module subclass
        # have their parameters registered for training automatically.
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers = num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.logsoft = nn.LogSoftmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)

    def forward(self, x):
        # torch.nn.RNN also returns its hidden state but we don't use it.
        # While the RNN can also take a hidden state as input, the RNN
        # gets passed a hidden state initialized with zeros by default.
        if self.input_size==1: x=x.unsqueeze(-1)
        ### INSERT YOUR CODE HERE

        x, hidden_states = self.gru(x)

        x = x[:, -1, :].contiguous().view(-1, self.hidden_size)
        x = self.linear(x)
        x = self.logsoft(x)
        return x

import numpy as np

class Embeddings(nn.Module):
    def create_embeddings(self, emmbeddings_dim, max_seq):
        ret_embed = []
        for i in range(max_seq):
            temp_sin = []
            for j in range(0, emmbeddings_dim, 2):
                temp_sin.append(np.sin(i/((10**4)**(2*(j//2)/emmbeddings_dim))))
            temp_cos = []

            for j in range(1, emmbeddings_dim, 2):
                temp_cos.append(np.cos(i/((10**4)**(2*(j//2)/emmbeddings_dim))))
            temp = []
            for i in range(0, emmbeddings_dim):
                if(j%2 == 0):
                    temp.append(temp_sin[j//2])
                else:
                    temp.append(temp_cos[j//2])
            ret_embed.append(temp)
        np_ret_embed = np.array(ret_embed)
        tensor_ret_embed = torch.FloatTensor(np_ret_embed)
        tensor_ret_embed.requires_grad = False
        return tensor_ret_embed
    def create_position(self, batch_size, max_seq):
        ret_embed = [
                    [j for i in range(max_seq)] for j in range(batch_size)
        ]
        return torch.tensor(np.array(ret_embed), dtype=torch.long)
    def __init__(self, embedings_dim, max_seq, lr):
          super().__init__()
          ### INSERT YOUR CODE HERE
          self.max_seq = max_seq
          self.embedings_dim = embedings_dim
          self.embeddings = nn.Embedding(max_seq, embedings_dim)
          self.embeddings.load_state_dict({'weight': self.create_embeddings(embedings_dim, max_seq)})
          self.layer_norm = nn.LayerNorm(embedings_dim, eps = 1e-9)
          self.optimizer = optim.Adam(self.parameters(),lr=lr)
    def forward(self, x):
          y = x.size()
          if(len(y) == 2):
              x = x.reshape((y[0], y[1], 1))
          position = self.create_position(x.size()[0], self.max_seq)
          x = x + self.embeddings(position)[:x.size(0), :x.size(1), :x.size(2)]
          x = self.layer_norm(x)
          return x

  
class TransformerEncoder(nn.Module):
    def __init__(self, embedings_dim, max_seq, num_encoder_layer, num_head, hidden_size, output_size, lr):
          super().__init__()
          ### INSERT YOUR CODE HERE
          self.max_seq = max_seq
          self.embedings_dim = embedings_dim
          self.emmbedings = Embeddings(embedings_dim, max_seq, lr)
          self.layers = nn.ModuleList()
          for i in range(num_encoder_layer):
              #encoder layer
              layers_encoder = nn.ModuleList()
              layers_encoder.append(nn.MultiheadAttention(embedings_dim, num_head))
              layers_encoder.append(nn.LayerNorm(embedings_dim))
              layers_encoder.append(nn.Linear(embedings_dim, hidden_size))
              layers_encoder.append(nn.ReLU())
              layers_encoder.append(nn.Linear(hidden_size, embedings_dim))
              layers_encoder.append(nn.ReLU())
              layers_encoder.append(nn.LayerNorm(embedings_dim))
              self.layers.append(layers_encoder)
          self.final_layer = nn.Linear(embedings_dim, output_size)
          self.logsoft = nn.LogSoftmax(dim=-1)
          self.optimizer = optim.Adam(self.parameters(),lr=lr)
    def forward(self, x):
          if self.embedings_dim==1: x=x.unsqueeze(-1)
          x = self.emmbedings(x)
          for i in range(len(self.layers)):
              out, attention = self.layers[i][0](x, x, x)
              out = self.layers[i][1](out + x)
              fc = self.layers[i][2](out)
              fc = self.layers[i][3](fc)
              fc = self.layers[i][4](fc)
              fc = self.layers[i][5](fc)
              x = self.layers[i][6](out + fc)
          x, _ = torch.max(x, dim = 1)
          x = self.final_layer(x)
          x = self.logsoft(x)
          return x