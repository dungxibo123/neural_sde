import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde
from torchdiffeq import odeint_adjoint as odeint

import sys
import os

sys.path.insert(0,os.path.abspath(__file__))
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
    def evaluate(self, test_loader):
        correct = 0
        total = 0 
        running_loss = 0
        count = 0 
        with torch.no_grad():
            for test_data in test_loader:
                count += 1
                data, label = test_data
                outputs = self.forward(data)
                _, correct_labels = torch.max(label, 1) 
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == correct_labels).sum().item()
                running_loss += F.torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs.float(), label.float()).item()
        acc = correct / total
        running_loss /= count
        
        return running_loss,acc
    def loss_surface(self):
        pass

def F(nn.Module):
    def __init__(self,state_size, input_size, output_size, hidden_sizes): 
        hidden_sizes.reverse(); hidden_sizes.append(input_size); hidden_sizes.reverse(); hidden_sizes.append(output_size)
        layers = []
        for i in range(1, len(hidden_sizes)):
            layers.append(hidden_sizes[i - 1], hidden_sizes[i])
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

def G(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes): 
        hidden_sizes.reverse(); hidden_sizes.append(input_size); hidden_sizes.reverse(); hidden_sizes.append(output_size)
        layers = []
        for i in range(1, len(hidden_sizes)):
            layers.append(hidden_sizes[i - 1], hidden_sizes[i])
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)
class SDEBlock(nn.Module):
    noise_type="general"
    sde_type="ito"
    def __init__(self, state_size, brownian_size, batch_size, drift_hidden, diffusion_hidden,config = dict(), device="cpu", parallel=False):
        super(SDEBlock, self).__init__()
        self.state_size = state_size
        self.batch_szie = batch_size
        self.brownian_size = brownian_size
        self.parallel = parallel
        if not parallel:
            self.f = F(state_size,state_size, drift_hidden)
            self.g = G(state_size, state_size * brownian_size)
        else:
            self.f = torch.nn.DataParallel(F(state_size,state_size, drift_hidden))
            
            self.g = torch.nn.DataParallel(G(state_size, state_size * brownian_size))


    def f(self,t,x): 
        return self.f.module(x) if self.parallel else self.f(x)
    def g(self,t,x):
        if parallel:
            out = self.g.module(x)
        else:
            out = self.g.module(x)
        return out.view(batch_size, state_size, brownian_size)


        


   

"""
SDEBlock: Drift dx + Diffusion dW
SDENet: fe -> SDEBlock -> fcc
"""
    
    

class SDENet(nn.Module):
    def __init__(self, input_size, state_size, brownian_size, drift_hidden, diffusion_hidden, sde_config = dict(), device = "cpu", parallel = False):
    super(SDENet, self).__init__()






