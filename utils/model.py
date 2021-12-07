import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde
from torchsde import sdeint_adjoint as sdeint
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


class Drift(nn.Module):
    def __init__(self,in_hidden,out_hidden, device="cuda"): 
        super(Drift,self).__init__()
        self.net = nn.Sequential(*[
            nn.Linear(in_hidden,8),
            nn.ReLU(),
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,out_hidden),
            nn.ReLU(),

        ]).to(device)
    def forward(self,x):
        return self.net(x)

class Diffusion(nn.Module):
    def __init__(self, in_hidden, out_hidden, device="cuda"): 
        super(Diffusion,self).__init__()
        self.net = nn.Sequential(*[
            nn.Linear(in_hidden, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,out_hidden),
            nn.ReLU()
        ]).to(device)
    def forward(self,x):
        return self.net(x)
class SDEBlock(nn.Module):
    noise_type="general"
    sde_type="ito"
    def __init__(self, state_size, brownian_size, batch_size, option = dict(), device="cpu", parallel=False,
        method="euler", noise_type="general", integral_type="ito"):
        super(SDEBlock, self).__init__()
        self.noise_type=noise_type
        self.sde_type=integral_type
        self.state_size = state_size
        self.batch_size = batch_size
        self.brownian_size = brownian_size
        self.parallel = parallel
        if parallel:
            self.batch_size = int(self.batch_size / 2)
        
        self.drift = Drift(state_size, state_size).to(device)
        self.diffusion = Diffusion(state_size, state_size * brownian_size).to(device)

    def f(self,t,x):  
        out = self.drift(x)
        return out
        #return self.f(x)
    def g(self,t,x):
        out = self.diffusion(x)
        return out.view(self.batch_size, self.state_size, self.brownian_size)


        


   

"""
SDEBlock: Drift dx + Diffusion dW
SDENet: fe -> SDEBlock -> fcc
"""
    
    
class SDENet(Model):
    def __init__(self, input_channel, input_size, state_size, brownian_size, batch_size, option = dict(), method="euler",
        noise_type="general", integral_type="ito", device = "cpu", parallel = False):
        """"""""""""""""""""""""
        super(SDENet, self).__init__()
        self.batch_size = batch_size
        self.parallel = parallel
        self.option = option
        state_size = 64 * 6 * 6
        self.fe = nn.Sequential(*[
            nn.Conv2d(input_channel,16,3,1),
            nn.GroupNorm(8,16),
            nn.ReLU(),
            nn.Conv2d(16,32,3,2),
            nn.GroupNorm(16,32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,2),
            nn.GroupNorm(32,64),
            nn.ReLU(),

        ]).to(device)
        # Output shape from (B,3,32,32) -> (B,64,6,6)
        if parallel:
            self.batch_size = int(self.batch_size /  2)
        self.rm = SDEBlock(
                state_size=state_size,
                brownian_size = brownian_size,
                batch_size = batch_size,
                option=option,
                method=method,
                integral_type=integral_type,
                noise_type=noise_type,
                device=device,
                parallel=parallel
            ).to(device)


        self.fcc = nn.Sequential(*[
            nn.Linear(state_size,10),
            nn.Softmax(dim = 1)
        ]).to(device)


        self.intergrated_time = torch.Tensor([0.0,1.0]).to(device)
        self.device = device
        self.method = method
    def forward(self,x):
        out = self.fe(x)
#        print(f"Shape after Feature Extraction Layer: {out.shape}")
        out = out.view(self.batch_size,-1)
#        print(f"Device of out {out.device}")
#        print(f"Shape before the SDE Intergral: {out.shape}")
        out = sdeint(self.rm,out,self.intergrated_time, options=self.option,method="euler", atol=5e-2,rtol=5e-2,dt=0.05, dt_min=0.05)[-1]
        out = self.fcc(out)
        return out



# Test
#sde = SDENet(input_channel=3,input_size=32,state_size=128,brownian_size=2,batch_size=32,device="cuda", parallel=False,option=dict(step_size=0.1)).to("cuda")
#data = torch.rand((32,3,32,32)).to("cuda")
#print(sde(data).shape)


# Test 2
#f = Drift(2304,16)
#print(f(torch.rand(128,2304)).shape) # OK, SHAPE: [32,16]
#g = Diffusion2304,16 * 3)
#print(g(torch.rand(32,2304)).shape) # OK, SHAPE: [32,48]
