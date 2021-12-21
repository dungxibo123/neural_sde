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


class LinearDrift(nn.Module):
    def __init__(self,in_hidden,out_hidden, device="cpu"): 
        super(LinearDrift,self).__init__()
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

class LinearDiffusion(nn.Module):
    def __init__(self, in_hidden, out_hidden, device="cpu"): 
        super(LinearDiffusion,self).__init__()
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

class ConvolutionDrift(nn.Module):
    def __init__(self, in_channel, size=32, device="cpu"):
        super(ConvolutionDrift,self).__init__()
        self.size=size
        self.in_channel=in_channel
        self.net = nn.Sequential(*[
            nn.Conv2d(in_channel, 64,3,padding=1),
            nn.GroupNorm(32,64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.GroupNorm(32,64),
            nn.ReLU(),
            nn.Conv2d(64, in_channel,3,padding=1),
            nn.ReLU(),
            
        ]).to(device)
    def forward(self,x):
        bs = x.shape[0]
        out = x.view(bs, self.in_channel, self.size, self.size)
#        print(f"{out.shape}\n\n\n\n")
        out = self.net(out)
        out = out.view(bs,-1)
        return out
class ConvolutionDiffusion(nn.Module):
    def __init__(self, in_channel, size=32, device="cpu"):
        super(ConvolutionDiffusion,self).__init__()
        self.size=size
        self.in_channel=in_channel
        self.net = nn.Sequential(*[
            nn.Conv2d(in_channel, 64,3,padding=1),
            nn.GroupNorm(32,64),
            nn.ReLU(),
            nn.Conv2d(64, 64,3,padding=1),
            nn.GroupNorm(32,64),
            nn.ReLU(),
            nn.Conv2d(64,in_channel * 2,3,padding=1),
            nn.ReLU(),
            
        ]).to(device)
    def forward(self,x):
        bs = x.shape[0]
        out = x.view(bs, self.in_channel, self.size, self.size)
        out = self.net(out)
        out = out.view(bs,-1)
        return out
        
class SDEBlock(nn.Module):
    noise_type="general"
    sde_type="ito"
    def __init__(self, state_size, brownian_size, batch_size, option = dict(), device="cpu", parallel=False,
        method="euler", noise_type="general", integral_type="ito", is_ode=False, input_conv_channel = 64,input_conv_size=6, layers="linear"):
        super(SDEBlock, self).__init__()
        self.noise_type=noise_type
        self.sde_type=integral_type
        self.state_size = state_size
        self.batch_size = batch_size
        self.brownian_size = brownian_size
        self.parallel = parallel
        self.device = device
        self.is_ode = is_ode
        if parallel:
            self.batch_size = int(self.batch_size / 2)
        
        if layers=="linear":
            self.drift = LinearDrift(state_size, state_size).to(device)
            self.diffusion = LinearDiffusion(state_size, state_size * brownian_size).to(device)

        elif layers=="conv":
            self.drift = ConvolutionDrift(input_conv_channel, input_conv_size).to(device)
            self.diffusion = ConvolutionDiffusion(input_conv_channel, input_conv_size).to(device)


    def f(self,t,x):  
        out = self.drift(x)
        return out
        #return self.f(x)
    def g(self,t,x):
        if self.is_ode:
            out =  torch.zeros_like((self.batch_size,self.state_size, self.brownian_size)).to(self.device)
            return out
        out = self.diffusion(x)
        
        out =  out.view(self.batch_size, self.state_size, self.brownian_size)
        return out

        


   

"""
SDEBlock: LinearDrift dx + LinearDiffusion dW
SDENet: fe -> SDEBlock -> fcc
"""
    
    
class SDENet(Model):
    def __init__(self, input_channel, input_size, state_size, brownian_size, batch_size, option = dict(), method="euler",
        noise_type="general", integral_type="ito", device = "cpu", is_ode=False,parallel = False):
        """"""""""""""""""""""""
        super(SDENet, self).__init__()
        self.batch_size = batch_size
        self.parallel = parallel
        self.option = option
        state_size = 64 * 6 * 6
        self.device = device
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
#        print(f"Init features extraction layer with device {self.device}")
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
                parallel=parallel,
                is_ode=is_ode,
                layers="conv"
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
#        print(f"After the feature extraction step, shape is: {out.shape}")
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
#f = LinearDrift(2304,16)
#print(f(torch.rand(128,2304)).shape) # OK, SHAPE: [32,16]
#g = LinearDiffusion2304,16 * 3)
#print(g(torch.rand(32,2304)).shape) # OK, SHAPE: [32,48]

# Test 3
#f = ConvolutionDrift(64,6)
#g = ConvolutionDiffusion(64,6)
#print(f(torch.rand(32,2304)).shape)
#print(g(torch.rand(32,2304)).shape)

# Test 4
sde = SDENet(input_channel=3,input_size=32,state_size=128,brownian_size=2,batch_size=32,device="cuda", parallel=False,option=dict(step_size=0.1)).to("cuda")
u = torch.rand((32,3,32,32)).to("cuda")
print(sde(u).shape)
