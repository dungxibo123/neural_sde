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
                data = data.to(self.device)
                label = label.to(self.device)
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
class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class Norm(nn.Module):
    def __init__(self, dim):
        super(Norm, self).__init__()
        self.norm = nn.GroupNorm(min(dim,32), dim)
    def forward(self,x):
        return self.norm(x)

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
    def forward(self,t,x):
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
    def forward(self,t,x):
        return self.net(x)

class ConvolutionDrift(nn.Module):
    def __init__(self, in_channel, size=32, device="cpu"):
        super(ConvolutionDrift,self).__init__()
        self.size=size
        self.in_channel=in_channel
        self.conv1 = ConcatConv2d(in_channel, 64,ksize=3,padding=1)
        self.norm1 = Norm(64) 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConcatConv2d(64, 64, ksize=3,padding=1)
        self.norm2 = Norm(in_channel) 
        
    def forward(self,t,x):
        bs = x.shape[0]
        out = x.view(bs, self.in_channel, self.size, self.size)
#        print(f"{out.shape}\n\n\n\n")
        out = self.conv1(t,out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(t,out)
        out = self.norm2(out)
        out = self.relu(out)
        out = out.view(bs,-1)
        return out
class ConvolutionDiffusion(nn.Module):
    def __init__(self, in_channel, size=32, brownian_size = 2, device="cpu"):
        super(ConvolutionDiffusion,self).__init__()
        self.size=size
        self.in_channel=in_channel
#        self.net = nn.Sequential(*[
#            nn.Conv2d(in_channel, 64,3,padding=1),
#            nn.GroupNorm(32,64),
#            nn.ReLU(),
#            nn.Conv2d(64, 64,3,padding=1),
#            nn.GroupNorm(32,64),
#            nn.ReLU(),
#            nn.Conv2d(64,in_channel * 2,3,padding=1),
#            nn.ReLU(),
#            
#        ]).to(device)
        self.relu = nn.ReLU()
        self.norm1 = Norm(64)
        self.conv1 = ConcatConv2d(in_channel, 64, ksize=3, padding = 1)
        self.conv2 = ConcatConv2d(64,64, ksize=3, padding = 1)
        self.norm2 = Norm(64)
        self.conv3 = ConcatConv2d(64, in_channel * brownian_size, ksize = 3, padding = 1)
        self.norm3 = Norm(in_channel * brownian_size)
    def forward(self,t,x):
        bs = x.shape[0]
        out = x.view(bs, self.in_channel, self.size, self.size)
        # out = self.net(out)
        out = self.conv1(t,out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(t,out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(t,out)
        out = self.norm3(out)
        out = self.relu(out)
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
            self.diffusion = ConvolutionDiffusion(input_conv_channel, input_conv_size, brownian_size = self.brownian_size).to(device)


    def f(self,t,x):  
        out = self.drift(t,x)
        return out
        #return self.f(x)
    def g(self,t,x):
        bs = x.shape[0]
        if self.is_ode:
            out =  torch.zeros_like((self.batch_size,self.state_size, self.brownian_size)).to(self.device)
            return out
        out = self.diffusion(t,x)
        
        out =  out.view(bs, self.state_size, self.brownian_size)
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
        self.input_size = input_size
        self.option = option
        self.input_channel = input_channel
        #state_size = 64 * 14 * 14
        self.device = device
        self.fe = nn.Sequential(*[
            nn.Conv2d(input_channel,16,3,padding=1),
            nn.GroupNorm(8,16),
            nn.ReLU(),
            nn.Conv2d(16,32,4,padding=2),
            nn.GroupNorm(16,32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,2),
            nn.GroupNorm(32,64),
            nn.ReLU(),

        ]).to(device)
        state_size, input_conv_channel, input_conv_size = self.get_state_size()
        self.input_conv_channel = input_conv_channel
        self.input_conv_size = input_conv_size
        
#        print(state_size, input_conv_channel, input_conv_size, "ehehehehehe\n\n\n\n")
#        print(f"Init features extraction layer with device {self.device}")
        # Output shape from (B,3,32,32) -> (B,64,14,14)
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
                input_conv_channel=input_conv_channel,
                input_conv_size=input_conv_size,
                layers="conv"
            ).to(device)


        self.fcc = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_conv_channel,10),
            nn.Softmax(dim = 1)
        ]).to(device)


        self.intergrated_time = torch.Tensor([0.0,1.0]).to(device)
        self.device = device
        self.method = method
    def get_state_size(self):
        out = torch.rand((1,self.input_channel,self.input_size, self.input_size)).to(self.device)
        with torch.no_grad():
            shape = self.fe(out)
        return shape.view(1,-1).shape[-1], shape.shape[1], shape.shape[2]
    def forward(self,x):
        out = self.fe(x)
        bs = x.shape[0]
#        print(f"Shape after Feature Extraction Layer: {out.shape}")
        out = out.view(bs,-1)
#        print(f"After the feature extraction step, shape is: {out.shape}")
#        print(f"Device of out {out.device}")
#        print(f"Shape before the SDE Intergral: {out.shape}")
        out = sdeint(self.rm,out,self.intergrated_time, options=self.option,method="euler", atol=5e-2,rtol=5e-2, dt=0.1, dt_min=0.05)[-1]
        out = out.view(bs,self.input_conv_channel, self.input_conv_size, self.input_conv_size)
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

# Test 4
if __name__ == "__main__":
    import time
    sde = SDENet(input_channel=3,input_size=32,state_size=128,brownian_size=2,batch_size=256,device="cuda", parallel=False,option=dict(step_size=0.1)).to("cuda")
    bz = 1024
    u = torch.rand((bz,3,32,32)).to("cuda")
    out = sde(u)
#    f = ConvolutionDrift(64,6)
#    g = ConvolutionDiffusion(64,6)
#    print(f(torch.rand(32,2304)).shape)
#    print(g(torch.rand(32,2304)).shape)
    tar = torch.zeros_like(out).to("cuda")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out,tar)
    now = time.time()
    loss.backward()
    print(f"Time for backward process: {time.time() - now}")
