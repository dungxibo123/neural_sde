import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import torchvision
from tqdm import tqdm
import os.path
import argparse

import json
from utils import *



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=str, default="cpu", help="Device which the PyTorch run on")
parser.add_argument("-bs", "--batch-size", type=int, default=1024, help="Batch size of 1 iteration")
parser.add_argument("-ep", "--epochs", type=int, default=200, help="Numbers of epoch")
parser.add_argument("-da", "--data", type=str, default="svhn", help="cifar10/svhn/mnist")
parser.add_argument("-r", "--result", type=str, default="./result", help="Folder where the result going in")
parser.add_argument("-tr", "--train", type=int, default=20000, help="Number of train images")
parser.add_argument("-vl", "--valid", type=int, default=5000, help="Number of validation images")
parser.add_argument("-lr", "--learning-rate",type=float, default=1e-3, help="Learning rate in optimizer")
parser.add_argument("-md", "--model", type=str, default="./model", help="Where model going to")
parser.add_argument("-pr", "--parallel", type=bool, default=False, help="Parallel or not")
parser.add_argument("-wd", "--weight-decay", type=float, default=5e-3, help="Weight decay")
parser.add_argument("-br", "--brownian-size", type=int, default=2, help="Brownian size")
parser.add_argument("-nt", "--noise-type", type=str, default="general", help="Type of noise")
parser.add_argument("-it", "--integral-type", type=str, default="ito", help="Ito or Stratonovich intergral")
parser.add_argument("-so", "--solver", type=str, default="euler", help="Solver")
parser.add_argument("-rlr", "--reduce-lr",type=float, default=0.1, help="Reduce On Plateau")
args = parser.parse_args()


 # CONSTANT 
device = args.device
#torch.device("cuda")
EPOCHS=args.epochs
BATCH_SIZE=args.batch_size
DATA_DIR=f"./data/{args.data}"
DATA_TYPE=args.data
RESULT_DIR=args.result
TRAIN_NUM=args.train
VALID_NUM=args.valid
TEST_NUM=50000-TRAIN_NUM-VALID_NUM
DATA_DISTRIBUTION=[TRAIN_NUM,VALID_NUM,TEST_NUM]
MODEL_DIR=args.model
PARALLEL=args.parallel
INTEGRAL_TYPE=args.integral_type
SOLVER=args.solver
BROWNIAN_SIZE=args.brownian_size
NOISE_TYPE=args.noise_type
WEIGHT_DECAY=args.weight_decay



def train(model, optimizer, train_loader, val_loader,loss_fn, lr_scheduler=None, epochs=100, parallel=None):
    #print(model.eval())
    print(f"> Numbers of parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    best_model, best_acc, best_epoch = None, 0, 0
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    for epoch_id in tqdm(range(epochs)):
        total = 0
        correct = 0
        running_loss = 0
        print(f"\n > Start epoch number: {epoch_id + 1}")
#        print(next(enumerate(train_loader,0)))
        loads = list(enumerate(train_loader,0))
        for batch_id, data in loads:
            start = time.time() 
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            _, correct_labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == correct_labels).sum().item()
            loss = loss_fn(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
            end = time.time() 
            print("\t\t\t >>> Time for the batch number {batch_id + 1} in epoch number {epoch_id + 1} is {end - start}")
        if lr_scheduler:
            lr_scheduler.step()
        acc = correct / total

        history["acc"].append(acc)
        history["loss"].append(running_loss)
        if parallel:
            val_loss, val_acc = model.module.evaluate(val_loader)
        else:
            val_loss, val_acc = model.evaluate(val_loader)
        if acc > best_acc and val_acc > 0:
            best_acc = acc
            best_epoch = epoch_id + 1
            best_model = model
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        running_loss /= len(loads)
        #print(f"Epoch(s) {epoch_id + 1} | loss: {loss} | acc: {acc} | val_loss: {val_loss} | val_acc: {val_acc}")
        checkpoint = {
            'epoch': epoch_id + 1,
            'model': model,
            'best_epoch': best_epoch,
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, "./checkpoints/checkpoint.pt")
        print("\t >>> Epoch(s) {:04d}/{:04d} | acc: {:.05f} | loss: {:.09f} | val_acc: {:.05f} | val_loss: {:.09f} | Best epochs: {:04d} | Best acc: {:09f}".format(
            epoch_id + 1, epochs, acc, running_loss, val_acc, val_loss, best_epoch, best_acc
            ))

    return history, best_model, best_epoch, best_acc



def main(ds_len, train_ds, valid_ds, model_type = "sde", data_name = "mnist_50", batch_size=32, epochs=100, lr=1e-3,
    train_num = 0, valid_num = 0, test_num = 0, weight_decay=0,reduce_lr = None, device="cpu", result_dir="./result", model_dir="./model",
    integral_type="ito", brownian_size=2, noise_type="general",solver="euler", parallel=None, option=dict()):
    # START THE MAIN PART
########################################################################################################################
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
    val_loader  = DataLoader(valid_ds, shuffle=True, batch_size= batch_size, drop_last=True)
    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    #    epochs= int(epochs * 2.5)
    model = SDENet(
        state_size=None,
        input_channel=3,
        input_size=32,
        brownian_size=brownian_size,
        batch_size=batch_size,
        option=option,
        parallel=parallel,
        device=device
    ).to(device)
    if parallel:
        model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if reduce_lr:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min")
    else:
        lr_scheduler = None
   
    history, best_model, best_epoch, best_acc = train(model=model,
                                                      train_loader=train_loader,
                                                      val_loader=val_loader,
                                                      optimizer=optimizer,
                                                      loss_fn=loss_fn,
                                                      epochs=epochs,
                                                      lr_scheduler=
                                                      lr_scheduler,
                                                      parallel=parallel)
    return best_model


DATA = None
if DATA_TYPE == "cifar10":
    DATA = torchvision.datasets.CIFAR10(DATA_DIR, download=True)
elif DATA_TYPE == "mnist":
    DATA = torchvision.datasets.MNIST(DATA_DIR,
                                   train=True,
                                   transform=None,
                                   target_transform=None, download=True)
elif DATA_TYPE=="svhn":
    DATA = torchvision.datasets.SVHN(DATA_DIR,download=True)
    

ds_len_, ds_ = preprocess_data(DATA,data_type=DATA_TYPE,device=device, sigma=None)
_, perturbed_ds_ = preprocess_data(DATA,data_type=DATA_TYPE, device=device, sigma=[15.])
sde_model = main( ds_len_, ds_, perturbed_ds_, device=device, model_type="sde", data_name=f"svhn_origin", batch_size=BATCH_SIZE, 
    epochs=EPOCHS, train_num=TRAIN_NUM, valid_num=VALID_NUM, test_num=TEST_NUM, result_dir=RESULT_DIR, parallel=PARALLEL, integral_type=INTEGRAL_TYPE, solver=SOLVER, noise_type=NOISE_TYPE, weight_decay=WEIGHT_DECAY)
"""
sigmas = [10.0, 15.0 , 20.0]
loaders = [(key,DataLoader(preprocess_data(DATA, sigma=[key], device=device, train=True)[1], batch_size=args.batch_size,drop_last=True)) for key in sigmas]
if isinstance(sde_model, nn.DataParallel): sde_model = sde_model.module
for k,l in loaders:
    _, sde_acc = sde_model.evaluate(l)
    print(f"SDEs for {k}-gaussian-pertubed SVHN = {sde_acc}")

"""
