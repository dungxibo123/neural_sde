import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
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
parser.add_argument("-f", "--folder", type=str, default="./data/mnist", help="Folder /path/to/mnist/dataset")
parser.add_argument("-r", "--result", type=str, default="./result", help="Folder where the result going in")
parser.add_argument("-tr", "--train", type=int, default=20000, help="Number of train images")
parser.add_argument("-vl", "--valid", type=int, default=5000, help="Number of validation images")
parser.add_argument("-lr", "--learning-rate",type=float, default=1e-3, help="Learning rate in optimizer")
parser.add_argument("-md", "--model", type=str, default="./model", help="Where model going to")
parser.add_argument("-pr", "--parallel", type=bool, default=False, help="Parallel or not")
parser.add_argument("-wd", "--weight-decay", type=float, default=0.1, help="Weight decay")
args = parser.parse_args()


 # CONSTANT 
device = args.device
#torch.device("cuda")
EPOCHS=args.epochs
BATCH_SIZE=args.batch_size
DATA_DIR=args.folder
RESULT_DIR=args.result
TRAIN_NUM=args.train
VALID_NUM=args.valid
#TEST_NUM=60000-TRAIN_NUM-VALID_NUM
DATA_DISTRIBUTION=[TRAIN_NUM,VALID_NUM,TEST_NUM]
MODEL_DIR=args.model
PARALLEL=args.parallel
WEIGHT_DECAY=args.weight_decay



def train_model(model, optimizer, train_loader, val_loader,loss_fn, lr_scheduler=None, epochs=100, parallel=None):
    #print(model.eval())
    print(f"Numbers of parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    best_model, best_acc, best_epoch = None, 0, 0
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    for epoch_id in tqdm(range(epochs)):
        total = 0
        correct = 0
        running_loss = 0
        print(f"Start epoch number: {epoch_id + 1}")
#        print(next(enumerate(train_loader,0)))
        loads = list(enumerate(train_loader,0))
        for batch_id, data in loads:
            inputs, labels = data
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
        if lr_scheduler:
            lr_scheduler.step()
        acc = correct / total

        history["acc"].append(acc)
        history["loss"].append(running_loss)
        if parallel is not None:
            val_loss, val_acc = model.module.evaluate(val_loader)
        else:
            val_loss, val_acc = model.evaluate(val_loader)
        if acc > best_acc and val_acc > 0.85:
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
        print("Epoch(s) {:04d}/{:04d} | acc: {:.05f} | loss: {:.09f} | val_acc: {:.05f} | val_loss: {:.09f} | Best epochs: {:04d} | Best acc: {:09f}".format(
            epoch_id + 1, epochs, acc, running_loss, val_acc, val_loss, best_epoch, best_acc
            ))

    return history, best_model, best_epoch, best_acc



def main(ds_len, train_ds, valid_ds,model_type = "ode",data_name = "mnist_50",batch_size=32,epochs=100, lr=1e-3,train_num = 0, valid_num = 0, test_num = 0, weight_decay=None, device="cpu", result_dir="./result", model_dir="./model", parallel=None):

    pass
