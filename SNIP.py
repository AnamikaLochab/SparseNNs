# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
import torch.nn.init as init
import pickle
import wandb
# Custom Libraries
import utils

# Main
def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)      
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet 

    # If you want to add extra datasets paste here
    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)
    
    # Importing Network Architecture
    global model
    if args.arch_type == "fc1":
       model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16().to(device)  
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(device)   
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(device)   
    
    else:
        print("\nWrong Model choice\n")
        exit()

    # Weight Initialization
    model.apply(weight_init)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Perform SNIP Pruning before training
    snip_prune(model, train_loader, criterion, args.prune_percent, device)

    # Training Loop
    best_accuracy = 0
    for iter_ in tqdm(range(args.end_iter), desc="Training Epochs"):
        # Validation and Model Saving
        if iter_ % args.valid_freq == 0:
            accuracy = test(model, test_loader, criterion)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
                torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{iter_}_model_snip.pth.tar")
            wandb.log({"Accuracy/test": accuracy, "Best Accuracy": best_accuracy})

        # Training
        train_loss = train(model, train_loader, optimizer, criterion)
        wandb.log({"Train Loss": train_loss})

    wandb.finish()
                           
# Function for Training
def train(model, train_loader, optimizer, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()
        optimizer.step()
    return train_loss.item()

# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# SNIP Pruning Function
def snip_prune(model, train_loader, criterion, prune_percent, device):
    model.zero_grad()
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()

    all_scores = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_scores.append(torch.abs(param.grad))
    all_scores = torch.cat([torch.flatten(x) for x in all_scores])
    num_params_to_keep = int(len(all_scores) * (100 - prune_percent) / 100)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    # Prune the weights below the threshold
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask = (torch.abs(param.grad) >= acceptable_score).float()
            param.data *= mask

# Function for Initialization
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)

if __name__=="__main__":
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--end_iter", default=100, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")

    args = parser.parse_args()
    wandb.init(project="pruning_project", config={"arch":args.arch_type, "learning_rate": args.lr, "prune_percent":args.prune_percent})

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    # Run main
    main(args, ITE=1)
