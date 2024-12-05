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
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
# import seaborn as sns
import torch.nn.init as init
import pickle
import wandb
# Custom Libraries
import utils
import random
# # Tensorboard initialization
# writer = SummaryWriter()

# # Plotting Style
# sns.set_style('darkgrid')
# wandb.init(project="pruning_project", config={"arch":args.arch_type, "learning_rate": args.lr, "prune_percent":args.prune_percent, "prune_iter": args.prune_iterations})
# Main
def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reinit = args.prune_type == "reinit"

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

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet 

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)   
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet  
    
    # If you want to add extra datasets paste here

    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
    #train_loader = cycle(train_loader)
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

    # Copying and Saving Initial State
    # initial_state_dict = copy.deepcopy(model.state_dict())
    # utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
    # torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar")

    # Making Initial Mask
    # make_mask(model)

    # Optimizer and Loss
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

    # Layer Looper
    # for name, param in model.named_parameters():
    #     print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Pruning Loop
    best_accuracy = 0
    comp=1000
    l1_lambda=args.l1_lambda
    threshold = 1e-5
    lr=args.lr
    # sparsity = args.prune_percent / 100  # Convert to fraction
    # n = args.prune_iterations-1
    # r = sparsity ** (1 / n)  # Per-iteration remaining fraction
    # per_iteration_prune_percent = (1 - r) * 100  # Convert back to percentage
    comp = utils.print_nonzeros(model)
    sparsity = calculate_sparsity(model)
    for iter_ in tqdm(range(args.end_iter), desc="Training Epochs"):
            # Validation and Model Saving
        metrics = {}
        
        # wandb.log({"Sparsity": sparsity})

        if iter_ % args.valid_freq == 0:
            accuracy = test(model, test_loader, criterion)
                # if accuracy > best_accuracy:
                #     best_accuracy = accuracy
                #     utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
                #     torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")
                # wandb.log({"Accuracy/test": accuracy, "Best Accuracy": best_accuracy, "Compression": comp[_ite]})
            metrics.update({
            "Accuracy/test": accuracy,
            "Best Accuracy": best_accuracy,
            "Compression": comp,
            "Sparsity": sparsity})
            # Training
        train_loss = train(model, train_loader, optimizer, criterion,l1_lambda,lr)
          # Adjust threshold based on your needs
        apply_weight_threshold(model, threshold)
        metrics["Train Loss"] = train_loss
        comp = utils.print_nonzeros(model)
        sparsity = calculate_sparsity(model)
        print(f"Sparsity: {sparsity:.2f}%")
            # print(_ite, iter_)
        wandb.log(metrics)
    wandb.finish()
def proximal_operator(model, l1_lambda, lr):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.data = torch.sign(param.data) * torch.clamp(param.data.abs() - l1_lambda * lr, min=0.0)
                          
   
# Function for Training
def train(model, train_loader, optimizer, criterion,l1_lambda,lr):
    # EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        # l1_penalty = 0
        # for name, param in model.named_parameters():
        #     if 'weight' in name:  # Apply L1 only to weights, not biases
        #         l1_penalty = l1_penalty + param.abs().sum()
        # print(l1_penalty)
        # train_loss = train_loss + l1_lambda * l1_penalty
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        # for name, p in model.named_parameters():
        #     if 'weight' in name:
        #         tensor = p.data.cpu().numpy()
        #         grad_tensor = p.grad.data.cpu().numpy()
        #         grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
        #         p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
        proximal_operator(model, l1_lambda, lr)
    return train_loss.item()

# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy



    
def calculate_sparsity(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    sparsity = 100. * zero_params / total_params
    return sparsity

def apply_weight_threshold(model, threshold):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = param.data * (param.data.abs() > threshold).float()

def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__=="__main__":
    
    #from gooey import Gooey
    #@Gooey      
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # If you are using CUDA to run your PyTorch code, set this too:
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=10, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="l1", type=str, help="l1_reg | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=3, type=int, help="Pruning iterations count")
    parser.add_argument("--l1_lambda", default=1e-3, type=float, help="L1 regularization strength")

    
    args = parser.parse_args()
    wandb.init(project="pruning_project", config={"arch":args.arch_type, "dataset":args.dataset, "learning_rate": args.lr, "l1 lambda": args.l1_lambda, "prune_percent":args.prune_percent, "end_iter":args.end_iter, "prune_iter": args.prune_iterations, "train_iter": args.end_iter, "prune_type": args.prune_type})


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    
    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)
