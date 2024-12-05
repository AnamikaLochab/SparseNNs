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
# Updated Training Loop with New Features
def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loader with Dynamic Dataset Support
    if args.dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        traindataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        traindataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet

    elif args.dataset == "fashionmnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        traindataset = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        traindataset = datasets.CIFAR100('../data', train=True, download=True, transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet

    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")
    
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False)

    # Importing Network Architecture Dynamically
    global model
    if args.arch_type == "fc1":
        model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg":
        model = vgg.VGG().to(device)
    elif args.arch_type == "resnet":
        model = resnet.ResNet().to(device)
    elif args.arch_type == "densenet" and args.dataset == "cifar10":
        model = densenet.DenseNet().to(device)
    else:
        raise ValueError(f"Architecture {args.arch_type} is not supported for {args.dataset}.")

    # Weight Initialization
    model.apply(weight_init)

    # Copy and Save Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
    torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar")

    # Making Initial Mask
    make_mask(model)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Iterative Pruning Loop
    best_accuracy = 0
    comp, bestacc = np.zeros(args.prune_iterations, float), np.zeros(args.prune_iterations, float)
    sparsity = args.prune_percent / 100  # Convert to fraction
    n = args.prune_iterations-1
    r = sparsity ** (1 / n)  # Per-iteration remaining fraction
    per_iteration_prune_percent = (1 - r) * 100
    print("Calling global")
    for _ite in range(args.start_iter, args.prune_iterations):
        
        if _ite != 0:
            prune_by_global_percentile(per_iteration_prune_percent)
            
            # prune_dead_neurons(model)  # Prune dead neurons
            # adjust_dropout(model, args.prune_percent / 100)  # Adjust dropout rates
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*(1/10), weight_decay=1e-4)
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{args.prune_iterations}]: ---")

        # Track compression
        comp[_ite] = utils.print_nonzeros(model)

        for iter_ in tqdm(range(args.end_iter), desc="Training Epochs"):
            # Validation and Model Saving
            metrics = {}
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader, criterion)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
                    torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")
                metrics.update({
            "Accuracy/test": accuracy,
            "Best Accuracy": best_accuracy,
            "Compression": comp[_ite]})
            # Training with L2 Regularization
            train_loss = train(model, train_loader, optimizer, criterion)
            metrics["Train Loss"] = train_loss
            # print(_ite, iter_)
            wandb.log(metrics)

    wandb.finish()

# Function to apply L2 regularization during training
def apply_l2_regularization(optimizer, lambda_l2=1e-4):
    """
    Applies L2 regularization to the optimizer by adding the L2 penalty to the gradients.
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.requires_grad:
                param.grad.data.add_(lambda_l2 * param.data)         

# Function for Training with L2 Regularization
def train_with_l2(model, train_loader, optimizer, criterion, lambda_l2=1e-4):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Apply L2 regularization
        # apply_l2_regularization(optimizer, lambda_l2)

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()
 
# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()

# Function to adjust Dropout after Pruning
def adjust_dropout(model, prune_ratio):
    """
    Adjusts dropout rates based on pruning ratio.
    Dropout rates are reduced proportionally to preserve model capacity.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            original_p = m.p
            m.p = max(0, original_p * (1 - prune_ratio))  # Reduce dropout rate
            print(f"Adjusted Dropout: {original_p} -> {m.p}")

# Function to prune neurons with zero input/output connections
def prune_dead_neurons(model):
    """
    Prunes neurons that have zero input or zero output connections.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check for dead input neurons (all weights to this neuron are zero)
            weight_matrix = module.weight.data.cpu().numpy()
            dead_inputs = np.all(weight_matrix == 0, axis=0)
            if np.any(dead_inputs):
                print(f"Pruning {np.sum(dead_inputs)} dead input neurons in {name}")

            # Check for dead output neurons (all weights from this neuron are zero)
            dead_outputs = np.all(weight_matrix == 0, axis=1)
            if np.any(dead_outputs):
                print(f"Pruning {np.sum(dead_outputs)} dead output neurons in {name}")

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

# Magnitude-based Prune by Percentile module
def magnitude_prune_by_percentile(percent, **kwargs):
    global step
    global mask
    global model

    # Calculate percentile value for pruning
    step = 0
    prune_rates = {'classifier.4.weight': 0.5, 'default': 1}
    for name, param in model.named_parameters():

        # We do not prune bias term
        if 'weight' in name:
            prune_rate = prune_rates['classifier.4.weight'] if 'classifier.4.weight' in name else prune_rates['default']
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent*prune_rate)

            # Convert Tensors to numpy and calculate mask
            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
            
            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0
def prune_by_global_percentile(percent):
    # Collect all weights in a list
    global step
    global mask
    global model

    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:  # Assuming you're only pruning convolutional layers
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]
            all_weights.extend(abs(alive))
    
    # Calculate global percentile value
    global_percentile_value = np.percentile(all_weights, percent)

    # Apply pruning based on global percentile value
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            weight_dev = param.device
            new_mask = np.where(abs(tensor) < global_percentile_value, 0, mask[step])
            
            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0
# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None] * step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
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
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)

if __name__=="__main__":
    # Arguement Parser
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # If you are using CUDA to run your PyTorch code, set this too:
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default= 1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="magnitude", type=str, help="lt | reinit | magnitude")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, choices=["mnist", "cifar10", "fashionmnist", "cifar100"],
                    help="Dataset to use: mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, 
                    choices=["fc1", "lenet5", "alexnet", "vgg", "resnet", "densenet"],
                    help="Architecture to use: fc1 | lenet5 | alexnet | vgg | resnet | densenet")
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=35, type=int, help="Pruning iterations count")

    args = parser.parse_args()
    wandb.init(project="pruning_project", config={"arch":args.arch_type, "dataset":args.dataset, "learning_rate": args.lr, "prune_percent":args.prune_percent, "end_iter":args.end_iter, "prune_iter": args.prune_iterations, "train_iter": args.end_iter, "prune_type": args.prune_type})

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    # Looping Entire process
    main(args, ITE=1)