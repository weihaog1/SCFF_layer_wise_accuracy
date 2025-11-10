"""
Modified SCFF CIFAR-10 Parallel Training for Layer-wise Accuracy Research
Tracks accuracy of each layer independently during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import ExponentialLR, StepLR, LinearLR

import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt

import torchvision
from torchvision.transforms import transforms, ToPILImage
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split,Subset
import argparse
import time

import numpy as np
from numpy import fft
import math
import json
from tqdm import tqdm
from datetime import datetime

# Try to import sklearn, if not available will use simple linear probe
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, using PyTorch linear classifier instead")


#custom the trainloader to include the augmented views of the original batch
torch.manual_seed(1234)
# Define the two sets of transformations
s = 0.5
transform1 = transforms.Compose([
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform2 = transforms.Compose([
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class DualAugmentCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, augment="No", *args, **kwargs):
        super(DualAugmentCIFAR10, self).__init__(root,*args, **kwargs)
        self.augment = augment

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img_pil = ToPILImage()(img)
        img_original = transform_train(img_pil)

        if self.augment == "single":
            img1 = transform1(img_pil)
            return img_original, img1, img_original, target
        elif self.augment == "dual":
            img1 = transform1(img_pil)
            img2 = transform2(img_pil)
            return img_original, img1, img2, target
        else:
            return img_original, target


class DualAugmentCIFAR10_test(torchvision.datasets.CIFAR10):
    def __init__(self, aug=False, *args, **kwargs):
        super(DualAugmentCIFAR10_test, self).__init__(*args, **kwargs)
        self.aug = aug

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = ToPILImage()(img)

        if self.aug:
            img = transform_train(img)
        else:
            img = transform_test(img)

        return img, target


def get_train(batchsize, augment, Factor):
    torch.manual_seed(1234)
    trainset = DualAugmentCIFAR10(root='./data', train=True, download=True, augment=augment)
    sup_trainset = DualAugmentCIFAR10_test(root='./data', aug = True, train=True, download=True)

    factor = Factor
    train_len = int(len(trainset) * factor)

    indices = torch.randperm(len(trainset)).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    train_data = Subset(trainset, train_indices)
    sup_train_data = Subset(sup_trainset, train_indices)
    val_data = Subset(sup_trainset, val_indices)

    testset = DualAugmentCIFAR10_test(root='./data',aug = False, train=False, download=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

    trainloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)

    if factor ==1:
        valloader = testloader
    else:
        valloader = DataLoader(val_data, batch_size=1000, shuffle=True, num_workers=2)

    sup_trainloader = DataLoader(sup_train_data, batch_size=64, shuffle=True, )

    return trainloader, valloader, testloader, sup_trainloader

def get_pos_neg_batch_imgcats(batch_pos1, batch_pos2, p = 1):
    batch_size = len(batch_pos1)
    batch_pos =torch.cat((batch_pos1, batch_pos2), dim = 1)

    #create negative samples
    random_indices = (torch.randperm(batch_size - 1) + 1)[:min(p,batch_size - 1)]
    labeles = torch.arange(batch_size)

    batch_negs = []
    for i in random_indices:
        batch_neg = batch_pos2[(labeles+i)%batch_size]
        batch_neg = torch.cat((batch_pos1, batch_neg), dim = 1)
        batch_negs.append(batch_neg)

    return batch_pos, torch.cat(batch_negs)

def stdnorm (x, dims = [1,2,3]):
    x = x - torch.mean(x, dim=(dims), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(dims), keepdim=True))
    return x

class standardnorm(nn.Module):
    def __init__(self, dims = [1,2,3]):
        super(standardnorm, self).__init__()
        self.dims = dims

    def forward(self, x):
        x = x - torch.mean(x, dim=(self.dims), keepdim=True);  x = x / (1e-10 + torch.std(x, dim=(self.dims), keepdim=True))
        return x

class L2norm(nn.Module):
    def __init__(self, dims = [1,2,3]):
        super(L2norm, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x / (x.norm(p=2, dim=(self.dims), keepdim=True) + 1e-10)

class triangle(nn.Module):
    def __init__(self):
        super(triangle, self).__init__()

    def forward(self, x):
        x = x - torch.mean(x, axis=1, keepdims=True)
        return F.relu(x)

class Conv2d(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        pad=0,
        batchnorm=False,
        normdims=[1,2,3],
        norm="stdnorm",
        bias=True,
        dropout=0.0,
        padding_mode="reflect",
        concat=True,
        act="relu"
    ):
        super(Conv2d, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.normdims = normdims
        self.concat = concat
        self.relu = torch.nn.ReLU()

        self.conv_layer = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            bias=bias
        )

        init.xavier_uniform_(self.conv_layer.weight)
        self.padding_mode = padding_mode
        self.F_padding = (pad, pad, pad, pad)

        if act == 'relu':
            self.act = torch.nn.ReLU()
        else:
            self.act = triangle()

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(self.input_channels, affine=False)
        else:
            self.bn1 = nn.Identity()

        if norm == "L2norm":
            self.norm = L2norm(dims = normdims)
        elif norm == "stdnorm":
            self.norm = standardnorm(dims = normdims)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.bn1(x)
        x = F.pad(x, self.F_padding, self.padding_mode)
        x = self.norm(x)

        if self.concat:
            lenchannel = x.size(1)//2
            out = self.conv_layer(x[:, :lenchannel]) + self.conv_layer(x[:, lenchannel:])
        else:
            out = self.conv_layer(x)

        return out


def evaluate_layer_accuracies(nets, pool, extra_pool, config, trainloader, testloader, num_layers):
    """
    Evaluate accuracy for each layer independently using a simple linear classifier.
    Trains on training set features, evaluates on test set features.

    Returns:
        dict: Dictionary with keys 'layer_0', 'layer_1', etc. containing accuracies
    """
    layer_accuracies = {}

    for layer_idx in range(num_layers):
        print(f"  Evaluating Layer {layer_idx}...")

        # Set networks to eval mode
        with torch.no_grad():
            for net in nets:
                net.eval()

            # Extract TRAINING features
            train_features_list = []
            train_labels_list = []

            for x, labels in trainloader:
                x = x.to(config.device)

                # Forward pass up to layer_idx
                for i in range(layer_idx + 1):
                    if nets[i].concat:
                        x = stdnorm(x, dims=config.dims_in)
                        x = torch.cat((x, x), dim=1)
                    x = pool[i](nets[i].act(nets[i](x)))

                # Apply extra pooling
                out = extra_pool[layer_idx](x)
                if config.stdnorm_out:
                    out = stdnorm(out, dims=config.dims_out)
                out = out.flatten(start_dim=1)

                train_features_list.append(out.cpu())
                train_labels_list.append(labels)

            # Extract TEST features
            test_features_list = []
            test_labels_list = []

            for x, labels in testloader:
                x = x.to(config.device)

                # Forward pass up to layer_idx
                for i in range(layer_idx + 1):
                    if nets[i].concat:
                        x = stdnorm(x, dims=config.dims_in)
                        x = torch.cat((x, x), dim=1)
                    x = pool[i](nets[i].act(nets[i](x)))

                # Apply extra pooling
                out = extra_pool[layer_idx](x)
                if config.stdnorm_out:
                    out = stdnorm(out, dims=config.dims_out)
                out = out.flatten(start_dim=1)

                test_features_list.append(out.cpu())
                test_labels_list.append(labels)

        # Concatenate all batches
        train_features = torch.cat(train_features_list, dim=0)
        train_labels = torch.cat(train_labels_list, dim=0)
        test_features = torch.cat(test_features_list, dim=0)
        test_labels = torch.cat(test_labels_list, dim=0)

        # Train classifier on TRAIN features, evaluate on TEST features
        if SKLEARN_AVAILABLE and train_features.shape[0] < 100000:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
            clf.fit(train_features.numpy(), train_labels.numpy())
            accuracy = clf.score(test_features.numpy(), test_labels.numpy())
        else:
            # Use PyTorch linear classifier
            classifier = nn.Linear(train_features.shape[1], 10).to(config.device)
            optimizer = optim.Adam(classifier.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # Train on TRAIN features
            train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

            for _ in range(10):
                for batch_feats, batch_labels in train_loader:
                    batch_feats = batch_feats.to(config.device)
                    batch_labels = batch_labels.to(config.device)

                    optimizer.zero_grad()
                    outputs = classifier(batch_feats)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

            # Evaluate on TEST features
            classifier.eval()
            with torch.no_grad():
                test_feats = test_features.to(config.device)
                test_lbls = test_labels.to(config.device)
                outputs = classifier(test_feats)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == test_lbls).float().mean().item()

        layer_accuracies[f'layer_{layer_idx}'] = accuracy * 100
        print(f"    Layer {layer_idx} Accuracy: {accuracy * 100:.2f}%")

    return layer_accuracies


def train(nets, device, optimizers, schedulers, threshold1, threshold2, dims_in, dims_out, epochs, pool,
          a, b, lamda, freezelayer, period, extra_pool, Layer_out, all_neurons, trainloader,
          valloader, testloader, suptrloader, pre_std, stdnorm_out, search, p,
          config, alleps, eval_frequency=10):
    """
    Modified training function with layer-wise accuracy tracking.

    Args:
        eval_frequency (int): Evaluate layer accuracies every N epochs
    """

    all_pos = []
    all_neg = []
    NL = len(nets)
    for i in range(NL):
        all_pos.append([])
        all_neg.append([])

    # Track layer accuracies over time
    accuracy_history = {f'layer_{i}': [] for i in range(NL)}
    epoch_numbers = []

    firstpass = True
    nbbatches = 0

    NBLEARNINGEPOCHS = epochs
    N_all = NBLEARNINGEPOCHS if epochs != 0 else NBLEARNINGEPOCHS + 1

    Dims = []
    best_acc = 0

    print(f"\n{'='*70}")
    print(f"Starting SCFF Training - CIFAR-10 Parallel")
    print(f"{'='*70}")
    print(f"Total Epochs: {epochs}")
    print(f"Number of Layers: {NL}")
    print(f"Evaluation Frequency: Every {eval_frequency} epochs")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    for epoch in range(N_all):
        epoch_start_time = time.time()

        print(f"\n{'─'*70}")
        print(f"Epoch {epoch+1}/{N_all}")
        print(f"{'─'*70}")

        if epoch < NBLEARNINGEPOCHS and epochs != 0:
            for i, net in enumerate(nets):
                net.train()

            UNLAB = True
            zeloader = trainloader
            mode = "Training"
        else:
            for net in nets:
                net.eval()
            if epoch == NBLEARNINGEPOCHS:
                UNLAB = False
                zeloader = testloader
                mode = "Final Evaluation"
            else:
                raise(ValueError("Wrong epoch!"))

        goodness_pos = 0
        goodness_neg = 0

        # Progress bar for batches
        pbar = tqdm(enumerate(zeloader), total=len(zeloader),
                   desc=f"{mode}",
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')

        for numbatch, (x, _) in pbar:
            nbbatches += 1
            x = x.to(device)

            for i in range(NL):
                if nets[i].concat:
                    x = stdnorm(x, dims=dims_in)
                    x, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)

                x = nets[i](x)
                x_neg = nets[i](x_neg)

                yforgrad = nets[i].relu(x).pow(2).mean([1])
                yforgrad_neg = nets[i].relu(x_neg).pow(2).mean([1])

                if i < freezelayer:
                    UNLAB = False
                else:
                    UNLAB = True

                if UNLAB and epoch < alleps[i]:
                    optimizers[i].zero_grad()
                    loss = torch.log(1 + torch.exp(
                        a * (-yforgrad + threshold1[i]))).mean([1,2]).mean(
                    ) + torch.log(1 + torch.exp(
                        b * (yforgrad_neg - threshold2[i]))).mean([1,2]).mean(
                    ) + lamda[i] * torch.norm(yforgrad, p=2, dim=(1,2)).mean()
                    loss.backward()
                    optimizers[i].step()

                    if (nbbatches + 1) % period[i] == 0:
                        schedulers[i].step()

                x = pool[i](nets[i].act(x)).detach()
                x_neg = pool[i](nets[i].act(x_neg)).detach()

                if firstpass:
                    _, channel, h, w = x.shape
                    Dims.append(channel * h * w)

            firstpass = False
            goodness_pos += (torch.mean(yforgrad.mean([1,2]))).item()
            goodness_neg += (torch.mean(yforgrad_neg.mean([1,2]))).item()

            # Update progress bar
            pbar.set_postfix({
                'G+': f'{goodness_pos/(numbatch+1):.3f}',
                'G-': f'{goodness_neg/(numbatch+1):.3f}'
            })

        avg_goodness_pos = goodness_pos / len(zeloader)
        avg_goodness_neg = goodness_neg / len(zeloader)

        if UNLAB:
            all_pos[i].append(goodness_pos)
            all_neg[i].append(goodness_neg)

        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Avg Goodness (Positive): {avg_goodness_pos:.4f}")
        print(f"  Avg Goodness (Negative): {avg_goodness_neg:.4f}")

        # Evaluate layer-wise accuracies periodically
        if epoch > 0 and (epoch + 1) % eval_frequency == 0:
            print(f"\n{'='*70}")
            print(f"Layer-wise Accuracy Evaluation at Epoch {epoch+1}")
            print(f"{'='*70}")

            layer_accs = evaluate_layer_accuracies(
                nets, pool, extra_pool, config, trainloader, testloader, NL
            )

            # Store results
            epoch_numbers.append(epoch + 1)
            for layer_name, acc in layer_accs.items():
                accuracy_history[layer_name].append(acc)

            print(f"\nCurrent Best Accuracies:")
            for layer_idx in range(NL):
                layer_name = f'layer_{layer_idx}'
                current_acc = layer_accs[layer_name]
                best_layer_acc = max(accuracy_history[layer_name]) if accuracy_history[layer_name] else 0
                print(f"  Layer {layer_idx}: {current_acc:.2f}% (Best: {best_layer_acc:.2f}%)")
            print(f"{'='*70}\n")

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}\n")

    # Save accuracy history
    history_data = {
        'epochs': epoch_numbers,
        'accuracies': accuracy_history,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_layers': NL,
        'total_epochs': epochs,
        'eval_frequency': eval_frequency
    }

    with open('layer_accuracies_history.json', 'w') as f:
        json.dump(history_data, f, indent=2)

    print("✓ Accuracy history saved to: layer_accuracies_history.json")

    return nets, all_pos, all_neg, Dims, accuracy_history


class CustomStepLR(StepLR):
    def __init__(self, optimizer, nb_epochs):
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5 for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


class EvaluationConfig:
    def __init__(self, device, dims, dims_in, dims_out, stdnorm_out, out_dropout, Layer_out, pre_std, all_neurons):
        self.device = device
        self.dims = dims
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.stdnorm_out = stdnorm_out
        self.out_dropout = out_dropout
        self.Layer_out = Layer_out
        self.all_neurons = all_neurons
        self.pre_std = pre_std


def calculate_output_length(dims, nets, extra_pool, Layer, all_neurons):
    lengths = 0
    if all_neurons:
        for i, length in enumerate(dims):
            if i in Layer:
                lengths += length
    else:
        for i, length in enumerate(dims):
            if i in Layer:
                len_after_pool = math.ceil((math.sqrt(length / nets[i].output_channels) - extra_pool[i].kernel_size) / extra_pool[i].stride + 1)
                lengths += len_after_pool * len_after_pool * nets[i].output_channels
    return lengths


def build_classifier(lengths, config):
    classifier = nn.Sequential(
        nn.Dropout(config.out_dropout),
        nn.Linear(lengths, 10)
    ).to(config.device)
    if torch.cuda.device_count() > 2:
        classifier = nn.DataParallel(classifier)
    return classifier


def train_readout(classifier, nets, pool, extra_pool, loader, criterion, optimizer, config, epoch):
    classifier.train()
    correct = 0
    total = 0

    for i, (x, labels) in enumerate(loader):
        x = x.to(config.device)
        labels = labels.to(config.device)
        outputs = []

        with torch.no_grad():
            for j, net in enumerate(nets):
                if net.concat:
                    x = stdnorm(x, dims=config.dims_in)
                    x = torch.cat((x, x), dim=1)
                x = pool[j](net.act(net(x)))
                if not config.all_neurons:
                    out = extra_pool[j](x)
                if config.stdnorm_out:
                    out = stdnorm(out, dims=config.dims_out)
                out = out.flatten(start_dim=1)
                if j in config.Layer_out:
                    outputs.append(out)

        outputs = torch.cat(outputs, dim=1)
        optimizer.zero_grad()
        outputs = classifier(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total


def test_readout(classifier, nets, pool, extra_pool, loader, criterion, config, epoch, mode):
    classifier.eval()
    running_loss = 0.
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (x, labels) in enumerate(loader):
            x = x.to(config.device)
            labels = labels.to(config.device)
            outputs = []

            for j, net in enumerate(nets):
                if net.concat:
                    x = stdnorm(x, dims=config.dims_in)
                    x = torch.cat((x, x), dim=1)
                x = pool[j](net.act(net(x)))
                if not config.all_neurons:
                    out = extra_pool[j](x)
                if config.stdnorm_out:
                    out = stdnorm(out, dims=config.dims_out)
                out = out.flatten(start_dim=1)
                if j in config.Layer_out:
                    outputs.append(out)

            outputs = torch.cat(outputs, dim=1)
            outputs = classifier(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    if mode == 'Val':
        print(f'Accuracy of the network on the 10000 {mode} images: {100 * correct / total:.2f}%')
        print(f'[{epoch + 1}] loss: {running_loss / total:.3f}')

    return correct / total


def evaluate_model(nets, pool, extra_pool, config, loaders, search, Dims):
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(42)

    lengths = calculate_output_length(Dims, nets, extra_pool, config.Layer_out, config.all_neurons)
    print(f"Classifier input length: {lengths}")
    classifier = build_classifier(lengths, config)

    _, valloader, testloader, suptrloader = loaders
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    lr_scheduler = CustomStepLR(optimizer, nb_epochs=50)
    criterion = nn.CrossEntropyLoss()

    if not search:
        valloader = testloader

    for j, net in enumerate(nets):
        net.eval()

    print("\nTraining final classifier...")
    for epoch in tqdm(range(50), desc="Classifier Training"):
        acc_train = train_readout(classifier, nets, pool, extra_pool, suptrloader, criterion, optimizer, config, epoch)
        lr_scheduler.step()

        if epoch % 20 == 0 or epoch == 49:
            print(f'  Epoch {epoch}: Train Accuracy: {100 * acc_train:.2f}%')
            acc_train = test_readout(classifier, nets, pool, extra_pool, suptrloader, criterion, config, epoch, 'Train')
            acc_val = test_readout(classifier, nets, pool, extra_pool, valloader, criterion, config, epoch, 'Val')

    torch.set_rng_state(current_rng_state)

    return acc_train, acc_val


def create_layer(layer_config, opt_config, load_params, device, act):
    layer_num = layer_config['num'] - 1
    net = Conv2d(layer_config["ch_in"], layer_config["channels"],
                 (layer_config["kernel_size"], layer_config["kernel_size"]),
                 pad=layer_config["pad"], norm="stdnorm",
                 padding_mode=layer_config["padding_mode"], act=act)

    if load_params:
        net.load_state_dict(torch.load('./results/params_CIFAR_l' + str(layer_num) + '.pth', map_location='cpu'))
        for param in net.parameters():
            param.requires_grad = False

    if layer_config["pooltype"] == 'Avg':
        pool = nn.AvgPool2d(kernel_size=layer_config["pool_size"], stride=layer_config["stride_size"],
                           padding=layer_config["padding"], ceil_mode=True)
    else:
        pool = nn.MaxPool2d(kernel_size=layer_config["pool_size"], stride=layer_config["stride_size"],
                           padding=layer_config["padding"], ceil_mode=True)

    extra_pool = nn.AvgPool2d(kernel_size=layer_config["extra_pool_size"], stride=layer_config["extra_pool_size"],
                             padding=0, ceil_mode=True)
    net.to(device)
    optimizer = AdamW(net.parameters(), lr=opt_config["lr"], weight_decay=opt_config["weight_decay"])
    scheduler = ExponentialLR(optimizer, opt_config["gamma"])

    return net, pool, extra_pool, optimizer, scheduler


def hypersearch(dims, dims_in, dims_out, Batchnorm, epochs, a, b, all_neurons, NL, Layer_out,
                pre_std, stdnorm_out, search, device_num, loaders, p, seed_num,
                lr, weight_decay, gamma, threshold1, threshold2, lamda, period, out_dropout,
                act, concats, alleps, eval_frequency):

    trainloader, valloader, testloader, suptrloader = loaders
    torch.manual_seed(seed_num)

    device = 'cuda:' + str(device_num) if torch.cuda.is_available() else 'cpu'
    nets = []
    optimizers = []
    schedulers = []
    pools = []
    extra_pools = []

    with open('config.json', 'r') as f:
        config_json = json.load(f)

    freezelayer = 0

    for i, (layer_config, opt_config) in enumerate(zip(config_json['CIFAR']['layer_configs'][:NL],
                                                        config_json['CIFAR']['opt_configs'][:NL])):
        load_params = False
        net, pool, extra_pool, _, _ = create_layer(layer_config, opt_config,
                                                    load_params=load_params, device=device, act=act[i])
        nets.append(net)
        pools.append(pool)
        extra_pools.append(extra_pool)
        optimizer = AdamW(net.parameters(), lr=lr[i], weight_decay=weight_decay[i])
        optimizers.append(optimizer)
        schedulers.append(ExponentialLR(optimizer, gamma[i]))

    for (net, concat) in zip(nets, concats):
        net.concat = concat

    config = EvaluationConfig(device=device, dims=dims, dims_in=dims_in, dims_out=dims_out,
                             stdnorm_out=stdnorm_out, out_dropout=out_dropout, Layer_out=Layer_out,
                             pre_std=pre_std, all_neurons=all_neurons)

    nets, all_pos, all_neg, Dims, accuracy_history = train(
        nets, device, optimizers, schedulers, threshold1, threshold2, dims_in, dims_out, epochs, pools,
        a, b, lamda, freezelayer, period, extra_pools, Layer_out, all_neurons, trainloader, valloader,
        testloader, suptrloader, pre_std, stdnorm_out, search, p, config, alleps, eval_frequency)

    print("\nFinal evaluation with combined classifier...")
    _, tacc = evaluate_model(nets, pools, extra_pools, config, loaders, search, Dims)

    return tacc, all_pos, all_neg, nets, accuracy_history


def main(device_num, save_model, loaders, NL, lr, weight_decay, gamma, lamda, threshold1, threshold2,
         act, concats, period, alleps, seed_num, eval_frequency):

    tacc, all_pos, all_neg, nets, accuracy_history = hypersearch(
        dims=(1, 2, 3),
        dims_in=(1, 2, 3),
        dims_out=(1, 2, 3),
        Batchnorm=False,
        epochs=max(alleps),
        a=1,
        b=1,
        all_neurons=False,
        NL=NL,
        Layer_out=[2, 1, 0],
        pre_std=True,
        stdnorm_out=True,
        search=False,
        device_num=device_num,
        loaders=loaders,
        p=1,
        seed_num=seed_num,
        lr=lr,
        weight_decay=weight_decay,
        gamma=gamma,
        threshold1=threshold1,
        threshold2=threshold2,
        lamda=lamda,
        period=period,
        out_dropout=0.2,
        act=act,
        concats=concats,
        alleps=alleps,
        eval_frequency=eval_frequency
    )

    if save_model:
        import os
        os.makedirs('./results', exist_ok=True)
        for i, net in enumerate(nets):
            torch.save(net.state_dict(), f'./results/params_CIFAR_parallel_l{i}.pth')
        print(f"\n✓ Models saved to ./results/")

    return tacc, accuracy_history


def get_arguments():
    parser = argparse.ArgumentParser(description="SCFF CIFAR-10 Parallel Training with Layer-wise Tracking", add_help=False)

    parser.add_argument("--lr", nargs='+', type=float, default=[0.02, 0.001, 0.0004], help="Learning rate per layer")
    parser.add_argument("--gamma", nargs='+', type=float, default=[0.99, 0.9, 0.99], help="LR decay rate per layer")
    parser.add_argument("--period", nargs='+', type=int, default=[500, 500, 500], help="LR decay period")
    parser.add_argument("--weight_decay", nargs='+', type=float, default=[0.0001, 0.0003, 0.0001], help="Weight decay")
    parser.add_argument("--lamda", nargs='+', type=float, default=[0.0008, 0.0004, 0.0016], help="Regularization lambda")

    parser.add_argument("--th1", nargs='+', type=int, default=[1, 4, 5], help="Positive sample thresholds")
    parser.add_argument("--th2", nargs='+', type=int, default=[2, 5, 7], help="Negative sample thresholds")

    parser.add_argument("--NL", type=int, default=3, help="Number of layers")
    parser.add_argument("--concats", type=tuple, default=(1, 0, 1), help="Concatenation setting per layer")
    parser.add_argument("--act", nargs='+', type=str, default=["triangle", "triangle", "relu"], help="Activation per layer")
    parser.add_argument("--alleps", nargs='+', type=int, default=[100, 100, 100], help="Epochs per layer")

    parser.add_argument("--device_num", type=int, default=0, help="GPU device")
    parser.add_argument("--seed_num", type=int, default=1234, help="Random seed")
    parser.add_argument("--eval_frequency", type=int, default=10, help="Evaluate every N epochs")

    parser.add_argument("--save_model", action="store_true", help="Save trained model")

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SCFF Research Script', parents=[get_arguments()])
    args = parser.parse_args()

    print("\n" + "="*70)
    print("SCFF CIFAR-10 Parallel Training - Layer-wise Accuracy Research")
    print("="*70)
    print("\nConfiguration:")
    for arg in vars(args):
        print(f"  {arg:20s} = {getattr(args, arg)}")
    print("="*70 + "\n")

    # Load dataset
    print("Loading CIFAR-10 dataset...")
    loaders = get_train(batchsize=100, augment="no", Factor=1)
    print("✓ Dataset loaded\n")

    # Run training
    tacc, accuracy_history = main(
        device_num=args.device_num,
        save_model=args.save_model,
        loaders=loaders,
        NL=args.NL,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        lamda=args.lamda,
        threshold1=args.th1,
        threshold2=args.th2,
        act=args.act,
        concats=args.concats,
        period=args.period,
        alleps=args.alleps,
        seed_num=args.seed_num,
        eval_frequency=args.eval_frequency
    )

    print(f"\n{'='*70}")
    print(f"Final Test Accuracy (Combined Classifier): {tacc * 100:.2f}%")
    print(f"{'='*70}\n")
