'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv

from models import *

#from utils import progress_bar

import numpy as np
import random
import time


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--arch', default='resnext', type=str, help='model name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='resnext', type=str, help='session name')
parser.add_argument('--seed', default=123, type=int, help='random seed')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default=5e-4)')
parser.add_argument('--batchsize', default=128, type=int, help='batch size')
parser.add_argument('--n_epoch', default=300, type=int, help='total number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='base learning rate (default=0.1)')



parser.add_argument('--num_parallel', default=8, type=int, help='# of parallel branches')
parser.add_argument('--alpha_scale', default=0.3, type=float, help='scaling coefficient of STAM aggregation, sum(alpha_{i})*sqrt(C)')
parser.add_argument('--no_uniform', action='store_true', help='use non-uniform alpha')


parser.add_argument('--c100', action='store_true', help='use cifar100')



args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = args.batchsize


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if(args.c100):
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
else:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Model
if(args.resume):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint_file = './checkpoint/' + args.sess + str(args.seed) + '.ckpt'
    checkpoint = torch.load(checkpoint_file)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
    print('resume succeed')
    resume=True
else:
    print('resume failed')
    resume=False

    hyper_params = {}
    hyper_params['alpha_scale'] = args.alpha_scale
    hyper_params['no_uniform'] = args.no_uniform
    hyper_params['num_parallel'] = args.num_parallel

    num_class = 10
    if(args.c100):
        num_class = 100

    
    print("=> creating ResNext29, %dx64"%(args.num_parallel))
    net = eval('resnext(hyper_params, num_class)')



result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder  + args.sess + '_seed' + str(args.seed) + '.csv'

if use_cuda and not resume:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')
    
num_p = 0
for p in net.named_parameters():
    num_p += p[1].numel()

print('# of parameters: %.2f'%(num_p/(1000**2)), 'M')


nesterov = True


loss_func = nn.CrossEntropyLoss()

optimizer = optim.SGD(
        net.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=args.weight_decay, nesterov=nesterov)




# Training
def train(epoch):

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    time0= time.time()

    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        
        outputs = net(inputs)
        optimizer.zero_grad()
        loss = loss_func(outputs, targets)# * 0.
        loss.backward()

        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).float().cpu().sum()
        acc = 100.*float(correct)/float(total)
    time1=time.time()
 

    print('Epoch %d, trn loss %.3f, trn acc %.3f, trn time %.f s'%(epoch, train_loss/(batch_idx+1), acc, time1-time0), end=" ")
    return (train_loss/batch_idx, acc)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print(', test loss %.3f, tst acc %.3f'%(test_loss/(batch_idx+1), 100.*float(correct)/float(total)))
        # Save checkpoint.
        acc = 100.*float(correct)/float(total)
        if acc > best_acc:
            best_acc = acc
            checkpoint(acc, epoch)

    return (test_loss/batch_idx, acc)

def checkpoint(acc, epoch):
    # Save checkpoint.
    #print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + args.sess + str(args.seed) + '.ckpt')

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    if(epoch<150):
        decay = 1.
    elif(epoch<225):
        decay = 10.
    else:
        decay = 100.
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr / decay

    return args.lr / decay

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'lr', 'train loss', 'train acc', 'test loss', 'test acc'])


for epoch in range(start_epoch, args.n_epoch):
    lr = adjust_learning_rate(optimizer, epoch)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, lr, train_loss, train_acc, test_loss, test_acc])
