import torch
from torchvision import datasets, transforms
import models
import torch.nn.functional as F

import math
from utils import *
import os, random
import numpy as np
import argparse
from tqdm import trange, tqdm
import shutil

savepath = '.'

parser = argparse.ArgumentParser(description='Training code - ATC')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train transfer learning')

parser.add_argument('--source', default='dslr', help='dslr | webcam | amazon')
parser.add_argument('--target', default='webcam', help='dslr | webcam | amazon')

parser.add_argument('--r', type=float, default=0, help='samples rate')
parser.add_argument('--n', type=int, default=10, help='Training set category')
parser.add_argument('--m', type=int, default=20, help='Test set category')
args = parser.parse_args()


train_transform = transforms.Compose(
            [transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

if not os.path.exists(os.path.join(savepath, 'results')):  
    os.makedirs(os.path.join(savepath, 'results'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



kwargs = {'num_workers': 1, 'pin_memory': True}
class_name_alllist = os.listdir(os.path.join('Office31', args.source, 'images'))
class_name_alllist.sort()

if os.path.isdir('Ablation'):
    shutil.rmtree('Ablation')

os.makedirs(os.path.join('Ablation', args.source))
os.makedirs(os.path.join('Ablation', args.source, 'images'))
os.makedirs(os.path.join('Ablation', args.target))
os.makedirs(os.path.join('Ablation', args.target, 'images'))

for clsname in class_name_alllist[:args.n]:
    shutil.copytree(os.path.join('Office31', args.source, 'images', clsname), os.path.join('Ablation', args.source, 'images', clsname))

for clsname in (class_name_alllist[:10]+class_name_alllist[21:21+args.m-10]):
    shutil.copytree(os.path.join('Office31', args.target, 'images', clsname), os.path.join('Ablation', args.target, 'images', clsname))


dataset_s = datasets.ImageFolder(root=os.path.join('Ablation', args.source, 'images'))
dataset_t = datasets.ImageFolder(root=os.path.join('Ablation', args.target, 'images'))

dataset_s_samples = np.array(dataset_s.samples)
dataset_t_samples = np.array(dataset_t.samples)

source_dataset = CustomImageDataset(dataset_s_samples, train_transform)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)

target_dataset = CustomImageDataset(dataset_t_samples, train_transform)
target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size)

samples_total = len(target_loader.dataset)

def get_per_class(m,s,n=0,t='uniform'):
    temp = m - n
    templist = [0]*n
    if t == 'uniform':
        u, r = s // temp, s % temp
        templist += [u] * temp
        for _ in range(r):
            c = random.randint(n,m-1)
            templist[c] = templist[c] + 1
    elif t == 'random':
        templist = [0] * m
        for _ in range(s):
            c = random.randint(n,m-1)
            templist[c] = templist[c] + 1
    else:
        raise 't is uniform or random'

    return templist


def nex_samples(co, list_per, raw_samples):
    samples_use = None
    for c, cn in zip(co, list_per):
        conditionS = np.where(raw_samples[:,1] == c)
        a = conditionS[0] 
        np.random.shuffle(a) 
        try:
            if samples_use == None:
                samples_use = raw_samples[a][:cn]
            else:
                samples_use = np.concatenate((samples_use, raw_samples[a][:cn]))
        except:
            samples_use = np.concatenate((samples_use, raw_samples[a][:cn]))
    return samples_use

def DSAN_train(epoch, model, source_loader, target_train_loader, optimizer):
    model.train()
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        if data_source.shape[0] < args.batch_size:
            iter_source = iter(source_loader)
            data_source, label_source = iter_source.next()

        data_target, _ = iter_target.next()
        if data_target.shape[0] < args.batch_size:
            iter_target = iter(target_train_loader)
            data_target, _ = iter_target.next()

        data_source, label_source = data_source.to(device), label_source.to(device)
        data_target = data_target.to(device)

        optimizer.zero_grad()
        label_source_pred, loss_lmmd = model(
            data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        loss = loss_cls + 0.5 * lambd * loss_lmmd

        loss.backward()
        optimizer.step()

    # print(f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')

def DSAN_test(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            _, pred = model.predict(data)
            # sum up batch loss
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    return correct, correct / len(dataloader.dataset)



if __name__ == '__main__':
    # list_per = get_per_class(args.m, int(samples_total*args.r), t='random')

    # co = list(set(dataset_t_samples[:,1]))
    # co.sort()
    # assert len(co) == len(list_per)
    # samples_use = nex_samples(co, list_per, dataset_t_samples)

    # print(list_per, '\n', samples_use)




    # 随机采样 + 域适应 No Stage I/II

    a = dataset_t_samples
    np.random.shuffle(a) 
    samples_use = a[:int(len(target_dataset) * args.r)]



    dataset_NEXs_samples = np.concatenate((dataset_s_samples, samples_use))
    NEXsource_dataset = CustomImageDataset(dataset_NEXs_samples, train_transform)
    NEXsource_loader = torch.utils.data.DataLoader(NEXsource_dataset, batch_size=args.batch_size, shuffle=True)
    source_loader = NEXsource_loader

    model = models.LoopNet_DANN(args.m).to(device)
    model.train()

    optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': 0.01},
        ], lr=0.001, momentum=0.9, weight_decay=5e-4)
    model = model.to(device)

    correct = 0
    seen_correct = 0
    with trange(1, args.nepoch + 1) as pbar:
    # for epoch in range(1, args.nepoch + 1):
        for epoch in pbar:
            pbar.set_description(f"Transfer learning train {epoch}")

            optimizer.param_groups[0]['lr'] = 0.001 / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)
            optimizer.param_groups[1]['lr'] = 0.01 / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)
            DSAN_train(epoch, model, source_loader, target_loader, optimizer)

            t_correct, AllAcy = DSAN_test(model, target_loader)
            if t_correct >= correct:
                correct = t_correct

            seen_correct = max(seen_correct, AllAcy)

            pbar.set_postfix(OSF=100. * correct / len(target_dataset))


