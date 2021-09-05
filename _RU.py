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
from time import sleep

savepath = '.'

parser = argparse.ArgumentParser(description='Training code - ATC')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train transfer learning')

parser.add_argument('--source', default='dslr', help='dslr | webcam | amazon')
parser.add_argument('--target', default='webcam', help='dslr | webcam | amazon')

parser.add_argument('--r', type=float, default=0, help='samples rate')
parser.add_argument('--n', type=int, default=3, help='Training set category')
parser.add_argument('--m', type=int, default=10, help='Test set category')
args = parser.parse_args()


train_transform = transforms.Compose(
            [transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

if not os.path.exists(os.path.join(savepath, 'results')):  
    os.makedirs(os.path.join(savepath, 'results'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_s = datasets.ImageFolder(root=os.path.join('Office31', args.source, 'images'))
dataset_s_n_index = [_ in [*range(args.n)] for _ in dataset_s.targets]
dataset_s_samples = np.array(dataset_s.samples)[dataset_s_n_index]
source_dataset = CustomImageDataset(dataset_s_samples, train_transform)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)


dataset_t = datasets.ImageFolder(root=os.path.join('Office31', args.target, 'images'))
dataset_t_n_index = [_ in [*range(args.m)] for _ in dataset_t.targets]
dataset_t_samples = np.array(dataset_t.samples)[dataset_t_n_index]
samples_total = sum(dataset_t_n_index)
target_dataset = CustomImageDataset(dataset_t_samples, train_transform)
target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size)

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
        if i % len(target_train_loader) == 0:
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


def fineTrain(model, source_loader, epochnum = 10):
    unc_optimizer = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)
    bestloss = 1e10 
    # for epoch in trange(1, epochnum+1, desc="FineTrain: "):
    with trange(1, epochnum+1) as pbar:
        for epoch in pbar:
            sleep(0.5)
            pbar.set_description(f"FineTrain {epoch}")

            model.train()
            sumloss = 0
            for i, data in enumerate(source_loader):
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(device), labels.to(device)
                unc_optimizer.zero_grad()
                annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32), torch.tensor(epoch / epochnum, dtype=torch.float32))
                output, klloss = model.uncforward(inputs, labels, annealing_coef)
                loss_cls = F.nll_loss(F.log_softmax(
                output, dim=1), labels)
                loss = klloss + 0.1 * loss_cls

                sumloss = sumloss + loss
                loss.backward()
                unc_optimizer.step()
            loss_ = sumloss/len(source_loader)

            if loss_ <= bestloss:
                bestloss = loss_
                torch.save(model.state_dict(), 'fine_weights.pth')
            pbar.set_postfix(loss=loss_.item(), minloss=bestloss.item())

    model.load_state_dict(torch.load('fine_weights.pth'))
    print('Finished Training')
    return model

def calc_u(model, dataloader):
    features_array = None
    u_list = []
    for batch_ndx, sample in enumerate(dataloader):    
        with torch.no_grad(): 
            data = sample[0].to(device)
            f, _ = model.predict(data)
            evidence = F.relu(_)
            alpha = evidence + 1
            uncertainty = args.n / torch.sum(alpha, dim=1, keepdim=True)
            f = f.cpu().detach().numpy()    
            u_list += (uncertainty.cpu().detach().numpy()[:,0].tolist())
            try: 
                if features_array == None:
                    features_array = f
            except:
                features_array = np.concatenate((features_array, f), axis=0)
    return u_list, features_array


if __name__ == '__main__':
    # Uncertainty
    model = models.LoopNet(args.m).to(device)
    model = fineTrain(model, source_loader, 50)

    source_uncertainty_list, _ = calc_u(model, source_loader)
    target_uncertainty_list, _ = calc_u(model, target_loader)

    mean_u = np.mean(source_uncertainty_list)

    # out of distribution
    a = np.arange(sum(np.array(target_uncertainty_list) >= mean_u))
    np.random.shuffle(a) 
    samples_use = a[:int(len(target_dataset) * args.r)]
    ChooseSlist = dataset_t_samples[np.array(target_uncertainty_list) >= mean_u][samples_use]

    S_label = np.array(target_dataset.label)[np.array(target_uncertainty_list) >= mean_u]
    print(set(S_label))


    dataset_NEXs_samples = np.concatenate((dataset_s_samples, ChooseSlist))
    NEXsource_dataset = CustomImageDataset(dataset_NEXs_samples, train_transform)
    NEXsource_loader = torch.utils.data.DataLoader(NEXsource_dataset, batch_size=args.batch_size, shuffle=True)
    source_loader = NEXsource_loader
    model.train()


    optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': 0.01},
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
            optimizer.param_groups[2]['lr'] = 0.01 / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)
            DSAN_train(epoch, model, source_loader, target_loader, optimizer)

            t_correct, AllAcy = DSAN_test(model, target_loader)
            if t_correct >= correct:
                correct = t_correct

            seen_correct = max(seen_correct, AllAcy)

            pbar.set_postfix(AllAccuracy=100. * correct / len(target_dataset))

