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
from random import shuffle

savepath = '.'

parser = argparse.ArgumentParser(description='Training code - ATC')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train transfer learning')

parser.add_argument('--source', default='dslr', help='dslr | webcam | amazon')
parser.add_argument('--target', default='webcam', help='dslr | webcam | amazon')

parser.add_argument('--r', type=float, default=0, help='samples rate')
parser.add_argument('--n', type=int, default=20, help='Training set category')
parser.add_argument('--m', type=int, default=20, help='Test set category')
args = parser.parse_args()

class_name_alllist = os.listdir(os.path.join('Office31', args.source, 'images'))
class_name_alllist.sort()

if os.path.isdir('Semisup'):
    shutil.rmtree('Semisup')

os.makedirs(os.path.join('Semisup', args.source))
os.makedirs(os.path.join('Semisup', args.source, 'images'))
os.makedirs(os.path.join('Semisup', args.source, 'images', 'unkown'))
os.makedirs(os.path.join('Semisup', args.target))
os.makedirs(os.path.join('Semisup', args.target, 'images'))
os.makedirs(os.path.join('Semisup', args.target, 'images', 'unkown'))

for clsname in class_name_alllist[:10]:
    shutil.copytree(os.path.join('Office31', args.source, 'images', clsname), os.path.join('Semisup', args.source, 'images', clsname))
for clsname in class_name_alllist[11:11+args.n-10]:
    for imgname in os.listdir(os.path.join('Office31', args.source, 'images', clsname)):
        shutil.copy(os.path.join('Office31', args.source, 'images', clsname, imgname), 
                               os.path.join('Semisup', args.source, 'images', 'unkown', clsname+imgname))


for clsname in class_name_alllist[:10]:
    shutil.copytree(os.path.join('Office31', args.target, 'images', clsname), os.path.join('Semisup', args.target, 'images', clsname))
for clsname in class_name_alllist[21:21+args.m-10]:
    for imgname in os.listdir(os.path.join('Office31', args.target, 'images', clsname)):
        shutil.copy(os.path.join('Office31', args.target, 'images', clsname, imgname), 
                               os.path.join('Semisup', args.target, 'images', 'unkown', clsname+imgname))


# semiSup
ns = 3
for clsname in class_name_alllist[21:21+args.m-10]:
    list_data = os.listdir(os.path.join('Office31', args.target, 'images', clsname))
    shuffle(list_data)
    for imgname in list_data[:ns]:
        shutil.copy(os.path.join('Office31', args.target, 'images', clsname, imgname), 
                               os.path.join('Semisup', args.source, 'images', 'unkown', 'ns_'+clsname+imgname))

for clsname in class_name_alllist[:10]:
    list_data = os.listdir(os.path.join('Office31', args.target, 'images', clsname))
    shuffle(list_data)
    for imgname in list_data[:ns]:
        shutil.copy(os.path.join('Office31', args.target, 'images', clsname, imgname), 
                               os.path.join('Semisup', args.source, 'images', clsname, 'ns_'+clsname+imgname))



train_transform = transforms.Compose(
            [transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

if not os.path.exists(os.path.join(savepath, 'results')):  
    os.makedirs(os.path.join(savepath, 'results'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_s = datasets.ImageFolder(root=os.path.join('Semisup', args.source, 'images'))
dataset_s_samples = np.array(dataset_s.samples)
source_dataset = CustomImageDataset(dataset_s_samples, train_transform)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)

dataset_t = datasets.ImageFolder(root=os.path.join('Semisup', args.target, 'images'))
dataset_t_samples = np.array(dataset_t.samples)
target_dataset = CustomImageDataset(dataset_t_samples, train_transform)
target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True)


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


def DSAN_test(model, dataloader):
    model.eval()

    correct = 0

    share_total = 0
    share_correct = 0

    unkown_total = 0
    unkown_correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            _, pred = model.predict(data)
            # sum up batch loss
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            for i,j in zip(target, pred):
                if i in [*range(10)]:
                    share_total += 1
                    if i==j:
                        share_correct += 1
                else:
                    unkown_total += 1
                    if i==j:
                        unkown_correct += 1

    return  correct / len(dataloader.dataset), \
            share_correct / share_total, unkown_correct / unkown_total



if __name__ == '__main__':
    model = models.LoopNet_DANN(11).to(device)
    model.train()

    optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            # {'params': model.bottle.parameters(), 'lr': 0.01},
            {'params': model.cls_fc.parameters(), 'lr': 0.01},
        ], lr=0.001, momentum=0.9, weight_decay=5e-4)
    model = model.to(device)

    max_totalAcy = 0
    max_shareAcy = 0
    max_unknoweAcy = 0


    with trange(1, args.nepoch + 1) as pbar:
    # for epoch in range(1, args.nepoch + 1):
        for epoch in pbar:
            pbar.set_description(f"Transfer learning train {epoch}")

            optimizer.param_groups[0]['lr'] = 0.001 / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)
            optimizer.param_groups[1]['lr'] = 0.01 / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)
            # optimizer.param_groups[2]['lr'] = 0.01 / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)
            DSAN_train(epoch, model, source_loader, target_loader, optimizer)

            total_acy, share_acy, unknow_acy = DSAN_test(model, target_loader)


            max_totalAcy = max(max_totalAcy, total_acy)
            max_shareAcy = max(max_shareAcy, share_acy)
            max_unknoweAcy = max(max_unknoweAcy, unknow_acy)

            pbar.set_postfix(TotalAccuracy=100. * total_acy, ShareAccuracy=100. * share_acy, UnknowAccuracy=100. * unknow_acy)

    print(f'\n max total acy: {max_totalAcy}, max share acy: {max_shareAcy},max unkown acy: {max_unknoweAcy}')



