from ntpath import join
import random
import time
import warnings
import sys
import argparse
import shutil
import os, logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy, ImageClassifier
from dalib.modules.kernels import GaussianKernel
from utils.data import ForeverDataIterator
from utils.metric import accuracy
from utils.meter import AverageMeter, ProgressMeter
from utils.logger import CompleteLogger
from utils.analysis import collect_feature, tsne, a_distance
from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
import tsne

import ops
import _utils
import numpy as np
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_source_iter, train_target_iter, model, mkmmd_loss, optimizer, lr_scheduler, epoch, args):
    model.train()
    try:
        mkmmd_loss.train()
    except:
        pass

    iter_source = iter(train_source_iter)
    iter_target = iter(train_target_iter)
    num_iter = len(train_source_iter)
    for i in range(num_iter):
        x_s, labels_s = iter_source.next()
        x_t, labels_t = iter_target.next()
        if x_s.size(0) < args.batch_size:
            iter_source = iter(train_source_iter)
            x_s, labels_s = iter_source.next()
        if x_t.size(0) < args.batch_size:
            iter_target = iter(train_target_iter)
            x_t, labels_t = iter_target.next()

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # compute output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)

        cls_loss = F.cross_entropy(y_s, labels_s)
        try:
            transfer_loss = mkmmd_loss(f_s, f_t)
            loss = cls_loss + transfer_loss * args.trade_off
        except:
            loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        try:
            lr_scheduler.step()
        except:
            pass

        if i % args.print_freq == 0:
            args.logger.info(f'Epoch: [{epoch}][{i}/{num_iter}]\t, cls_acc: {cls_acc}%\t, tgt_acc: {tgt_acc}%')


def validate(val_loader, model, args):
    all_target = []
    all_pred = []
    # 针对目标域的结果估计
    # OS*   : 已知类别中识别正确的数据/已知类别中所有数据                       -  Na/nk
    # OS    : 已知类别中识别正确的数据+未知类别识别正确为unkown数据/所有数据      -  (Na+Mu)/(nk+nu)
    # OSF   : 已知类别中识别正确的数据+未知类别识别正确的数据/所有数据            -  (Na+Ma)/(nk+nu)
    # Na\Ma\Mu -- nk\nu
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            
            # compute output
            output = model(images)
            pred = output.data.max(1)[1]

            all_target += (target.cpu().detach().numpy().tolist())
            all_pred += (pred.cpu().detach().numpy().tolist())

        nk_index = list(map(lambda x: x in args.raw_know, all_target))
        nk = np.sum(nk_index)
        Na = np.sum(np.array(all_target)[nk_index] == np.array(all_pred)[nk_index])

        nu_index = list(map(lambda x: x not in args.raw_know, all_target))
        nu = np.sum(nu_index)
        Ma = np.sum(np.array(all_target)[nu_index] == np.array(all_pred)[nu_index])

        Mu = np.sum(list(map(lambda x: x not in args.raw_know, np.array(all_pred)[nu_index])))
        
        OSp = Na/nk
        OS = (Na+Mu)/(nk+nu)
        OSF = (Na+Ma)/(nk+nu)
        args.logger.info(f'[IT] OS\':{Na}/{nk}({100*OSp:.2f}%), OS:{Na+Mu}/{nk+nu}({100*OS:.2f}%), OSF:{Na+Ma}/{nk+nu}({100*OSF:.2f}%)')

        return OSp, OS, OSF

def main(args):
    source_dataset = ImageFolder(os.path.join(args.root, args.source), transform=ops.get_train_transform())
    target_dataset = ImageFolder(os.path.join(args.root, args.target), transform=ops.get_train_transform())
    # val_loader = ImageFolder(os.path.join(args.root, args.target), transform=ops.get_val_transform())

    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # train_source_iter = ForeverDataIterator(source_loader)
    # train_target_iter = ForeverDataIterator(target_loader)

    backbone = _utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, args.num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)


    # DAN
    optimizer = SGD(classifier.get_parameters(), 0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  0.01 * (1. + 0.001 * float(x)) ** (-0.75))
    if args.method == 'DAN':
        mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            linear=False
        )
        dis_loss = mkmmd_loss
    elif args.method == 'DDC':
        mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel()],
            linear=False
        )
        dis_loss = mkmmd_loss
    elif args.method == 'ResNet':
        dis_loss = None
    elif args.method == 'DANN':
        domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
        optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    0.01, momentum=0.9, weight_decay=1e-3, nesterov=True)
        lr_scheduler = LambdaLR(optimizer, lambda x:  0.01 * (1. + 0.001 * float(x)) ** (-0.75))
        domain_adv = DomainAdversarialLoss(domain_discri).to(device)
        dis_loss = domain_adv
    else:
        raise NotImplementedError
    
    stop = 0
    max_OSF = 0
    for epoch in range(args.epochs):
        if args.method == 'DAN' or args.method == 'DDC' or args.method == 'DANN' or args.method == 'ResNet':
            train(source_loader, target_loader, classifier, dis_loss, optimizer, lr_scheduler, epoch, args)
        else:
            raise NotImplementedError
        
        OSp, OS, OSF = validate(target_loader, classifier, args)
        if OSF > max_OSF:
            max_OSF = OSF
            stop = 0
            temp = [OSp, OS, OSF]
        stop += 1

        if stop >= args.patience:
            break

    args.logger.info(f'[IT] Max OS\': {100*temp[0]:.2f}%, OS: {100*temp[1]:.2f}%, OSF:{100*temp[2]:.2f}%')

    #
    # # extract features from both domains
    # feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
    # source_feature, source_label = tsne.collect_feature(source_loader, feature_extractor)
    # target_feature, target_label = tsne.collect_feature(target_loader, feature_extractor)
    # # plot t-SNE
    # name = os.path.join(args.output_dir, f'tsne_with')
    #
    # tsne.visualize_swich(args, source_feature, target_feature, source_label, target_label, 'normal', name+'_normal.svg')
    # tsne.visualize_swich(args, source_feature, target_feature, source_label, target_label, 'unkonw', name+'_unkonw.svg')
    # tsne.visualize_swich(args, source_feature, target_feature, source_label, target_label, 'label', name+'_label.svg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='smisupervised Domain Adaptation')
    # dataset parameters    
    parser.add_argument('--root', type=str, default='/data/smisup/')
    parser.add_argument('--dataset', type=str, default='Office31smiWD')
    parser.add_argument('--raw_know', type=list, default=[*range(10)], help='raw know')
    parser.add_argument('--raw_unknow', type=list, default=[*range(21, 31)], help='raw unknow')

    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--source', type=str, default='webcam')
    parser.add_argument('--target', type=str, default='dslr')
    parser.add_argument('--num-classes', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--method', type=str, default='DAN')
    parser.add_argument('--bottleneck-dim', default=256, type=int, help='Dimension of bottleneck')
    parser.add_argument('--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--trade-off', default=0.5, type=float)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--iters-per_epoch', type=int, default=500)
    parser.add_argument('--print-freq', default=10, type=int)

    args = parser.parse_args()
    args.root = os.path.join(args.root, args.dataset)
    args.output_dir = f'{args.dataset}-{args.source}-{args.target}-{args.method}'
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # 使用FileHandler输出到文件
    fh = logging.FileHandler(os.path.join(args.output_dir, f'{args.dataset}-{args.source}-{args.target}.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # 给logger添加上面两个Handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    args.logger = logger
    args.logger.info(vars(args))

    main(args)

