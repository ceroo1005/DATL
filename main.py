import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import models
import logging

import math
from tqdm import trange, tqdm
from clusters import *
from utils import *
from time import sleep
import shutil
from collections import Counter

import argparse
parser = argparse.ArgumentParser(description='Training code - ATC - office home')
# parser.add_argument('--root_path', default='/Volumes/Jiang/DA_dataset/')
parser.add_argument('--root_path', type=str, default='/data', help='dataset root')
parser.add_argument('--dataset', type=str, default='office31', help='dataset name')
parser.add_argument('--source', default='amazon', help='amazon, dslr, webcam')
parser.add_argument('--target', default='dslr', help='amazon, dslr, webcam')
parser.add_argument('--c_n', default=4, type=int, help='number of cluster classes')
parser.add_argument('--s_n', default=5, type=int, help='number of samples per unknown cluster ')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=10, help='patience')
args = parser.parse_args()

args.output_dir = f'{args.dataset}-{args.source}-{args.target}'
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

savepath = '.'
n = 10
m = 10
c_n = args.c_n
s_n = args.s_n
m_n = 4
batch_size = args.batch_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose(
            [transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

test_transform = transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ToTensor()])

class_name_alllist = os.listdir(os.path.join(args.root_path, 'office31', args.source, 'images'))
class_name_alllist.sort()

SemisupPath = os.path.join(args.root_path, f'Semisup-{args.source}-{args.target}')
try:
    shutil.rmtree(SemisupPath)
except:
    pass
os.makedirs(os.path.join(SemisupPath, args.source))
os.makedirs(os.path.join(SemisupPath, args.target))

for clsname in class_name_alllist[:n]:
  shutil.copytree(os.path.join(args.root_path, 'office31', args.source, 'images', clsname), os.path.join(SemisupPath, args.source, clsname))

for clsname in (class_name_alllist[:n]+class_name_alllist[-m:]):
  shutil.copytree(os.path.join(args.root_path, 'office31', args.target, 'images', clsname), os.path.join(SemisupPath, args.target, clsname))

class ATC:
    def __init__(self) -> None:
        self.model = models.LoopNet_DANN(n).to(device)
        self.load_data()
        self.nl, self.ul = [*range(n)], []
        self.Sub, self.Ssd = {}, {}
        # [类别数量、抽样比例、已有类别精度、所有类别精度]
        self.progress = []

    def load_data(self):
        self.dataset_s = datasets.ImageFolder(root=os.path.join(SemisupPath, args.source), transform=train_transform)
        self.dataset_t = datasets.ImageFolder(root=os.path.join(SemisupPath, args.target), transform=train_transform)
        # dataset_t_test = datasets.ImageFolder(root=os.path.join('Semisup', args.target))

        # ImageFolder 继承 DatasetFolder， DatasetFolder 带 samples 变量
        self.dataset_s_samples = np.array(self.dataset_s.samples)
        self.dataset_t_samples = np.array(self.dataset_t.samples)

        self.source_dataset = self.dataset_s
        self.source_dataset.label = np.array([int(target) for target in self.dataset_s_samples[:,1]])
        self.source_dataset.label = self.source_dataset.label - min(self.source_dataset.label)
        self.source_loader = torch.utils.data.DataLoader(self.source_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.target_dataset = self.dataset_t
        self.target_dataset.label = np.array([int(target) for target in self.dataset_t_samples[:,1]])
        self.target_dataset.label = self.target_dataset.label - min(self.target_dataset.label)
        self.target_dataset.id = np.array([str(_) for _ in range(len(self.target_dataset))])
        self.target_loader = torch.utils.data.DataLoader(self.target_dataset, batch_size=batch_size)
        self.target_loader_disorder = torch.utils.data.DataLoader(self.target_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        args.logger.info(f'Number: {len(self.source_dataset)}\t Source Domain Category: {set(self.source_dataset.label)}')
        args.logger.info(f'Number: {len(self.target_dataset)}\t Target Domain Category: {set(self.target_dataset.label)}')
        # args.logger.info('self.source_dataset.samples:',self.source_dataset.samples)
    def UpdateModel(self):
        if self.model.num_classes != len(self.nl):
            switch_class = len(self.nl)
            
            self.model.num_classes = len(self.nl)
            # self.model.lmmd_loss = models.LMMD_loss(class_num=len(self.nl))
            self.model.edl_loss = models.EDL_loss(num_classes=len(self.nl))


            # self.model.cls_fc = nn.Linear(2048, len(self.nl))
            current_num_classes = self.model.cls_fc.weight.data.size(0)
            current_weight = self.model.cls_fc.weight.data
            current_bias = self.model.cls_fc.bias.data
            
            self.model.cls_fc = nn.Linear(2048, switch_class)
            hl_input = torch.zeros([switch_class - current_num_classes, 2048]).cuda()
            nn.init.xavier_uniform_(hl_input, gain=nn.init.calculate_gain('relu'))
            self.model.cls_fc.weight.data = torch.cat([current_weight, hl_input], dim=0)
            self.model.cls_fc.bias.data = torch.cat([current_bias, torch.zeros(switch_class - current_num_classes).cuda()])

    def fineTrain(self, epochnum = 10):
        self.model.train()
        self.unc_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0003, momentum=0.9)

        stop_iter = 0
        min_loss = 1e10
        with trange(1, epochnum+1) as pbar:
            for epoch in pbar:
                sleep(0.5)
                pbar.set_description(f"FineTrain {epoch}")                
                sumloss = 0
                for i, data in enumerate(self.source_loader):
                    inputs, labels = data[0], data[1]
                    labels = torch.tensor(list(map(lambda x:(torch.tensor(self.nl) == x).nonzero(as_tuple=True)[0], labels)))
                    inputs, labels = inputs.to(device), labels.to(device)
                    self.unc_optimizer.zero_grad()
                    annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32), torch.tensor(epoch / epochnum, dtype=torch.float32))
                    output, klloss = self.model.uncforward(inputs, labels, annealing_coef)
                    loss_cls = F.nll_loss(F.log_softmax(output, dim=1), labels)
                    loss = klloss + 0.1 * loss_cls
                    sumloss = sumloss + loss
                    loss.backward()
                    self.unc_optimizer.step()

                stop_iter += 1
                if sumloss < min_loss:
                    min_loss = sumloss
                    stop_iter = 0
                    torch.save(self.model.state_dict(), 'fine_weights.pth')
                if stop_iter > args.patience:
                    break

                pbar.set_postfix(loss=loss.item())

        try:
            self.model.load_state_dict(torch.load('fine_weights.pth'))    
        except:
            pass
        print('Finished Training')
    
    def calc_u(self, dataloader):
        features_array = None
        u_list = []
        for batch_ndx, sample in enumerate(dataloader):    
            with torch.no_grad(): 
                data = sample[0].to(device)
                f, _ = self.model.predict(data)
                evidence = F.relu(_)
                alpha = evidence + 1
                uncertainty = len(self.nl) / torch.sum(alpha, dim=1, keepdim=True)
                f = f.cpu().detach().numpy()    
                u_list += (uncertainty.cpu().detach().numpy()[:,0].tolist())
                try: 
                    if features_array == None:
                        features_array = f
                except:
                    features_array = np.concatenate((features_array, f), axis=0)
        return u_list, features_array


    def uncertaintySample(self, Visio = False):
        self.source_uncertainty_list, _ = self.calc_u(self.source_loader)
        self.target_uncertainty_list, self.target_features_array = self.calc_u(self.target_loader)

        mean_u = np.max(self.source_uncertainty_list)

        self.seen_array = np.array(self.target_uncertainty_list)[list(map(lambda x:x in self.nl, self.target_dataset.label))]
        self.unseen_array = np.array(self.target_uncertainty_list)[list(map(lambda x:x not in self.nl, self.target_dataset.label))]

        self.S_sample = self.target_features_array[np.array(self.target_uncertainty_list) >= mean_u]
        self.S_id = np.array(self.target_dataset.id)[np.array(self.target_uncertainty_list) >= mean_u]
        self.S_label = np.array(self.target_dataset.label)[np.array(self.target_uncertainty_list) >= mean_u]

        # 对数据解包的匿名函数 [[[距离 - ID - 特征]]] -> [ID - Label]
        # self.unpack_set_id_label = lambda x:[(self.S_id[int(_[0][1])], self.S_label[int(_[0][1])]) \
        #     if isinstance(_[0],list) else (self.S_id[int(_[1])], self.S_label[int(_[1])]) for _ in x]
        self.unpack_set_id_label = lambda x:[(int(_[0][1]), self.target_dataset.label[int(_[0][1])]) \
            if isinstance(_[0],list) else (_[1], self.target_dataset.label[int(_[1])]) for _ in x if len(_)!=0]
        self.unpack_list_id_label = lambda x:[([_[0] for _ in self.unpack_set_id_label(x_)], \
            [_[1] for _ in self.unpack_set_id_label(x_)]) for x_ in [x]][0]
        self.pack_dict_id_label = lambda x:[dict(zip(map(lambda x:str(x), self.unpack_list_id_label(x_)[0]), self.unpack_list_id_label(x_)[1])) for x_ in [x]][0]
        if self.S_sample.shape == 0:
            return True
        # 显示内外分布的柱状图
        if Visio:
            fig, ax = plt.subplots()
            ax.hist(self.seen_array, density=True, color='C0', label='Seen', alpha=0.5)
            ax.hist(self.unseen_array, density=True, color='C2', label='UnSeen', alpha=0.5)
            ax.hist(self.source_uncertainty_list, density=True, color='C3', label='source', alpha=0.5)
            ax.axvline(mean_u)

            plt.legend(loc='best')
            plt.savefig(os.path.join(args.output_dir, f"{len(self.nl)} of class number.svg"), dpi=150)
            plt.savefig(os.path.join(args.output_dir, f"{len(self.nl)} of class number.pdf"), dpi=300)
            # plt.show()
        return False

    def clustersSample(self, max_epochs = 10, Visio = False):
        newS_sample = [[id, item] for id, item in zip(self.S_id, self.S_sample)]
        cosine_clusters = CosineClusters(c_n, Euclidean=False)
        cosine_clusters.add_random_training_items(newS_sample)
        for i in trange(0, max_epochs):
            # args.logger.info("Epoch "+str(i))
            added = cosine_clusters.add_items_to_best_cluster(newS_sample)
            if added == 0:
                break

        # 4. 对聚类中心和离散点查询标签 记为$ql$  
        centroids = cosine_clusters.get_centroids(1)
        outliers = cosine_clusters.get_outliers(1)
        # args.logger.info(centroids)
        # args.logger.info(outliers)
        # args.logger.info(self.S_id, self.S_id.shape)

        ql = []
        ql = self.unpack_list_id_label(centroids)[1] + self.unpack_list_id_label(outliers)[1]
        self.ul = set(ql) - set(self.nl)     # ul 查询的开放集标签

            # raise Exception('Stopped iteration due to no additional tags, now tags total {}'.format(len(self.nl)))

        # 对开放集标签的簇类 中心\离群抽样
        sel_id_class = {}
        for i in self.ul:
            sample_choose_target = []           # 存取某个标签的样本列表
            c_j = ql[:int(len(ql)/2)].count(i)
            j = c_j if c_j != 0 else ql[-int(len(ql)/2):].count(i) 
            k = [int(s_n/j)]*(j-1) + [s_n - int(s_n/j)*(j-1)]  # 从每个簇中取多少数据   

            for index, _ in enumerate(cosine_clusters.clusters):
                try:
                    if c_j != 0:
                        # 中心抽样
                        if ql[:int(len(ql)/2)][index] == i:
                            sample_choose_target += _.get_centroid(k[j-1])
                    else:
                        # 离群抽样
                        if ql[-int(len(ql)/2):][index] == i:
                            sample_choose_target += _.get_outlier(k[j-1])
                except:
                    pass
                j = max(1, j-1)
                    
            sample_choose_target_id, sample_choose_target_label = self.unpack_list_id_label(sample_choose_target)
            print(f'There are {c_j} cluster class centers labeled as {i}, sampled separately according to {k} and each sample with label {sample_choose_target_label}')

            if not sel_id_class:
                sel_id_class = dict(zip(map(lambda x:str(x), sample_choose_target_id), sample_choose_target_label))
            else:
                sel_id_class = dict(sel_id_class, **dict(zip(map(lambda x:str(x), sample_choose_target_id), sample_choose_target_label)))

        # 待使用样本 = 前一次待使用 + 簇类中心 + 簇类离散 + 未知标签周围抽样
        _ = dict(sel_id_class,**self.pack_dict_id_label(centroids))
        _ = dict(_,**self.pack_dict_id_label(outliers))
        if not self.Sub:
            self.Sub = _
        else:
            self.Sub = dict(self.Sub,**_)

        for index, _label in enumerate(set(list(self.Sub.values())) | set(self.ul)):
            self.SubLabel, self.SubCount, self.SubId = np.array(list(map(lambda x:[x, list(self.Sub.values()).count(x), \
            np.array(list(self.Sub.keys()))[[_==x for _ in list(self.Sub.values())]]], set(self.Sub.values())))).T.tolist()

            # 标签不在self.Sub中、标签数量小于某个阈值 进入下个循环
            if _label not in self.SubLabel or self.SubCount[self.SubLabel.index(_label)] < m_n and _label not in self.ul:
                continue

            _index = self.SubLabel.index(_label)
            _id = self.SubId[_index]
            assert len(set(list(map(lambda x:self.Sub[x], _id)))) == 1 and list(map(lambda x:self.Sub[x], _id))[0] == _label
            _dict = dict(zip(map(lambda x:str(x), _id), [_label]*len(_id)))
            if not self.Ssd:
                self.Ssd = _dict
            else:
                self.Ssd = dict(self.Ssd,**_dict)
            self.Sub = dict(self.Sub.items() - _dict.items())   

        self.nl = list(set(self.nl) | set(self.Ssd.values()))

        if len(self.ul) == 0:
            return True

        if Visio:
            tsne = TSNE(n_components=2).fit_transform(self.S_sample) 
            plt.figure(figsize=(18, 6))
            plt.subplot(121)
            for label in [*range(m)]:
                indices = [i for i, l in enumerate(self.S_label) if l == label]
                if not indices:
                    continue
                current_tx = np.take(tsne[:, 0], indices)
                current_ty = np.take(tsne[:, 1], indices)
                plt.scatter(current_tx, current_ty, c=f'C{label}', label=label)
                plt.title('tar_truth', color='C0', fontsize=16, fontweight='bold')
            plt.legend(loc='best')

            sample_y = [cosine_clusters.clusters.index(_) for _ in cosine_clusters.item_cluster.values()]
            plt.subplot(122)
            for plabel in [*range(len(set(sample_y)))]:
                indices = [i for i, l in enumerate(sample_y) if l == plabel]
                current_tx = np.take(tsne[:, 0], indices)
                current_ty = np.take(tsne[:, 1], indices)
                plt.scatter(current_tx, current_ty, c=f'C{plabel}', label=plabel)
                plt.title('tar_cos', color='C0', fontsize=16, fontweight='bold')
            plt.legend(loc='best')
            plt.savefig(os.path.join(args.output_dir, f"{len(self.nl)} TSNE.svg"), dpi=150)
            plt.savefig(os.path.join(args.output_dir, f"{len(self.nl)} TSNE.pdf"), dpi=300)
            # plt.show()

        return False

    def UpdateSource(self):
        # 从目标域中取出开放集到源域
        ChooseLabel = list(self.Ssd.values())
        ChooseId = list(map(lambda x:int(x), list(self.Ssd.keys())))
        ChooseSlist = (np.array(self.dataset_t.samples)[ChooseId]).tolist()


        print('ChooseSlist:',ChooseSlist)
        # self.dataset_s = datasets.ImageFolder(root=os.path.join('Semisup', args.source), transform=train_transform)
        # # ImageFolder 继承 DatasetFolder， DatasetFolder 带 samples 变量
        # self.dataset_s_samples = np.array(self.dataset_s.samples)
        # self.source_dataset = self.dataset_s
        # self.source_loader = torch.utils.data.DataLoader(self.source_dataset, batch_size=batch_size, shuffle=True)
        self.dataset_s.samples = np.concatenate((self.dataset_s_samples, np.array(ChooseSlist))).tolist()
        self.source_dataset = self.dataset_s
        self.source_dataset.label = np.array([int(target) for target in np.array(self.dataset_s.samples)[:,1]])
        self.source_dataset.label = self.source_dataset.label - min(self.source_dataset.label)

        # self.source_dataset.samples = np.array(self.source_dataset.samples)
        # self.source_dataset.samples[:,1] = self.source_dataset.label
        # self.source_dataset.samples = self.source_dataset.samples.tolist()
        # args.logger.info('self.source_dataset.label:',self.source_dataset.label)
        for i in range(len(self.source_dataset.samples)):
            self.source_dataset.samples[i][1] = self.source_dataset.label[i]
        # args.logger.info('self.source_dataset.samples:',self.source_dataset.samples)
        # self.dataset_NEXs_samples = np.concatenate((self.dataset_s_samples, np.array(ChooseSlist)))
        # self.NEXsource_dataset = CustomImageDataset(self.dataset_NEXs_samples, train_transform)
        self.NEXsource_loader = torch.utils.data.DataLoader(self.source_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

        # self.source_dataset = self.NEXsource_dataset
        # self.source_loader = self.NEXsource_loader
        assert abs(len(self.source_dataset) - len(self.source_loader)*batch_size) < batch_size
        assert len(self.nl) == len(set(self.source_dataset.label.tolist()))
        args.logger.info(f'Number: {len(self.source_dataset)}\t Target Domain Category: {set(self.source_dataset.label)}')

        
    def DSAN_train(self, epoch, model, source_loader, target_train_loader, optimizer, bar):
        model.train()
        iter_source = iter(source_loader)
        iter_target = iter(target_train_loader)
        num_iter = len(source_loader)
        # num_iter = (len(target_train_loader) // len(source_loader) + 1) * (len(source_loader) + 1)
        for i in range(1, num_iter):
            data_source, label_source = iter_source.next()
            if data_source.shape[0] < batch_size:
                iter_source = iter(source_loader)
                data_source, label_source = iter_source.next()
            label_source = torch.tensor(list(map(lambda x:(torch.tensor(self.nl) == x).nonzero(as_tuple=True)[0], label_source)))

            data_target, _ = iter_target.next()
            if data_target.shape[0] < batch_size:
                iter_target = iter(target_train_loader)
                data_target, _ = iter_target.next()

            data_source, label_source = data_source.to(device), label_source.to(device)
            data_target = data_target.to(device)

            optimizer.zero_grad()
            label_source_pred, loss_lmmd = model(
                data_source, data_target, label_source)
            loss_cls = F.nll_loss(F.log_softmax(
                label_source_pred, dim=1), label_source)
            lambd = 2 / (1 + math.exp(-10 * (epoch) / self.nepoch)) - 1
            loss = loss_cls + 0.5 * lambd * loss_lmmd

            loss.backward()
            optimizer.step()

            # if i % 50 == 0:
            #     args.logger.info(f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')
        return loss, loss_cls, loss_lmmd
        # bar.set_postfix(loss=loss.item(), cls_Loss=loss_cls.item(), loss_lmmd=loss_lmmd.item())

    def DSAN_test(self, dataloader):
        self.model.eval()

        # 针对目标域的结果估计
        # OS*   : 已知类别中识别正确的数据/已知类别中所有数据                       -  Na/nk
        # OS    : 已知类别中识别正确的数据+未知类别识别正确为unkown数据/所有数据      -  (Na+Mu)/(nk+nu)
        # OSF   : 已知类别中识别正确的数据+未知类别识别正确的数据/所有数据            -  (Na+Ma)/(nk+nu)
        # Na\Ma\Mu -- nk\nu
        unkown_correct_count = [0] * m
        unkown_error_count = [0] * m
        Na, Ma, Mu = 0, 0, 0
        nk, nu = 0, 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                _, pred = self.model.predict(data)
                pred = pred.data.max(1)[1]

                for i,j in zip(target, pred):
                    if i in [*range(n)]:
                        nk += 1
                        if i==j:
                            Na += 1
                    else:
                        nu += 1
                        if j not in [*range(n)]:
                            Mu += 1
                        try:
                            # 遇到不在列表中的会报错
                            if (torch.tensor(self.nl).to(device) == i).nonzero(as_tuple=True)[0]==j:
                                Ma += 1
                                unkown_correct_count[i-n] += 1
                            else:
                                unkown_error_count[i-n] += 1
                        except:
                            pass

        OSstar = Na / nk
        OS = (Na + Mu) / (nk + nu)
        OSF = (Na + Ma) / (nk + nu)

        return OS, OSF, OSstar, unkown_correct_count, unkown_error_count

    def DSANTrain(self, nepoch = 10):
        self.model.train()

        self.nepoch = nepoch
        self.optimizer = torch.optim.SGD([
            {'params': self.model.feature_layers.parameters()},
            {'params': self.model.cls_fc.parameters(), 'lr': 0.01},
        ], lr=0.001, momentum=0.9, weight_decay=5e-4)
        self.model = self.model.to(device)
        args.logger.info(self.model.cls_fc)

        max_OSF = 0
        stop_iter = 0
        with trange(1, self.nepoch + 1) as pbar:
            for epoch in pbar:
                pbar.set_description(f"Transfer learning train {epoch}")
                sleep(0.5)
                self.optimizer.param_groups[0]['lr'] = 0.001 / math.pow((1 + 10 * (epoch - 1) / self.nepoch), 0.75)
                self.optimizer.param_groups[1]['lr'] = 0.01 / math.pow((1 + 10 * (epoch - 1) / self.nepoch), 0.75)
                loss, loss_cls, loss_lmmd = self.DSAN_train(epoch, self.model, self.source_loader, self.target_loader_disorder, self.optimizer, pbar)

                OS, OSF, OSstar, unkown_correct_list, unkown_error_list = self.DSAN_test(self.target_loader_disorder)
                if OSF > max_OSF:
                    max_OSF = OSF
                    max_info = [OS, OSF, OSstar]
                    stop_iter = 0
                stop_iter += 1
                if stop_iter > args.patience:
                    break
                pbar.set_postfix(OS=100. * OS, OSF=100. * OSF, OSstar=100. * OSstar, loss=loss.item(), cls_Loss=loss_cls.item(), loss_lmmd=loss_lmmd.item())
        
        args.logger.info(f'Counter: {Counter(list(A.Ssd.values()))}')
        self.progress[-1] = [len(A.nl), f'query rate: {len(list(A.Ssd.values()))}/{len(self.target_dataset)}={len(list(A.Ssd.values()))/len(self.target_dataset)}', 
                             f'OS: {max_info[0]}', 
                             f'OSF: {max_info[1]}', 
                             f'OS*: {max_info[2]}',
                             f'unkown_correct_list: {unkown_correct_list}',
                             f'unkown_error_list: {unkown_error_list}']
        args.logger.info(self.progress[-1])

if __name__ == '__main__':
    args.logger.info(args)

    while True:
        A = ATC()
        A.progress.append([])
        A.DSANTrain(nepoch = 50)
        while True:
            A.progress.append([])
            A.fineTrain(epochnum = 60)
            if A.uncertaintySample(Visio = True):
                break
            if A.clustersSample(Visio = True):
                break
            A.UpdateSource()
            A.UpdateModel()
            A.DSANTrain(nepoch = 50)
            for _ in A.progress:
                args.logger.info(_)
        A.Ssd = dict(A.Ssd, **A.Sub)
        A.nl = list(set(A.nl) | set(A.Ssd.values()))
        A.UpdateSource()
        A.UpdateModel()
        A.DSANTrain(nepoch = 50)
        for _ in A.progress:
            args.logger.info(_)
        
        if A.progress[-1][0] == n + m:
            break
        else:
            for file in os.listdir(args.output_dir):
                if file.endswith('.pdf'):
                    os.remove(os.path.join(args.output_dir, file))
                if file.endswith('.svg'):
                    os.remove(os.path.join(args.output_dir, file))