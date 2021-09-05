import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import models

import math
from tqdm import trange, tqdm
from clusters import *
from utils import *
from time import sleep
import shutil
from collections import Counter

import argparse
parser = argparse.ArgumentParser(description='Training code - ATC')
parser.add_argument('--source', default='dslr', help='dslr | webcam | amazon')
parser.add_argument('--target', default='webcam', help='dslr | webcam | amazon')
args = parser.parse_args()


savepath = '.'
n = 10
m = 20
c_n = 4
s_n = 5
m_n = 4
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose(
            [transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

test_transform = transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ToTensor()])

kwargs = {'num_workers': 1, 'pin_memory': True}


class_name_alllist = os.listdir(os.path.join('Office31', args.source, 'images'))
class_name_alllist.sort()

if os.path.isdir('Semisup'):
    shutil.rmtree('Semisup')

os.makedirs(os.path.join('Semisup', args.source))
os.makedirs(os.path.join('Semisup', args.source, 'images'))
os.makedirs(os.path.join('Semisup', args.target))
os.makedirs(os.path.join('Semisup', args.target, 'images'))

for clsname in class_name_alllist[:n]:
    shutil.copytree(os.path.join('Office31', args.source, 'images', clsname), os.path.join('Semisup', args.source, 'images', clsname))

for clsname in (class_name_alllist[:10]+class_name_alllist[21:21+m-10]):
    shutil.copytree(os.path.join('Office31', args.target, 'images', clsname), os.path.join('Semisup', args.target, 'images', clsname))


class ATC:
    def __init__(self) -> None:
        self.model = models.LoopNet_DANN(n).to(device)
        self.load_data()
        self.nl, self.ul = [*range(n)], []
        self.Sub, self.Ssd = {}, {}
        # [类别数量、抽样比例、已有类别精度、所有类别精度]
        self.progress = []

    def load_data(self):
        self.dataset_s = datasets.ImageFolder(root=os.path.join('Semisup', args.source, 'images'))
        self.dataset_t = datasets.ImageFolder(root=os.path.join('Semisup', args.target, 'images'))
        dataset_t_test = datasets.ImageFolder(root=os.path.join('Semisup', args.target, 'images'))

        self.dataset_s_samples = np.array(self.dataset_s.samples)
        self.dataset_t_samples = np.array(self.dataset_t.samples)

        self.source_dataset = CustomImageDataset(self.dataset_s_samples, train_transform)
        self.source_loader = torch.utils.data.DataLoader(self.source_dataset, batch_size=batch_size, shuffle=True)
        self.target_dataset = CustomImageDataset(self.dataset_t_samples, train_transform)
        self.target_dataset.id = np.array([str(_) for _ in range(len(self.target_dataset))])
        self.target_loader = torch.utils.data.DataLoader(self.target_dataset, batch_size=batch_size)
        self.target_loader_disorder = torch.utils.data.DataLoader(self.target_dataset, batch_size=batch_size, shuffle=True)

        print(f'Number: {len(self.source_dataset)}\t','Source Domain Category:',set(self.source_dataset.label))
        print(f'Number: {len(self.target_dataset)}\t','Target Domain Category:',set(self.target_dataset.label))

    def UpdateModel(self):
        if self.model.num_classes != len(self.nl):
            self.model.num_classes = len(self.nl)
            # self.model.lmmd_loss = models.LMMD_loss(class_num=len(self.nl))
            self.model.edl_loss = models.EDL_loss(num_classes=len(self.nl))
            self.model.cls_fc = nn.Linear(2048, len(self.nl))

    def finetest(self, dataloader, epoch):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                target = torch.tensor(list(map(lambda x:(torch.tensor(self.nl) == x).nonzero(as_tuple=True)[0], target)))
                data, target = data.to(device), target.to(device)

                # outputs, loss = self.model.uncforward(data, target)
                # pred = outputs.data.max(1)[1]
                # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                _, pred = self.model.predict(data)
                pred = pred.data.max(1)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        return  correct/len(dataloader.dataset)

    def fineTrain(self, epochnum = 10):
        self.unc_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0003, momentum=0.9)
        bestloss = 1e10 
        # for epoch in trange(1, epochnum+1, desc="FineTrain: "):
        with trange(1, epochnum+1) as pbar:
            for epoch in pbar:
                sleep(0.5)
                pbar.set_description(f"FineTrain {epoch}")

                self.model.train()
                sumloss = 0
                for i, data in enumerate(self.source_loader):
                    inputs, labels = data[0], data[1]
                    labels = torch.tensor(list(map(lambda x:(torch.tensor(self.nl) == x).nonzero(as_tuple=True)[0], labels)))
                    inputs, labels = inputs.to(device), labels.to(device)
                    self.unc_optimizer.zero_grad()
                    annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32), torch.tensor(epoch / epochnum, dtype=torch.float32))
                    output, klloss = self.model.uncforward(inputs, labels, annealing_coef)
                    loss_cls = F.nll_loss(F.log_softmax(
                    output, dim=1), labels)
                    loss = klloss + 0.1 * loss_cls

                    sumloss = sumloss + loss
                    loss.backward()
                    self.unc_optimizer.step()
                loss_ = sumloss/len(self.source_loader)
                Accuracy = self.finetest(self.source_loader, epoch)

                # print(f'Accuracy: ({100. * Accuracy:.2f}%) \t loss: {loss:.5f}')
                if loss_ <= bestloss:
                    bestloss = loss_
                    torch.save(self.model.state_dict(), 'fine_weights.pth')

                pbar.set_postfix(loss=loss.item(), minloss=bestloss.item(), accuracy=100. * Accuracy.item())

        self.model.load_state_dict(torch.load('fine_weights.pth'))
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
            plt.savefig(f"{len(self.nl)} of class number.svg", dpi=150)
            plt.show()
        return False

    def clustersSample(self, max_epochs = 10, Visio = False):
        newS_sample = [[id, item] for id, item in zip(self.S_id, self.S_sample)]
        cosine_clusters = CosineClusters(c_n, Euclidean=False)
        cosine_clusters.add_random_training_items(newS_sample)
        for i in trange(0, max_epochs):
            # print("Epoch "+str(i))
            added = cosine_clusters.add_items_to_best_cluster(newS_sample)
            if added == 0:
                break

        # 4. 对聚类中心和离散点查询标签 记为$ql$  
        centroids = cosine_clusters.get_centroids(1)
        outliers = cosine_clusters.get_outliers(1)
        # print(centroids)
        # print(outliers)
        # print(self.S_id, self.S_id.shape)

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
                if c_j != 0:
                    if ql[:int(len(ql)/2)][index] == i:
                        sample_choose_target += _.get_centroid(k[j-1])
                else:
                    if ql[-int(len(ql)/2):][index] == i:
                        sample_choose_target += _.get_outlier(k[j-1])
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
            plt.savefig(f"{len(self.nl)} TSNE.svg", dpi=150)
            plt.show()

        return False

    def UpdateSource(self):
        # 从目标域中取出开放集到源域
        ChooseLabel = list(self.Ssd.values())
        ChooseId = list(map(lambda x:int(x), list(self.Ssd.keys())))
        ChooseSlist = (np.array(self.dataset_t.samples)[ChooseId]).tolist()

        self.dataset_NEXs_samples = np.concatenate((self.dataset_s_samples, np.array(ChooseSlist)))
        self.NEXsource_dataset = CustomImageDataset(self.dataset_NEXs_samples, train_transform)
        self.NEXsource_loader = torch.utils.data.DataLoader(self.NEXsource_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.source_dataset = self.NEXsource_dataset
        self.source_loader = self.NEXsource_loader
        assert abs(len(self.source_dataset) - len(self.source_loader)*batch_size) < batch_size
        assert len(self.nl) == len(set(self.source_dataset.label.tolist()))
        print(f'Number: {len(self.source_dataset)}\t','Target Domain Category:',set(self.NEXsource_dataset.label.tolist()))

        
    def DSAN_train(self, epoch, model, source_loader, target_train_loader, optimizer, bar):
        model.train()
        iter_source = iter(source_loader)
        iter_target = iter(target_train_loader)
        num_iter = len(source_loader)
        # num_iter = (len(target_train_loader) // len(source_loader) + 1) * (len(source_loader) + 1)
        for i in range(1, num_iter):
            data_source, label_source = iter_source.next()
            if data_source.shape[0] < batch_size:
                # print('data_source.shape:',data_source.shape)
                iter_source = iter(source_loader)
                data_source, label_source = iter_source.next()
            label_source = torch.tensor(list(map(lambda x:(torch.tensor(self.nl) == x).nonzero(as_tuple=True)[0], label_source)))

            # if (i+1) % len(target_train_loader) == 0:
            #     iter_target = iter(target_train_loader)
            data_target, _ = iter_target.next()
            if data_target.shape[0] < batch_size:
                # print('data_source.shape:',data_source.shape)
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
            #     print(f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')
        bar.set_postfix(loss=loss.item(), cls_Loss=loss_cls.item(), loss_lmmd=loss_lmmd.item())

    def DSAN_test(self, dataloader):
        self.model.eval()
        correct = 0
        seen_total = 0
        seen_correct = 0

        share_total = 0
        share_correct = 0

        unkown_total = 0
        unkown_correct = 0
        unkown_cover_correct = 0

        unkown_correct_count = [0] * (m-n)
        unkown_error_count = [0] * (m-n)
        with torch.no_grad():
            for data, target in dataloader:
                # target = torch.tensor([len(nl)+1 if (torch.tensor(nl) == _).nonzero(as_tuple=True)[0].size()[0]==0 \
                #  else  (torch.tensor(nl) == _).nonzero(as_tuple=True)[0] for _ in target])
                data, target = data.to(device), target.to(device)
                _, pred = self.model.predict(data)
                # sum up batch loss
                pred = pred.data.max(1)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                for i,j in zip(target, pred):
                    # 一些值在已知列表中不存在 抛弃掉这些数据
                    if (torch.tensor(self.nl).to(device) == i).nonzero(as_tuple=True)[0].size()[0]!=0:
                        seen_total += 1
                        if (torch.tensor(self.nl).to(device) == i).nonzero(as_tuple=True)[0]==j:
                            seen_correct += 1

                for i,j in zip(target, pred):
                    if i in [*range(n)]:
                        share_total += 1
                        if i==j:
                            share_correct += 1

                for i,j in zip(target, pred):
                    if i not in [*range(n)]:
                        unkown_total += 1
                        if j not in [*range(n)]:
                            unkown_cover_correct += 1
                        try:
                            # 遇到不在列表中的会报错
                            if (torch.tensor(self.nl).to(device) == i).nonzero(as_tuple=True)[0]==j:
                                unkown_correct += 1
                                unkown_correct_count[i-n] += 1
                            else:
                                unkown_error_count[i-n] += 1
                        except:
                            pass

        if unkown_total == 0:
            # 全局粗略精度、已知类别精度、已知类别正确个数、已知类别全局精度、共享精度、细腻未知精度、粗略未知精度、未知细腻预测正确列表、未知细腻预测错误列表
            return (share_correct+unkown_cover_correct) / (share_total+unkown_total),seen_correct / seen_total, seen_correct, seen_correct / len(dataloader.dataset), share_correct / share_total, -1, unkown_cover_correct / unkown_total, unkown_correct_count, unkown_error_count
        else:
            return (share_correct+unkown_cover_correct) / (share_total+unkown_total),seen_correct / seen_total, seen_correct, seen_correct / len(dataloader.dataset), share_correct / share_total, unkown_correct / unkown_total, unkown_cover_correct / unkown_total, unkown_correct_count, unkown_error_count

    def DSANTrain(self, nepoch = 10):
        self.model.train()

        self.nepoch = nepoch
        self.optimizer = torch.optim.SGD([
            {'params': self.model.feature_layers.parameters()},
            {'params': self.model.cls_fc.parameters(), 'lr': 0.01},
        ], lr=0.001, momentum=0.9, weight_decay=5e-4)
        self.model = self.model.to(device)

        print(self.model.cls_fc, '\n\n')

        max_correct = 0
        max_seenAcy = 0
        max_shareAcy = 0
        max_CoverAcy = 0
        max_unknoweAcy = 0
        max_unknoweCoverAcy = 0

        # for epoch in range(1, self.nepoch + 1):
        with trange(1, self.nepoch + 1) as pbar:

            for epoch in pbar:
                pbar.set_description(f"Transfer learning train {epoch}")
                sleep(0.5)
                self.optimizer.param_groups[0]['lr'] = 0.001 / math.pow((1 + 10 * (epoch - 1) / self.nepoch), 0.75)
                self.optimizer.param_groups[1]['lr'] = 0.01 / math.pow((1 + 10 * (epoch - 1) / self.nepoch), 0.75)
                self.DSAN_train(epoch, self.model, self.source_loader, self.target_loader_disorder, self.optimizer, pbar)

                cover_acy, t_seenAcy, t_correct, t_AllAcy, share_acy, unknow_acy, unknow_cover_acy, unkown_correct_count, unkown_error_count = self.DSAN_test(self.target_loader_disorder)
                if unknow_acy >= max_unknoweAcy:
                    best_c ,best_e = unkown_correct_count, unkown_error_count

                max_correct = max(max_correct, t_correct)
                max_seenAcy = max(max_seenAcy, t_seenAcy)
                max_shareAcy = max(max_shareAcy, share_acy)
                max_CoverAcy = max(max_CoverAcy, cover_acy)
                max_unknoweAcy = max(max_unknoweAcy, unknow_acy)
                max_unknoweCoverAcy = max(max_unknoweCoverAcy, unknow_cover_acy)
                pbar.set_postfix(max_CoverAcy=100. * max_CoverAcy, SeenAccuracy=100. * t_seenAcy, AllAccuracy=100. * t_correct / len(self.target_dataset), ShareAccuracy= 100. * share_acy, UnkownAccuracy=100. * unknow_acy, UnkownCoverAccuracy=100. * unknow_cover_acy, unkown_correct_count=unkown_correct_count, unkown_error_count=unkown_error_count)
        
        print('\n',Counter(list(A.Ssd.values())))
        self.progress[-1] = [len(A.nl), f'query rate: {len(list(A.Ssd.values()))}/{len(self.target_dataset)}={len(list(A.Ssd.values()))/len(self.target_dataset)}', f'max seen acy: {max_seenAcy}', f'max cover acy: {max_CoverAcy}', f'max share acy: {max_shareAcy}', f'max unkown acy: {max_unknoweAcy}', f'max unkown cover acy: {max_unknoweCoverAcy}', f'max all acy: {max_correct / len(self.target_dataset)}', best_c ,best_e]

if __name__ == '__main__':
    A = ATC()
    A.progress.append([])
    A.DSANTrain(nepoch = 10)
    while True:
        A.progress.append([])
        A.fineTrain(epochnum = 60)
        if A.uncertaintySample(Visio = True):
            break
        if A.clustersSample(Visio=True):
            break
        A.UpdateSource()
        A.UpdateModel()
        A.DSANTrain(nepoch = 50)
    A.Ssd = dict(A.Ssd, **A.Sub)
    A.nl = list(set(A.nl) | set(A.Ssd.values()))
    A.UpdateSource()
    A.UpdateModel()
    A.DSANTrain(nepoch = 50)


    for _ in A.progress:
        print(_)