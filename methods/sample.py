import os
import copy
import random
import numpy as np

import scipy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy, ImageClassifier

import ops
import utils

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)


al_dict = {}
def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls
    return decorator

def get_strategy(sample, *args):
    if sample not in al_dict: raise NotImplementedError
    return al_dict[sample](*args)


class SamplingStrategy:
    """ 
    Sampling Strategy wrapper class
    """
    def __init__(self, dataset, train_idx, model, device, args):
        self.dataset = dataset
        self.train_idx = np.array(train_idx)
        self.model = model
        self.device = device
        self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)
        self.args = args

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb


@register_strategy('uniform')
class RandomSampling(SamplingStrategy):
    """
    Uniform sampling 
    """
    def __init__(self, dataset, train_idx, model, device, args):
        super(RandomSampling, self).__init__(dataset, train_idx, model, device, args)

    def query(self, n):
        return np.random.choice(np.where(self.idxs_lb==0)[0], n, replace=False)


@register_strategy('entropy')
class EntropySampling(SamplingStrategy):
    """
    Entropy sampling 
    """
    def __init__(self, dataset, train_idx, model, device, args):
        super(EntropySampling, self).__init__(dataset, train_idx, model, device, args)

    def query(self, n):
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(self.dataset, num_workers=2, batch_size=self.args.batch_size, drop_last=False)
        all_log_probs = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                scores = self.model(data)
                log_probs = F.log_softmax(scores, dim=1)
                all_log_probs.append(log_probs)

        all_log_probs = torch.cat(all_log_probs)
        all_probs = torch.exp(all_log_probs)

        # Compute entropy
        self.E = -(all_probs*all_log_probs).sum(1)
        # [2.3005, 2.2999, 2.2992, 2.2990, 2.2979, 2.2976, 2.2976, 2.2976, 2.2975
        # 从大到小排序
        values, indices = self.E.sort(descending=True)
        return indices[:n].cpu().numpy()


@register_strategy('margin')
class MarginSampling(SamplingStrategy):
    """
    Margin sampling 
    """
    def __init__(self, dataset, train_idx, model, device, args):
        super(MarginSampling, self).__init__(dataset, train_idx, model, device, args)

    def query(self, n):
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(self.dataset, num_workers=self.args.workers, batch_size=self.args.batch_size, drop_last=False)
        all_log_probs = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                scores = self.model(data)
                log_probs = F.log_softmax(scores, dim=1)
                all_log_probs.append(log_probs)

        all_log_probs = torch.cat(all_log_probs)
        all_probs = torch.exp(all_log_probs)

        all_probs = all_probs.sort(descending=True)[0]

        # Compute margin
        self.M = all_probs[:,0] - all_probs[:,1]
        # [3.6832e-02, 2.9257e-02, 2.5097e-02, 2.3324e-02, 2.0978e-02, 1.9782e-02,
        # 从大到小排序
        values, indices = self.M.sort(descending=True)
        return indices[:n].cpu().numpy()



from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

class ImageFolder(DatasetFolder):
    def __init__(
            self,
            root,
            loader = default_loader,
            extensions = None,
            transform = None,
            target_transform = None,
            is_valid_file = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        try:
            classes, class_to_idx = self._find_classes(self.root)
        except:
            classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset_beta(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

    def make_dataset_beta(self,
        directory,
        class_to_idx = None,
        extensions = None,
        is_valid_file = None,
    ):
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if os.path.isfile(path):
                        item = path, class_index
                        instances.append(item)

        return instances



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='smisupervised Domain Adaptation')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    query_rate = 0.1
    query_type = 'uniform'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dataset = ImageFolder(r'C:\Users\ceroo\Desktop\DA数据集\OfficeCaltech\dslr', transform=ops.get_train_transform())
    target_loader = DataLoader(target_dataset, batch_size=4, shuffle=True, num_workers=2)
    
    train_idx = np.arange(len(target_dataset))
    print('train_idx', len(train_idx))
    
    backbone = utils.get_model('resnet50', pretrain=True)
    classifier = ImageClassifier(backbone, 10, bottleneck_dim=256).to(device)

    sampling_strategy = get_strategy(query_type, target_dataset, train_idx, classifier, device, args)
    idxs_lb = np.zeros(len(train_idx), dtype=bool)
    idxs = sampling_strategy.query(int(query_rate * len(train_idx)))
    idxs_lb[idxs] = True
    print('idxs', len(idxs), idxs)
    
    train_sampler = SubsetRandomSampler(train_idx[idxs_lb])
    tgt_sup_loader = DataLoader(target_dataset, batch_size=4, num_workers=2, sampler=train_sampler)
    print('tgt_sup_loader', len(tgt_sup_loader))

    train_sampler = SubsetRandomSampler(np.concatenate([train_idx, idxs + len(train_idx)]))
    new_target_loader = DataLoader(ConcatDataset([target_dataset, target_dataset]), batch_size=4, num_workers=2, sampler=train_sampler)
    print('target_loader', len(target_loader))
    print('new_target_loader', len(new_target_loader))
    
    # for i, (x, y) in enumerate(new_target_loader):
    #     print('i', i, 'x', x.size(), 'y', y)

    # for i, (x, y) in enumerate(tgt_sup_loader):
    #     print('i', i, 'x', x.size(), 'y', y)

    iter_data = iter(new_target_loader)
    while True:
        x, y = next(iter_data)
        print('x', x.size(), 'y', y)