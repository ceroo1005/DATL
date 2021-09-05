import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torch.nn as nn
try:
    import ResNet
    import alexnet
except:
    from models import ResNet
    from models import alexnet
import torch.nn.functional as F

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


class LMMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).to(self.device)
        weight_tt = torch.from_numpy(weight_tt).to(self.device)
        weight_st = torch.from_numpy(weight_st).to(self.device)

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).to(self.device)
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32') 


class EDL_loss(nn.Module):
    def __init__(self, num_classes=31):
        super(EDL_loss, self).__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def relu_evidence(self, y):
        return F.relu(y)

    def one_hot_embedding(self, labels):
        y = torch.eye(self.num_classes)
        return y[labels]

    def loglikelihood_loss(self, y, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum(
            (y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood

    def kl_divergence(self, alpha):
        beta = torch.ones([1, self.num_classes], dtype=torch.float32, device=self.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - \
            torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                            keepdim=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                       keepdim=True) + lnB + lnB_uni
        return kl

    def mse_loss(self, y, alpha):
        loglikelihood = self.loglikelihood_loss(y, alpha)

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = self.kl_divergence(kl_alpha)
        return loglikelihood, kl_div


    def get_loss(self, output, y, beta = 1):
        target = self.one_hot_embedding(y).to(self.device)
        evidence = self.relu_evidence(output).to(self.device)
        alpha = evidence + 1
        loglikelihood, kl_div = self.mse_loss(target.float(), alpha)
        # loss = torch.mean(loglikelihood + beta * kl_div)
        loss = torch.mean(kl_div)
        return loss


class LoopNet(nn.Module):
    def __init__(self, num_classes=31):
        super(LoopNet, self).__init__()
        self.num_classes = num_classes
        self.lmmd_loss = LMMD_loss(class_num=self.num_classes)
        self.edl_loss = EDL_loss(num_classes=self.num_classes)
        self.feature_layers = ResNet.resnet50(True)
        self.bottle = nn.Linear(2048, 256)
        self.cls_fc = nn.Linear(256, self.num_classes)

    def forward(self, source, target, s_label):
        source = self.feature_layers(source)
        source = self.bottle(source)
        s_pred = self.cls_fc(source)

        target = self.feature_layers(target)
        target = self.bottle(target)
        t_label = self.cls_fc(target)
        loss = self.lmmd_loss.get_loss(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
        return s_pred, loss

    def uncforward(self, x, label, beta = 1):
        x = self.feature_layers(x)
        x = self.bottle(x)
        pred = self.cls_fc(x)
        loss = self.edl_loss.get_loss(pred, label, beta)
        return pred, loss

    def predict(self, x):
        f = self.feature_layers(x)
        x = self.bottle(f)
        pred = self.cls_fc(x)
        return [f, pred]


class LoopNet_DANN(nn.Module):
    def __init__(self, num_classes=31):
        super(LoopNet_DANN, self).__init__()
        self.num_classes = num_classes
        self.edl_loss = EDL_loss(num_classes=self.num_classes)
        self.feature_layers = ResNet.resnet50(True)
        self.cls_fc = nn.Linear(2048, self.num_classes)

    def forward(self, source, target, s_label):
        source = self.feature_layers(source)
        s_pred = self.cls_fc(source)

        target = self.feature_layers(target)
        t_label = self.cls_fc(target)
        loss = mmd_rbf_noaccelerate(source, target)
        return s_pred, loss

    def uncforward(self, x, label, beta = 1):
        x = self.feature_layers(x)
        pred = self.cls_fc(x)
        loss = self.edl_loss.get_loss(pred, label, beta)
        return pred, loss

    def predict(self, x):
        x = self.feature_layers(x)
        pred = self.cls_fc(x)
        return [x, pred]


class LoopNetAlexNet_DANN(nn.Module):
    def __init__(self, num_classes=31):
        super(LoopNetAlexNet_DANN, self).__init__()
        self.num_classes = num_classes
        self.edl_loss = EDL_loss(num_classes=self.num_classes)
        self.feature_layers = alexnet.alexnet(True)
        self.cls_fc = nn.Linear(2048, self.num_classes)

    def forward(self, source, target, s_label):
        source = self.feature_layers(source)
        s_pred = self.cls_fc(source)

        target = self.feature_layers(target)
        t_label = self.cls_fc(target)
        loss = mmd_rbf_noaccelerate(source, target)
        return s_pred, loss

    def uncforward(self, x, label, beta = 1):
        x = self.feature_layers(x)
        pred = self.cls_fc(x)
        loss = self.edl_loss.get_loss(pred, label, beta)
        return pred, loss

    def predict(self, x):
        x = self.feature_layers(x)
        pred = self.cls_fc(x)
        return [x, pred]

if __name__ == '__main__':
    x1 = torch.rand(1, 3, 224, 224)
    x2 = torch.rand(1, 3, 224, 224)
    y1 = torch.randint(3, 5, (1,))
    Net = LoopNet()
    pred, lmmdloss = Net(x1, x2, y1)
    print(lmmdloss)
    f, pred = Net.predict(x1)
    print(f.shape, pred.shape)
    pred, uncloss = Net.uncforward(x1, y1)
    print(uncloss)
