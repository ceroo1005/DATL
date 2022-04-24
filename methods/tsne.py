import os
import torch
import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.colors as mcolors

overlap = {name for name in mcolors.CSS4_COLORS
           if f'xkcd:{name}' in mcolors.XKCD_COLORS}
overlap = sorted(overlap)[7:]

def collect_feature(data_loader, feature_extractor, max_num_features=None) -> torch.Tensor:
    feature_extractor.eval()
    all_features = []
    all_target = []
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            all_target += (target.cpu().detach().numpy().tolist())
            if max_num_features is not None and i >= max_num_features:
                break
            if torch.cuda.is_available():
                images = images.cuda()
            feature = feature_extractor(images).cpu()
            all_features.append(feature)
    return torch.cat(all_features, dim=0), all_target


def visualize(args, model, temp_source_dataLoader, temp_target_dataLoader):
    if torch.cuda.is_available():
        model.cuda()
    source_feature, source_label = collect_feature(temp_source_dataLoader, model)
    target_feature, target_label = collect_feature(temp_target_dataLoader, model)
    name = os.path.join(args.output_dir, f'{args.num_classes}-{len(args.raw_know) + len(args.raw_unknow)}-tsne_with')

    visualize_swich(args, source_feature, target_feature, source_label, target_label, 'normal', name+'_normal.svg')
    visualize_swich(args, source_feature, target_feature, source_label, target_label, 'unkonw', name+'_unkonw.svg')
    visualize_swich(args, source_feature, target_feature, source_label, target_label, 'label', name+'_label.svg')

def visualize_swich(args, source_feature, target_feature, source_label, target_label, switch_type, filename):
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)
    all_label = source_label + target_label
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if switch_type == 'normal':
        # 区分源域和目标域
        s = ['r'] * len(source_label)
        t = ['b'] * len(target_label)
        plt.scatter(X_tsne[:, 0][:len(source_label)], X_tsne[:, 1][:len(source_label)], c=s, s=20)
        plt.scatter(X_tsne[:, 0][len(source_label):], X_tsne[:, 1][len(source_label):], c=t, s=20, marker='^')
    elif switch_type == 'unkonw':
        # 区分源域、目标域中未知域、目标域、目标域中未知域
        s = ['r' if i < len(args.raw_know)else 'g' for i in source_label]
        t = ['b' if i < len(args.raw_know)else 'g' for i in target_label]

        plt.scatter(X_tsne[:, 0][:len(source_label)], X_tsne[:, 1][:len(source_label)], c=s, s=20)
        plt.scatter(X_tsne[:, 0][len(source_label):], X_tsne[:, 1][len(source_label):], c=t, s=20, marker='^')
    elif switch_type == 'label':
        # 区分各个类别
        s = [mcolors.CSS4_COLORS[overlap[l]] for l in source_label]
        t = [mcolors.CSS4_COLORS[overlap[l]] for l in target_label]
        plt.scatter(X_tsne[:, 0][:len(source_label)], X_tsne[:, 1][:len(source_label)], c=s, s=20)
        plt.scatter(X_tsne[:, 0][len(source_label):], X_tsne[:, 1][len(source_label):], c=t, s=20, marker='^')

    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)