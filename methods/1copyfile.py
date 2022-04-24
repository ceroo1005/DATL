import os, shutil
from datasets import Office31
import numpy as np

class_names = Office31.CLASSES
share_classes = class_names[:10]
unkown_classes = class_names[21:31]
root = 'data'

def copy_file(src_name, dst_name, new_folder, args):
    try:
        shutil.rmtree(new_folder)
    except:
        pass

    old_folder = os.path.join(args.root, 'office31')

    old_src = os.path.join(old_folder, src_name, 'images')
    old_dst = os.path.join(old_folder, dst_name, 'images')
    new_src = os.path.join(new_folder, src_name)
    new_dst = os.path.join(new_folder, dst_name)

    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    if not os.path.exists(new_src):
        os.mkdir(new_src)
    if not os.path.exists(new_dst):
        os.mkdir(new_dst)

    all_target_img_path_list = []

    # 遍历类别名
    for class_name in os.listdir(old_src):
        # 转移数据
        if class_name in share_classes:
            # 对于共享类别/已知类别 源域和目标域同时转移到新文件夹中
            shutil.copytree(os.path.join(old_src, class_name), os.path.join(new_src, class_name))
            shutil.copytree(os.path.join(old_dst, class_name), os.path.join(new_dst, class_name))
            # 添加所有的图片路径
            all_target_img_path_list += list(map(lambda x: os.path.join(new_dst, class_name, x), os.listdir(os.path.join(new_dst, class_name))))
        elif class_name in unkown_classes:
            # 对于UNKOWN/未知类别 只有目标域转移到新文件夹中
            shutil.copytree(os.path.join(old_dst, class_name), os.path.join(new_dst, class_name))
            all_target_img_path_list += list(map(lambda x: os.path.join(new_dst, class_name, x), os.listdir(os.path.join(new_dst, class_name))))


    sample_num = int(len(all_target_img_path_list) * args.smi_sample_rate)
    np.random.shuffle(all_target_img_path_list)


    for class_ in unkown_classes:
        if not os.path.exists(os.path.join(new_src, class_)):
            os.mkdir(os.path.join(new_src, class_))
        old_dst_img = os.listdir(os.path.join(old_dst, class_))
        np.random.shuffle(old_dst_img)
        shutil.copy(os.path.join(old_dst, class_, old_dst_img[0]), os.path.join(new_src, class_, old_dst_img[0]))

    # 随机选择一些数据
    # 从目标域中的采样结果复制到源域中做半监督学习
    num = 10
    for index, img_path in enumerate(all_target_img_path_list):
        class_, name_ = img_path.split('/')[-2:]
        if not os.path.isfile(os.path.join(new_src, class_, name_)):
            shutil.copy(os.path.join(old_dst, class_, name_), os.path.join(new_src, class_, name_))
            num += 1
        if num >= sample_num:
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Semi-supervised method -- 1. Production data')
    parser.add_argument('--root', default='/data')
    parser.add_argument('--save', default='/data/smisup/')
    parser.add_argument('--smi-sample-rate', default=0.1, type=float, help='sampling rate')
    parser.add_argument('--test-only', action='store_true', help='Make only one domain task data')
    args = parser.parse_args()
    if args.test_only:
        copy_file('webcam', 'dslr', os.path.join(args.save, 'Office31smiWD'), args)
    else:
        copy_file('webcam', 'dslr', os.path.join(args.save, 'Office31smiWD'), args)
        copy_file('webcam', 'amazon', os.path.join(args.save, 'Office31smiWA'), args)
        copy_file('dslr', 'amazon', os.path.join(args.save, 'Office31smiDA'), args)
        copy_file('dslr', 'webcam', os.path.join(args.save, 'Office31smiDW'), args)
        copy_file('amazon', 'webcam', os.path.join(args.save, 'Office31smiAW'), args)
        copy_file('amazon', 'dslr', os.path.join(args.save, 'Office31smiAD'), args)