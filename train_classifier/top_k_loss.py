"""
2022-07-12
Part1: Training biased classfiers & Extracting bias-conflict samples with methodology
This implementation is for understaning the overall structure of our methods.
"""

import argparse
import os, cv2, json, random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
import sklearn.metrics as m

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as T

from models import * # calling call_by_name models = mlp, conv ... and others...
from utils import *    # Generalized Cross Entropy

# Pytorch determistic
random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))    # 스크립트 경로
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)             # 프로젝트 경로(module import)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class CustomMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, res):
        self.dict = res

    def update(self, key, score):        
        self.dict[key].append(score)

    def save(self, target_directory, filename):
        if filename:
            pd.DataFrame(self.dict, index=None).to_csv(f'{target_directory}/{filename}.csv')
        else:
            pd.DataFrame(self.dict, index=None).to_csv(target_directory+'results.csv')

    def is_best(self, key):
        if len(self.dict[list(self.dict.keys())[0]]) == 1:
            return True
        maximum = max(self.dict[key][:-1])
        current = self.dict[key][-1]

        if maximum < current:
            return True

    def print_info(self, key):
        best1   = round(max(self.dict[key]),4)
        current1= round(self.dict[key][-1], 4)

        str1 = f'Best/Curr {key}: {best1}/{current1}'
        print(f'\t{str1}')

def save(state, epoch, save_dir, model, is_parallel=None):
    os.makedirs(save_dir, exist_ok=True)
    
    target_path = f'{save_dir}/{state}.path.tar'
    
    with open(target_path, "wb") as f:
        if not is_parallel:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),}, f)
        else:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),}, f)

def get_transform(args):
    
    train_transform = T.Compose([
            T.Resize(args.image_size),
            T.ToTensor()
    ])

    valid_transform = T.Compose([
    T.Resize(args.image_size),
    T.ToTensor()
    ])

    return train_transform, valid_transform


class BaseDataset(nn.Module):
    def __init__(self, path, args, transform=None):
        super(BaseDataset, self).__init__()
        self.path = path
        self.args = args
        self.transform = transform

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        origin_path  = self.path[idx]

        if self.args.exp == 'new-cmnist' or self.args.exp == 'cifar10c':
            label = int(origin_path.split('_')[-2]) 
        else:
            label = int(origin_path.split('_')[-2])
            # print(label)
            # label = int(origin_path.split('/')[-2])

        image = Image.open(origin_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return idx, image, label, origin_path


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.lr_decay_schedule:
        lr *= args.lr_decay_rate if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_args(args):
    if args.exp == 'new-cmnist' or 'cmnist':
        args.w, args.h = 28, 28
        args.lr = 0.001
        args.batch = 256
        args.num_classes = 10
    if args.exp == 'cifar10c':
        args.w, args.h = 32, 32
        args.lr = 0.001
        args.batch = 128
        args.num_classes = 10
    if args.exp == 'bffhq':
        args.w, args.h = 256, 256
        args.lr = 0.0001
        args.batch = 128
        args.num_classes = 2
    if args.exp == 'bar':
        args.w, args.h = 256, 256
        args.lr = 0.0001
        args.batch = 64
        args.num_classes = 6
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Training Biased Classifier")

    # parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--root", type=str, default=os.path.join(ROOT_PROJECT_DIR, 'data'), help="Dataset root")
    parser.add_argument("--save_model_root", type=str, default=os.path.join(PROJECT_DIR, 'results'), help="where the model was saved")
    parser.add_argument("--save_image_root", type=str, default=os.path.join(PROJECT_DIR, 'results'), help="where the model was saved")
    parser.add_argument("--exp", type=str, default='bffhq', help="Dataset name")     # new-cmnist/cifar10c/bffhq/bar
    parser.add_argument("--data_type", type=str, default='bffhq', help='kind of data used')
    parser.add_argument("--pct", type=str, default="5pct", help="Percent name")
    parser.add_argument("--etc", type=str, default='vanilla', help="Experiment name")
    # parser.add_argument("--loss", type=str, required=True)          # GCE || CE
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--K", type=int, default=10)
    # parser.add_argument("--scheduler",  action='store_true', help='Using scheduler')
    # parser.add_argument("--pretrained", action='store_true', help='Using Imagenet Pretrained')
    
    args = parser.parse_args()
    set_args(args)

    root = args.root
    args.image_size = (args.w, args.h)
    args.image_shape = (3, args.w, args.h)
    
    # args.lr_decay_rate = 0.1
    # args.lr_decay_schedule = [40, 60, 80]
    
    args.data_root = f'{root}/{args.exp}/'

    args.save_model_root = f'{args.save_model_root}/pretrained/{args.exp}-{args.pct}-{args.etc}/'

    model = call_by_name(args)
    model.load_state_dict(torch.load(os.path.join(args.save_model_root, 'best.path.tar'))['model_state_dict'])
    model.cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion_GCE = GeneralizedCELoss().cuda() # GCE Loss
    criterion_CE  = nn.CrossEntropyLoss().cuda() # GCE Loss

    ### BUILD DATASET
    train_align    = [y for x in os.walk(os.path.join(args.data_root, args.pct, 'align')) for y in glob(os.path.join(x[0], '*/*.png'))]
    train_conflict = [y for x in os.walk(os.path.join(args.data_root, args.pct, 'conflict')) for y in glob(os.path.join(x[0], '*/*.png'))]

    train_data = train_align + train_conflict
 
    if args.exp == 'bffhq':
        valid_data = [y for x in os.walk(os.path.join(args.data_root, 'valid')) for y in glob(os.path.join(x[0], '*.png'))] 
    if args.exp == 'cifar10c':
        valid_data = [y for x in os.walk(os.path.join(args.data_root, args.pct, 'valid')) for y in glob(os.path.join(x[0], '*.png'))]
    test_data  = [y for x in os.walk(os.path.join(args.data_root, 'test')) for y in glob(os.path.join(x[0], '*.png'))]
    
    # print(args.pct)
    print(len(train_data))
    # print(len(valid_data), 'valid length')
    if args.exp == 'new-cmnist' or args.exp == 'cifar10c':
        label_attr = np.array([int(each.split('_')[-2]) for each in test_data])
        bias_attr = np.array([int(each.split('_')[-1][0]) for each in test_data])
    else:
        label_attr = np.array([int(each.split('_')[-1][0]) for each in test_data])
        bias_attr = np.array([int(each.split('_')[-2]) for each in test_data])
    test_align = np.array(test_data)[label_attr == bias_attr]
    test_conflict = np.array(test_data)[label_attr != bias_attr]

    train_transform, valid_transform = get_transform(args)
    trainSet = BaseDataset(train_data, args, transform=train_transform)
    validSet = BaseDataset(valid_data, args, transform=valid_transform)
    testSet_align = BaseDataset(test_align, args, transform=valid_transform)
    testSet_conflict = BaseDataset(test_conflict, args, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(trainSet, batch_size=args.batch, shuffle=True, drop_last=False, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(validSet, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=2)
    bias_test_loader  = torch.utils.data.DataLoader(testSet_align, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=2)
    unbias_test_loader  = torch.utils.data.DataLoader(testSet_conflict, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=2)

    res = {'train_accuracy':[],'train_loss':[], 'valid_accuracy':[],'valid_loss':[], 'bias_test_accuracy':[],'unbias_test_accuracy':[]}
    meter = CustomMeter(res)


    ###################################################
    ###################################################
    # Phase 2: Extracting Bias Conflict Samples
    ###################################################
    ###################################################
    
    # Calculating Cross-Entropy Loss
    model.eval()
    
    loss_dict = {each:{
        'instance':[], 'path':[], 'loss':[], } for each in range(args.num_classes)}
            
    with torch.no_grad():
        for sample_idx, input, label, path  in tqdm(trainSet, total=len(trainSet)):
            model.zero_grad()
            input = input.unsqueeze(0).cuda()
            label = torch.tensor(label).type(torch.LongTensor).unsqueeze(0).cuda()
            pred  = model(input)
            loss = criterion_CE(pred,label)
            # loss.backward()
                
            label_key = label.item()
            # encoder = nn.Sequential(*list(model.children())[:-1])
            # feat = encoder(input).flatten(0).detach().cpu()

            loss_dict[label_key]['instance'].append(input.cpu().numpy())
            loss_dict[label_key]['path'].append(path)
            loss_dict[label_key]['loss'].append(loss.item())
    
    # Extracting Top K samples based on class-wise criterion
    for each_label in loss_dict.keys():
        instance_list = np.array(loss_dict[each_label]['instance'])
        path_list     = np.array(loss_dict[each_label]['path'])
        loss_list     = np.array(loss_dict[each_label]['loss'])

        sorted_indexs = np.argsort(loss_list)

        instance_sorted = instance_list[sorted_indexs]
        path_sorted = path_list[sorted_indexs]

        os.makedirs(f'{args.save_image_root}/top_k_samples/{args.exp}-{args.pct}-{args.etc}/{each_label}', exist_ok=True)
        for idx, each_path in enumerate(path_sorted[::-1][:args.K]):
            shutil.copy(each_path, f'{args.save_image_root}/top_k_samples/{args.exp}-{args.pct}-{args.etc}/{each_label}')