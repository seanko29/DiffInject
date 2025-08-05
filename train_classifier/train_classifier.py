"""
2022-07-12
Part1: Training biased classfiers & Extracting bias-conflict samples with methodology
This implementation is for understaning the overall structure of our methods.

- biased classifier: 
    python train_classifier.py --pct 0.5pct
- unbiased classifier (random): 
    python train_classifier.py --synthetic_root /workspace/mnc-research/AmpliBias-main/results/diffstyle/bffhq/0.5pct/ffhq_p2_inv1000_gamma07 --pct 0.5pct --bias_conflict_ratio 0.1
- unbiased classifier (stratified): 
    python train_classifier.py --synthetic_root /workspace/mnc-research/AmpliBias-main/results/diffstyle/bffhq/0.5pct/ffhq_p2_inv1000_gamma07 --pct 0.5pct --bias_conflict_ratio 0.1 --stratified_sampling True
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

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


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
    def __init__(self, path, args, transform=None, mode='valid', meta_dict=None):
        super(BaseDataset, self).__init__()
        self.path = path
        self.args = args
        self.transform = transform
        self.meta_dict = meta_dict
        self.mode = mode

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        origin_path  = self.path[idx]

        if self.mode == 'train':
            if self.meta_dict[origin_path] == 'real': 
                if self.args.exp == 'new-cmnist' or self.args.exp == 'cifar10c':
                    label = int(origin_path.split('_')[-2]) 
                else:
                    label = int(origin_path.split('_')[-2])
            #! be careful with defining label for synthetic dataset 
            else: 
                if self.args.exp == 'new-cmnist' or self.args.exp == 'cifar10c':
                    label = int(origin_path.split('_')[-2]) # style
                else:
                    label = int(origin_path.split('_')[-6]) # content       
        else: 
            if self.args.exp == 'new-cmnist' or self.args.exp == 'cifar10c':
                label = int(origin_path.split('_')[-2]) 
            else:
                label = int(origin_path.split('_')[-2])
            

        if self.mode == 'train': 
            if self.meta_dict[origin_path] == 'real': 
                image = Image.open(origin_path).convert('RGB')
            else: 
                image = Image.open(origin_path).convert('RGB')
                # print('Before: ', image.size)
                width = height = image.size[0]-2 # remove padding on both sides 
                x_start, y_start = 1, 2*(image.size[1]-4)//3 + 3 
                image = image.crop((x_start, y_start, x_start + width, y_start + height)) 
                # print('After: ', image.size)
                # image.save('./test.png', format="png")
        else: 
            image = Image.open(origin_path).convert('RGB')
            # print(image.size)
        
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

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--root", type=str, default=os.path.join(ROOT_PROJECT_DIR, 'data'), help="Dataset root")
    parser.add_argument("--synthetic_root", type=str, default='', help="Synthetic dataset root")
    parser.add_argument("--save_root", type=str, default=os.path.join(PROJECT_DIR, 'results'), help="where the model was saved")

    parser.add_argument("--exp", type=str, default='bffhq', help="Dataset name")     # new-cmnist/cifar10c/bffhq/bar
    parser.add_argument("--data_type", type=str, default='bffhq', help='kind of data used')
    parser.add_argument("--pct", type=str, default="5pct", help="Percent name")
    parser.add_argument("--etc", type=str, default='vanilla', help="Experiment name")
    # parser.add_argument("--loss", type=str, required=True)          # GCE || CE
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--scheduler",  action='store_true', help='Using scheduler')
    parser.add_argument("--pretrained", action='store_true', help='Using Imagenet Pretrained')
    
    #! early stopping patience
    parser.add_argument("--early_stopping_patience", type=bool, default=False)
    
    #! bias_conflict_ratio
    parser.add_argument("--bias_conflict_ratio", type=float, default=0.0)    
    parser.add_argument("--stratified_sampling", type=bool, default=False)
    
    args = parser.parse_args()
    set_args(args)

    root = args.root
    args.image_size = (args.w, args.h)
    args.image_shape = (3, args.w, args.h)
    
    args.lr_decay_rate = 0.1
    args.lr_decay_schedule = [40, 60, 80]
    
    args.data_root = f'{root}/{args.exp}/'

    if args.synthetic_root: 
        if args.stratified_sampling:
            args.save_root = f'{args.save_root}/synthetic/{args.exp}-{args.pct}-{args.etc}-{args.bias_conflict_ratio}-stratified/'
        else:
            args.save_root = f'{args.save_root}/synthetic/{args.exp}-{args.pct}-{args.etc}-{args.bias_conflict_ratio}-random/'
    else: 
        args.save_root = f'{args.save_root}/pretrained/{args.exp}-{args.pct}-{args.etc}/'

    model = call_by_name(args)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion_GCE = GeneralizedCELoss().cuda() # GCE Loss
    criterion_CE  = nn.CrossEntropyLoss().cuda() # GCE Loss

    ### BUILD DATASET
    train_align    = [y for x in os.walk(os.path.join(args.data_root, args.pct, 'align')) for y in glob(os.path.join(x[0], '*/*.png'))]
    train_conflict = [y for x in os.walk(os.path.join(args.data_root, args.pct, 'conflict')) for y in glob(os.path.join(x[0], '*/*.png'))]

    train_data = train_align + train_conflict
    print('Original: ', len(train_data))
    train_meta_dict = {data: 'real' for data in train_data} 
    
    # synthetic dataset created using diffstyle
    if args.synthetic_root:
        total_synthetic_data = [y for y in glob(os.path.join(args.synthetic_root, '*.png'))]
        
        if args.stratified_sampling:
            synthetic_data = list()
            num_stratified_samples = int(args.bias_conflict_ratio * len(train_align) // args.num_classes)
            
            for class_i in range(args.num_classes): 
                if args.exp == 'new-cmnist' or args.exp == 'cifar10c':
                    total_stratified_synthetic_data = [y for y in glob(os.path.join(args.synthetic_root, f'content_*_style_*_{class_i}_*.png'))]
                    print(len(total_stratified_synthetic_data), total_stratified_synthetic_data[:3])
                    stratified_synthetic_data = list(np.random.choice(total_stratified_synthetic_data, num_stratified_samples))
                    # label = int(origin_path.split('_')[-2]) # style
                else:
                    total_stratified_synthetic_data = [y for y in glob(os.path.join(args.synthetic_root, f'content_*_{class_i}_*_style_*.png'))]
                    print(len(total_stratified_synthetic_data), total_stratified_synthetic_data[:3])
                    stratified_synthetic_data = list(np.random.choice(total_stratified_synthetic_data, num_stratified_samples))
                    # label = int(origin_path.split('_')[-6]) # content           
                synthetic_data.extend(stratified_synthetic_data)
        else: 
            synthetic_data = list(np.random.choice(total_synthetic_data, int(args.bias_conflict_ratio * len(train_align)))) # total_synthetic_data[:int(args.bias_conflict_ratio * len(total_synthetic_data))]
        
        print(f'Sampling {len(synthetic_data)} out of {len(total_synthetic_data)} ...')
        train_data = train_data + synthetic_data
        train_meta_dict.update({data: 'synthetic' for data in synthetic_data})
 
    if args.exp == 'bffhq':
        valid_data = [y for x in os.walk(os.path.join(args.data_root, 'valid')) for y in glob(os.path.join(x[0], '*.png'))] 
    if args.exp == 'cifar10c':
        valid_data = [y for x in os.walk(os.path.join(args.data_root, args.pct, 'valid')) for y in glob(os.path.join(x[0], '*.png'))]
    test_data  = [y for x in os.walk(os.path.join(args.data_root, 'test')) for y in glob(os.path.join(x[0], '*.png'))]
    
    # print(args.pct)
    print(len(train_data), 'train length')
    print(len(valid_data), 'valid length')
    if args.exp == 'new-cmnist' or args.exp == 'cifar10c':
        label_attr = np.array([int(each.split('_')[-2]) for each in test_data])
        bias_attr = np.array([int(each.split('_')[-1][0]) for each in test_data])
    else:
        label_attr = np.array([int(each.split('_')[-1][0]) for each in test_data])
        bias_attr = np.array([int(each.split('_')[-2]) for each in test_data])
    test_align = np.array(test_data)[label_attr == bias_attr]
    test_conflict = np.array(test_data)[label_attr != bias_attr]

    train_transform, valid_transform = get_transform(args)
    trainSet = BaseDataset(train_data, args, transform=train_transform, mode='train', meta_dict=train_meta_dict)
    validSet = BaseDataset(valid_data, args, transform=valid_transform, mode='valid')
    testSet_align = BaseDataset(test_align, args, transform=valid_transform, mode='test')
    testSet_conflict = BaseDataset(test_conflict, args, transform=valid_transform, mode='test')

    train_loader = torch.utils.data.DataLoader(trainSet, batch_size=args.batch, shuffle=True, drop_last=False, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(validSet, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=2)
    bias_test_loader  = torch.utils.data.DataLoader(testSet_align, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=2)
    unbias_test_loader  = torch.utils.data.DataLoader(testSet_conflict, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=2)

    res = {'train_accuracy':[],'train_loss':[], 'valid_accuracy':[],'valid_loss':[], 'bias_test_accuracy':[],'unbias_test_accuracy':[]}
    meter = CustomMeter(res)


    ###################################################
    ###################################################
    # Phase 1: Training Biased Classifier
    ###################################################
    ###################################################


    valid_best = 0
    for epoch in range(1, args.epoch+1):
        
        if args.scheduler:
            adjust_learning_rate(optimizer, epoch, args)
        
        # Training Process
        train_corr, train_loss = 0, 0
        model.train()
        for iter_idx, (sample_idx, inputs, labels, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if args.exp =='cmnist' or args.exp =='new-cmnist':
                inputs  = inputs.flatten(1).cuda(non_blocking=True)    
            else:
                inputs  = inputs.cuda(non_blocking=True)
            labels  = labels.cuda(non_blocking=True)

            outputs = model(inputs)
            loss = criterion_GCE(outputs, labels).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_corr += (outputs.argmax(1) == labels).float().sum().item()
            train_loss += loss.item()

        # Valid Process
        model.eval()
        with torch.no_grad():
            valid_corr, valid_loss = 0, 0
            for iter_idx, (_, inputs, labels, _) in enumerate(valid_loader):
                if args.exp =='cmnist' or args.exp =='new-cmnist':
                    inputs  = inputs.flatten(1).cuda(non_blocking=True)    
                else:
                    inputs  = inputs.cuda(non_blocking=True)
                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion_GCE(outputs, labels).mean()          

                valid_corr += (outputs.argmax(1) == labels).float().sum().item()
                valid_loss += loss.item()

        train_accuracy = train_corr/len(train_loader.dataset)
        valid_accuracy = valid_corr/len(valid_loader.dataset)
        print(f'Current valid accuracy: {valid_accuracy}')
        
        if valid_accuracy > valid_best:
            print(f'\t Best valid accuracy: {valid_best} -> {valid_accuracy}')
            valid_best = valid_accuracy
            save('best', epoch, args.save_root, model, False)
        save('last', epoch, args.save_root, model, False)

        # Test Process
        with torch.no_grad():
            bias_test_corr = 0
            for iter_idx, (_, inputs, labels, _) in enumerate(bias_test_loader):
                if args.exp =='cmnist' or args.exp =='new-cmnist':
                    inputs  = inputs.flatten(1).cuda(non_blocking=True)    
                else:
                    inputs  = inputs.cuda(non_blocking=True)
                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion_GCE(outputs, labels).mean()             
                bias_test_corr += (outputs.argmax(1) == labels).float().sum().item()
            bias_test_accuracy = bias_test_corr/len(bias_test_loader.dataset)

            unbias_test_corr = 0
            for iter_idx, (_, inputs, labels, _) in enumerate(unbias_test_loader):
                if args.exp =='cmnist' or args.exp =='new-cmnist':
                    inputs  = inputs.flatten(1).cuda(non_blocking=True)    
                else:
                    inputs  = inputs.cuda(non_blocking=True)
                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion_GCE(outputs, labels).mean()             
                unbias_test_corr += (outputs.argmax(1) == labels).float().sum().item()
            unbias_test_accuracy = unbias_test_corr/len(unbias_test_loader.dataset) 
        print(f'Current test accuracy: [{bias_test_accuracy}|{unbias_test_accuracy}]')

        # Stats on Board
        meter.update('train_accuracy', train_accuracy)
        meter.update('train_loss', train_loss/len(train_loader))
        meter.update('valid_accuracy', valid_accuracy)
        meter.update('valid_loss', valid_loss/len(valid_loader))
        meter.update('bias_test_accuracy', bias_test_accuracy)
        meter.update('unbias_test_accuracy', unbias_test_accuracy)
        meter.save(args.save_root, 'results')
