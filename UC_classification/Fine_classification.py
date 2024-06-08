from __future__ import print_function
import os
import torch
from torch.utils import data, EarlyStopping
import torch.nn.functional as F
import torchvision
import numpy as np
import random
import time
import torch.optim as optim
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import plot_confusion_matrix
from torchvision.models.resnet import resnet50
import argparse

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir_train", type=str, default='data_cla/train',
                        help="path to Dataset")
    parser.add_argument("--data_dir_test", type=str, default='data_cla/test',
                        help="path to Dataset")
    parser.add_argument("--input_size", type=str, default=224,
                        help="input size")
    parser.add_argument("--max_epoch", type=str, default=200,
                        help="epoc")
    parser.add_argument("--lr", type=str, default=1e-4,
                        help="lr")
    parser.add_argument("--model_name", type=str, default='resnet50_cpcd',
                        help="name")
    parser.add_argument("--net_weight", type=str, default='./SSL-CPCD/jigsaw_models/epoch_2000',
                        help="ssl weight")
    parser.add_argument("--save_dir", type=str, default='model',
                        help="save path")
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')

random_seed= 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.network = resnet50()
        self.network = torch.nn.Sequential(*list(self.network.children())[:-1])
        self.projection_original_features = nn.Linear(2048, 128)
        self.connect_patches_feature = nn.Linear(1152, 128)

    def forward_once(self, x):
        return self.network(x)

    def return_reduced_image_features(self, original):
        original_features = self.forward_once(original)
        original_features = original_features.view(-1, 2048)
        original_features = self.projection_original_features(original_features)
        return original_features

    def return_reduced_image_patches_features(self, original, patches):
        original_features = self.return_reduced_image_features(original)

        patches_features = []
        for i, patch in enumerate(patches):
            patch_features = self.return_reduced_image_features(patch)
            patches_features.append(patch_features)

        patches_features = torch.cat(patches_features, axis=1)

        patches_features = self.connect_patches_feature(patches_features)
        return original_features, patches_features

    def forward(self, images=None, patches=None, mode=0):
        '''
        mode 0: get 128 feature for image,
        mode 1: get 128 feature for image and patches       
        '''
        if mode == 0:
            return self.return_reduced_image_features(images)
        if mode == 1:
            return self.return_reduced_image_patches_features(images, patches)

class Net(nn.Module):
    def __init__(self , model):	
        super(Net, self).__init__()
        
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        
        
        self.Linear_layer = nn.Linear(2048, 3)
        
    def forward(self, x):
        x = self.resnet_layer(x)
 
        x = x.view(x.size(0), -1) 
 
        x = self.Linear_layer(x)
        
        return x

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path + "/" + name + '_' + str(iter_cnt) + '.pth')
    torch.save(model, save_name)
    return save_name

def calculate_topk_accuracy(y_pred, y, k = 2):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].view(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k
def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch=20):
    
    lr = init_lr * (0.9**(epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def loaddata(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(input_size, padding = 10),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]),
        'val': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]),
        }

    image_datasets = datasets.ImageFolder(os.path.join(data_dir_train), data_transforms['train'])
    image_datasets_val = datasets.ImageFolder(os.path.join(data_dir_train), data_transforms['val'])

    num_train=len(image_datasets)
    indices = list(range(num_train))
    split = int(np.floor(0.3 * num_train))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    
    train_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, sampler=train_sampler,
                    num_workers=2,)
    val_loader = torch.utils.data.DataLoader(image_datasets_val, batch_size=batch_size, sampler=valid_sampler,
                    num_workers=2,)
    
    return train_loader,val_loader

def train(model, train_loader, optimizer, criterion, device, epoch):  
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_2 = 0
    model.train()
    tims = time.time()
    for i, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device).long()
        y_pred = model(data)
        
        loss = criterion(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #calculate accuracy both top1% and top2%
        acc_1, acc_2 = calculate_topk_accuracy(y_pred, label)
        
        # accumulate them to display the mean for each epoch
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_2 += acc_2.item()
        
        # if you want to display here as mean for after some iterations then please append them!
        
        
    epoch_loss /= len(train_loader)
    epoch_acc_1 /= len(train_loader)
    epoch_acc_2 /= len(train_loader)
    
    return epoch_loss, epoch_acc_1, epoch_acc_2

def validate(model, val_loader, criterion, device):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_2 = 0
    
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            data, label = data.to(device), label.to(device).long()
            output = model(data)
            
            loss = criterion(output, label)
            acc_1, acc_2 = calculate_topk_accuracy(output, label)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_2 += acc_2.item()
        
    epoch_loss /= len(val_loader)
    epoch_acc_1 /= len(val_loader)
    epoch_acc_2 /= len(val_loader)
        
    return epoch_loss, epoch_acc_1, epoch_acc_2


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == '__main__':
    
    if args['early_stopping']:
        print('INFO: Initializing early stopping')
        early_stopping = EarlyStopping()

    device = torch.device("cuda")
    best_train_acc= 0.0
    
    best_valid_acc1=0.0
    best_valid_acc2=0.0
    

    model = Network()
    model.load_state_dict(torch.load(net_weight))
    model=Net(model)
    print(model)
    
    train_loader,val_loader= loaddata(data_dir=data_dir_train, batch_size=32, set_name='train', shuffle=True)
    
    
    print('{} train iters per epoch:'.format(len(train_loader)))

    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr,
                          weight_decay=0.0004)
    model = model.to(device)
    criterion = criterion.to(device)

    # start = time.time()
    from torch.utils.tensorboard import SummaryWriter
    save_dir='./Tensorboard'+'/'+ model_name
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)
    
    for epoch in range(max_epoch):
        start_time = time.time()
        #scheduler.step()
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=lr)
        
        # call train function
        train_loss, train_acc_1, train_acc_2 = train(model, train_loader, optimizer, criterion, device, epoch)
    
        # call validation function
        valid_loss, valid_acc_1, valid_acc_2 = validate(model, val_loader, criterion, device)
    
        #save your checkpoint if best
        if args['early_stopping']:
            early_stopping(valid_loss)
            if early_stopping.early_stop:
                break

        #if  valid_acc_1 > best_valid_acc1:
        #    best_valid_acc1 = valid_acc_1
        #    best_valid_acc2 =  valid_acc_2
            
        os.makedirs(save_dir, exist_ok=True)
        save_model(model, save_dir, model_name, epoch)
        print('save model')

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t Train Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
              f'Train Acc @2: {train_acc_2*100:6.2f}%')
        print(f'\t Valid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
              f'Valid Acc @2: {valid_acc_2*100:6.2f}%')
        print(f'\t best val acc1: {best_valid_acc1*100:.3f} | best val acc2: {best_valid_acc2*100:6.2f}% | ')
        
        
        writer.add_scalar('/loss/train_loss', train_loss, epoch)
        writer.add_scalar('/loss/val_loss', valid_loss, epoch)
        writer.add_scalar('/acc/train_acc', train_acc_1, epoch)
        writer.add_scalar('/acc/val_acc', valid_acc_1, epoch)
