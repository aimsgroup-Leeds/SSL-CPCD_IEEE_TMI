from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score,recall_score,cohen_kappa_score
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models.resnet import resnet50

net_weight = 'model_best/'


use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#data_dir_train= r'dataset_new/train'

data_dir_test = r'data_cla/test'
data_dir_train=r'data_cla/train'
random_seed= 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

batch_size = 16
input_size = 224
class_num = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        print(original_features.size())
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
        
        #self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        #self.transion_layer = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
        #self.pool_layer = nn.MaxPool2d(32)  
        self.Linear_layer = nn.Linear(2048, 4)
        
    def forward(self, x):
        x = self.resnet_layer(x)
        #x = self.transion_layer(x)
 
        #x = self.pool_layer(x)
        
        x = x.view(x.size(0), -1) 
        
        x = self.Linear_layer(x)
        
        return x

def loaddata_test(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]),

    }

    image_datasets = datasets.ImageFolder(os.path.join(data_dir_test), data_transforms['test'])
    
    
    test_loader =torch.utils.data.DataLoader(image_datasets,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=1,)
    
    return test_loader

def loaddata_train(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
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
    split = int(np.floor(0.2 * num_train))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, sampler=train_sampler,
                    num_workers=1,)
    val_loader = torch.utils.data.DataLoader(image_datasets_val, batch_size=batch_size, sampler=valid_sampler,
                    num_workers=1,)

    return train_loader, val_loader

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

def test_model(model, criterion, device):
    test_loader = loaddata_test(data_dir=data_dir_test, batch_size=1, set_name='test', shuffle=False)
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_2 = 0
    
    model.eval()
    
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            
            data, label = data.to(device), label.to(device)
            y_pred = model(data)
            loss = criterion(y_pred, label)
            acc_1, acc_2 = calculate_topk_accuracy(y_pred, label)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_2 += acc_2.item()
        
    epoch_loss /= len(test_loader)
    epoch_acc_1 /= len(test_loader)
    epoch_acc_2 /= len(test_loader)
    
    
    return epoch_loss, epoch_acc_1, epoch_acc_2

def train_model(model, criterion, device):
    train_loader, val_loader = loaddata_train(data_dir=data_dir_train, batch_size=16, set_name='train', shuffle=True)
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_2 = 0
    
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            y_pred = model(data)
            loss = criterion(y_pred, label)
            acc_1, acc_2 = calculate_topk_accuracy(y_pred, label)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_2 += acc_2.item()
        
    epoch_loss /= len(train_loader)
    epoch_acc_1 /= len(train_loader)
    epoch_acc_2 /= len(train_loader)
        
    return epoch_loss, epoch_acc_1, epoch_acc_2

def val_model(model, criterion, device):
    train_loader, val_loader = loaddata_train(data_dir=data_dir_train, batch_size=16, set_name='train', shuffle=True)
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_2 = 0
    
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            data, label = data.to(device), label.to(device)
            y_pred = model(data)
            loss = criterion(y_pred, label)
            acc_1, acc_2 = calculate_topk_accuracy(y_pred, label)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_2 += acc_2.item()
        
    epoch_loss /= len(val_loader)
    epoch_acc_1 /= len(val_loader)
    epoch_acc_2 /= len(val_loader)
        
    return epoch_loss, epoch_acc_1, epoch_acc_2

def test_label_predictions(model, device):
    test_loader= loaddata_test(data_dir=data_dir_test, batch_size=16, set_name='test', shuffle=False)
    test_preds = []
    labels = []
    with torch.no_grad():
        model.eval()

        for i, batch in enumerate(test_loader):
            img, label = batch
            img, label = img.to(device, dtype = torch.float), label.to(device, dtype = torch.long)
            output = model(img)
            output = output.detach().cpu().numpy()
            test_preds.extend(np.argmax(output, 1))
            labels.extend(label.detach().cpu().numpy())
        return labels, test_preds

def sen(Y_test,Y_pred,n):
    
    sen = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
    a=(sen[0]+sen[1]+sen[2])/3    
    return a

def spe(Y_test,Y_pred,n):
    
    spe = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    a=(spe[0]+spe[1]+spe[2])/3
    return a

model = Network()
model=Net(model)
model=torch.load(net_weight)
print('Load weights sucessful!')

print('-' * 10)

criterion = nn.CrossEntropyLoss().cuda()

start_time = time.time()
epoch_loss_test, epoch_acc_1_test, epoch_acc_2_test=test_model(model, criterion, device)
epoch_loss_train, epoch_acc_1_train, epoch_acc_2_train=train_model(model, criterion, device)
epoch_loss_val, epoch_acc_1_val, epoch_acc_2_val=val_model(model, criterion, device)

end_time = time.time()
time = end_time-start_time

print('Top1_train: ',epoch_acc_1_train)
print('Top1_val: ',epoch_acc_1_val)

print('Loss_test: ',epoch_loss_test)
print('Top1_test: ',epoch_acc_1_test)
print('Top2_test: ',epoch_acc_2_test)
print('time: ',time)

labels, test_preds  = test_label_predictions(model, device)
print('Confusion matrix:')
print(confusion_matrix(labels, test_preds))

print('QWK: %f' %cohen_kappa_score(labels, test_preds,weights="quadratic"))
print('Specificity: %f' %spe(labels, test_preds, 3))
print('Sensitivity: %f' %sen(labels, test_preds, 3))
#print('Precision: %f' % precision_score(labels, test_preds, average='macro'))

#print('Recall: %f' % recall_score(labels, test_preds, average='macro'))
print('F1 score: %f' % f1_score(labels, test_preds, average='macro'))                                     
