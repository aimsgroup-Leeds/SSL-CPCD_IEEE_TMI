import random

import numpy as np
import torch
#from torch._C import T
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.models.resnet import resnet50

from utils_SSL import (AverageMeter, Logger, Memory, ModelCheckpoint,
                   NoiseContrastiveEstimator, Progbar, pil_loader)

device = torch.device('cuda:0')
data_dir = r'data_pre/train'
negative_nb = 3000  # number of negative examples in NCE
lr = 0.001
checkpoint_dir = 'jigsaw_models'
log_filename = 'pretraining_log_jigsaw'


class JigsawLoader(DatasetFolder):
    def __init__(self, root_dir):
        super(JigsawLoader, self).__init__(root_dir, pil_loader, extensions=('jpg'))
        self.root_dir = root_dir
        self.color_transform = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        self.flips = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip()]
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        original = self.loader(path)
        image = torchvision.transforms.Resize((224, 224))(original)
        sample = torchvision.transforms.RandomCrop((255, 255))(original)

        crop_areas = [(i*85, j*85, (i+1)*85, (j+1)*85) for i in range(3) for j in range(3)]
        samples = [sample.crop(crop_area) for crop_area in crop_areas]
        samples = [torchvision.transforms.RandomCrop((64, 64))(patch) for patch in samples]
        # augmentation collor jitter
        image = self.color_transform(image)
        samples = [self.color_transform(patch) for patch in samples]
        # augmentation - flips
        image = self.flips[0](image)
        image = self.flips[1](image)
        # to tensor
        image = torchvision.transforms.functional.to_tensor(image)
        samples = [torchvision.transforms.functional.to_tensor(patch) for patch in samples]
        # normalize
        image = self.normalize(image)
        samples = [self.normalize(patch) for patch in samples]
        random.shuffle(samples)

        return {'original': image, 'patches': samples, 'index': index}


dataset = JigsawLoader(data_dir)
train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32, num_workers=2)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.network = resnet50(pretrained=True)
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
def KMeans(x, K=10, Niters=10, verbose=False):

    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = x[:, None, :]  # (Npoints, 1, D)

    for i in range(Niters):
        c_j = c[None, :, :]  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = cl.view(cl.size(0), 1).expand(-1, D)
        unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
        # As some clusters don't contain any samples, manually assign count as 1
        labels_count_all = torch.ones([K]).long().cuda()
        labels_count_all[unique_labels[:,0]] = labels_count
        c = torch.zeros([K, D], dtype=torch.float).cuda().scatter_add_(0, Ncl, x)
        c = c / labels_count_all.float().unsqueeze(1)

    return cl, c
def grouping(features_groupDis1, features_groupDis2, T):
    
    criterion = nn.CrossEntropyLoss().cuda()
    # K-way normalized cuts or k-Means. Default: k-Means
    
    cluster_label1, centroids1 = KMeans(features_groupDis1, K=num_clusters, Niters=num_iters)
    cluster_label2, centroids2 = KMeans(features_groupDis2, K=num_clusters, Niters=num_iters)
        

    # group discriminative learning
    affnity1 = torch.mm(features_groupDis1, centroids2.t())
    CPCD_loss = criterion(affnity1.div_(T), cluster_label2)

    affnity2 = torch.mm(features_groupDis2, centroids1.t())
    CPCD_loss = (CPCD_loss + criterion(affnity2.div_(T), cluster_label1))/2
    # print(CPCD_loss)
    return CPCD_loss


net = Network().to(device)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

memory = Memory(size=len(dataset), weight=0.5, device=device)
memory.initialize(net, train_loader)


checkpoint = ModelCheckpoint(mode='min', directory=checkpoint_dir)
noise_contrastive_estimator = NoiseContrastiveEstimator(device)
logger = Logger(log_filename)
Lambda=0.5
loss_weight = 0.5
num_clusters=4
num_iters=10

for epoch in range(2000):
    print('\nEpoch: {}'.format(epoch))
    memory.update_weighted_count()
    train_loss = AverageMeter('train_loss')
    bar = Progbar(len(train_loader), stateful_metrics=['train_loss', 'valid_loss'])


    for step, batch in enumerate(train_loader):

        # prepare batch
        images = batch['original'].to(device)
        patches = [element.to(device) for element in batch['patches']]
        index = batch['index']
        representations = memory.return_representations(index).to(device).detach()
        # zero grad
        optimizer.zero_grad()

        #forward, loss, backward, step
        output = net(images=images, patches=patches, mode=1)
        loss_1 = noise_contrastive_estimator(representations, output[1], index, memory, negative_nb=negative_nb)
        loss_2 = noise_contrastive_estimator(representations, output[0], index, memory, negative_nb=negative_nb)
        loss = loss_weight * loss_1 + (1 - loss_weight) * loss_2

        q0 = nn.functional.normalize(output[0], dim=1)
        q1 = nn.functional.normalize(output[1], dim=1)
        loss_cpcd= grouping(q0, q1, T=0.4)
        loss= (1-Lambda) * loss + Lambda * loss_cpcd
        loss.backward()
        optimizer.step()

        # update representation memory
        memory.update(index, output[0].detach().cpu().numpy())

        # update metric and bar
        train_loss.update(loss.item(), images.shape[0])
        bar.update(step, values=[('train_loss', train_loss.return_avg())])
    logger.update(epoch, train_loss.return_avg())

    # save model if improved
    checkpoint.save_model(net, train_loss.return_avg(), epoch)
