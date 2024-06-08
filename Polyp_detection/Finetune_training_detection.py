import argparse
import collections

import numpy as np
import os

import torch
import torch.optim as optim
from torchvision import transforms
# 
from retinanet import model

# from retinanet import model_updated as model

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))

os.environ["CUDA_VISIBLE_DEVICES"]='2'
device = torch.device('cuda')

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', default='kvasir_train.csv', help='Path to file containing training annotations')
    parser.add_argument('--csv_classes',default='kvasir_map.csv', help='Path to file containing class list')
    parser.add_argument('--csv_val', default='kvasir_val.csv', help='Path to file containing validation annotations')
    # parser.add_argument('--csv_val', default=None, help='Path to file containing validation annotations (optional, see readme)')
    #resnet 50
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=400)
    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':
        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':
        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')
        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')
            
#        print(parser.csv_train)
#        print(parser.csv_classes)
        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

# set batcch and worker
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=32, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
        #retinanet = model.resnet50_cbam(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 502:
        retinanet = model.resnext50_32x4d( num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152, 502')
        
        
    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    net_weight = './SSL-CPCD/jigsaw_models/epoch_2000'
    retinanet .load_state_dict(torch.load(net_weight), strict=False)
    
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=400)
    retinanet.train()
    
    save_weights_folder = 'kvasir_polyp_{}_cpcd'.format(parser.depth)
    os.makedirs(save_weights_folder, exist_ok=True)
    print('Num training images: {}'.format(len(dataset_train)))
    
    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        if np.mod(epoch_num, 50) == 0: 
            
            torch.save(retinanet.module, os.path.join(save_weights_folder,'ResNet{}_retinanet_{}_cpcd.pt'.format(parser.depth, epoch_num)))

    retinanet.eval()
    torch.save(retinanet, os.path.join(save_weights_folder, 'model_final.pt'.format(epoch_num)))
    
if __name__ == '__main__':
    main()
