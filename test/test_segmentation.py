import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from unet.unet import Unet

from utils.dataloader import test_dataset
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=512, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./checkpoints/')
    opt = parser.parse_args()
    model = Unet(backbone_name=resnet50, pretrained=False, classes = 2)
    model.load_state_dict(torch.load(opt.pth_path))
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    model.cuda()
    model.eval()
    for _data_name in ['kvasir_test', 'cvc']:

        ##### put data_path here #####
        data_path = './dataset/{}'.format(_data_name)
        ##### save_path #####
        save_path = './result_map/CPCD/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 512)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)
        print(_data_name, 'Finish!')
