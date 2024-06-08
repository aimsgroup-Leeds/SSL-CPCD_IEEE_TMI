
def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return []
    
def read_txt_file(txtfile):
    
    import numpy as np
    lines = []
    
    with open(txtfile, "r") as f:
        
        for line in f:
            line = line.strip()
            lines.append(line)
        
    return np.array(lines)
    
def read_obj_names(textfile):
    classnames = []
    with open(textfile) as f:
        for line in f:
            line = line.strip('\n')
            if len(line)>0:
                classnames.append(line)
    return np.hstack(classnames)
    
def write_bbox(filename, detections):
    with open(filename, 'w') as f:
        for det in detections:
            f.write('\t'.join([det[0], str(det[1]), str(det[2]), str(det[3]), str(det[4]), str(det[5])])+'\n')
            
    return []
   
def write_bbox_annot(filename, detections):
    
    with open(filename, 'w') as f:
        for det in detections:
            f.write('\t'.join([det[0], str(det[1]), str(det[2]), str(det[3]), str(det[4])])+'\n')
            
    return []    

def load_image(image_file):
    import skimage
    img = skimage.io.imread(image_file)

    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    return img.astype(np.float32)/255.0
  
def normalizer_(img):

    mean_ = np.array([[[0.485, 0.456, 0.406]]])
    std_ = np.array([[[0.229, 0.224, 0.225]]])

    return (img.astype(np.float32)-mean_)/std_

    
def detect_imgs(infolder, ext='.tif'):
    import os

    items = os.listdir(infolder)

    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)
    
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import pdb
import time
from torch.utils.data import Dataset, DataLoader

assert torch.__version__.split('.')[0] == '1'
#/well/rittscher/projects/felix-Sharib/pytorch-retinanet_EDD2020
import sys
import cv2
import pylab as plt 
# from scipy.misc import imsave


 
import seaborn as sns

cmap = sns.color_palette('hls', 8)
#print('CUDA available: {}'.format(torch.cuda.is_available()))

os.environ["CUDA_VISIBLE_DEVICES"]='2'
device = torch.device('cuda')

coco = False

model = torch.load('./kvasir_polyp_50/')
use_gpu = True

if use_gpu:
    model = model.cuda()
model.eval()


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    
savepredfolder = 'kvasir_results_CPCD_scaled_box'
mkdir(savepredfolder)


class_names=['polyp']
class_labels= read_obj_names('class_kvasir.txt')


validationDataArray = read_txt_file('kvasir_valid.txt')

import matplotlib.pyplot as plt
from skimage.transform import resize

for i in range (len(validationDataArray)):
#for imagePath in imgfiles[:]:
    """
    Grab the name of the file. 
    """
    imagePath=validationDataArray[i]
    filename = (imagePath.split('/')[-1]).split('.jpg')[0]
#    filename =
    print('filename is printing::=====>>', filename)
    img = load_image(imagePath)
    
    h, w, _ = img.shape
    print('shape of img:',  img.shape)
    
    # if ((h%2!=0) and (w%2==0)):
    #     img1 = resize(img1, (img1.shape[0]+1, img1.shape[1], 3))

    # elif ((w%2!=2) and (h%2==0)):
    #     img1 = resize(img1, (img1.shape[0], img1.shape[1]+1, 3))
    # elif ((h%2!=0) and (w%2!=0)):
    #     img1 = resize(img1, (img1.shape[0]+1, img1.shape[1]+1, 3))

    scaley = img.shape[0]/512
    scalex = img.shape[1]/512
    
    img1 = resize(img, (512, 512, 3))  
    
    
    print('new shape of img:',  img1.shape)
    img1 = normalizer_(img1)
    img_A = np.transpose(img1, (2,0,1))
    
    with torch.no_grad():
        st = time.time()
        data_=torch.FloatTensor(img_A).unsqueeze(0)
        print('shape {}, {}'.format(img_A.shape[1], img_A.shape[2]))
        scores, classification, transformed_anchors = model(data_.cuda())

        idxs = np.where(scores.cpu()>0.5)
        print('Elapsed time: {}'.format(time.time()-st))
        # img = np.array(255 * unnormalize(data_[0, :, :, :])).copy()
#
        img[img<0] = 0
        img[img>255] = 255

        """
        Draw the bbox onto the image.
        """
        # img = np.transpose(img, (1, 2, 0))
        img = img*255.0
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        """
        Grab the annotatin boxes and write out in a diff folder.
        """
        bboxes = []
#  0.688 (ResNet101), 
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0]*scalex)
            y1 = int(bbox[1]*scaley)
            x2 = int(bbox[2]*scalex)
            y2 = int(bbox[3]*scaley)

            label_name = class_names[int(classification[idxs[0][j]])]
            print(label_name)
            score = scores[idxs[0]][j].item()
            draw_caption(img, (x1, y1, x2, y2), label_name)

            lab_num = int(np.arange(len(class_labels))[class_labels==label_name])
            box_color = (np.array(cmap[lab_num])*255).astype(np.int)
            mytuple = tuple(map(int,box_color))
            cv2.rectangle(img, (x1, y1), (x2, y2), mytuple, thickness=8)

            """
            Get the confidence of the classification.
                return:
                    class_name, conf, x1, y1, x2, y2 (save as format voc or save as yolo?)
            """
            bboxes.append([label_name, score, x1, y1, x2, y2])

#            print(label_name)

        if len(bboxes) > 0:
            bboxes = np.array(bboxes)


        """
        Save out the boxes + accompanying image.
        """
        # imsave(os.path.join(savepredfolder, filename+'.jpg'), img[:,:,::-1])
        
        cv2.imwrite(os.path.join(savepredfolder, filename+'.jpg'),(img))
        # [:,:,[2,1,0]]
        write_bbox(os.path.join(savepredfolder, filename+'.txt'), bboxes)
