import os
import sys
from turtle import pos
sys.path.append('/home/zby/Cellseg')
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor,ColorJitter
from datasets.utils import *

class CellSeg(Dataset):
    def __init__(self, args, mode=True):
        super(CellSeg,self).__init__()
        # Import the data paths
        self.image_dir = args.image_dir
        self.anno_dir = args.anno_dir
        self.split_info = json.load(open(args.split_info,'r'))

        # Test dataset
        self.test_image_dir = args.test_image_dir
        self.test_anno_dir = args.test_anno_dir

        # Initialize the training list
        self.using_unlabel = False
        if mode=='train':
            image_names = [name+'.png' for name in self.split_info['train']]
            anno_names = [image_name for image_name in image_names]
            self.img_list = [os.path.join(self.image_dir, image_name) for image_name in image_names]
            self.anno_list = [os.path.join(self.anno_dir, anno_name) for anno_name in anno_names]
        elif mode=='val':
            image_names = [name+'.png' for name in self.split_info['val']]
            anno_names = [image_name for image_name in image_names]
            self.img_list = [os.path.join(self.image_dir, image_name) for image_name in image_names]
            self.anno_list = [os.path.join(self.anno_dir, anno_name) for anno_name in anno_names]
        elif mode=='all':
            image_names = [name+'.png' for name in self.split_info['train']] + [name+'.png' for name in self.split_info['val']]
            anno_names = [image_name for image_name in image_names]
            self.img_list = [os.path.join(self.image_dir, image_name) for image_name in image_names]
            self.anno_list = [os.path.join(self.anno_dir, anno_name) for anno_name in anno_names]
        elif mode=='val_all':
            image_names = [name+'.png' for name in self.split_info['train']] + [name+'.png' for name in self.split_info['val']]
            anno_names = [image_name for image_name in image_names]
            self.img_list = [os.path.join(self.image_dir, image_name) for image_name in image_names]
            self.anno_list = [os.path.join(self.anno_dir, anno_name) for anno_name in anno_names]
        elif mode=='test':  # for TuningSet
            image_names = sorted(os.listdir(self.test_image_dir))
            anno_names = [image_name for image_name in image_names]
            self.img_list = [os.path.join(self.test_image_dir, image_name) for image_name in image_names]
            self.anno_list = [os.path.join(self.test_anno_dir, anno_name) for anno_name in anno_names]
        else:
            raise NotImplementedError()
        
        # Initialize the pre-processing setting
        self.mode=mode
        self.scale_range = args.scale_range
        self.crop_size = args.crop_size
        self.flip = args.rand_flip
        self.rotate = args.rand_rotate
        self.bright = args.rand_bright
        self.contrast = args.rand_contrast
        self.saturation = args.rand_saturation
        self.hue = args.rand_hue
        self.test_multi_scale = args.test_multi_scale
        self.transform = ToTensor()
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        anno_path = self.anno_list[index]
        img = cv2.imread(img_path)
        if os.path.exists(anno_path):
            anno = cv2.imread(anno_path,flags=0)
            valid_label = 1
        else:
            anno = np.zeros_like(img[:,:,0])
            valid_label = 0
        img_meta = {'img_path': img_path, 'valid_label': valid_label, 'ori_shape':img.shape}
        
        if self.mode in ['train', 'all']:
            if self.scale_range:
                img, anno = random_scale(img, anno, self.scale_range, self.crop_size)
            if self.crop_size:
                img, anno = random_crop(img, anno, self.crop_size)
            if self.flip:
                img, anno = random_flip(img, anno, self.flip)
            if self.rotate:
                img, anno = random_rotate(img, anno)

            img = self.transform(img)
            img = ColorJitter(brightness=self.bright, contrast=self.contrast,saturation=self.saturation,hue=self.hue)(img)
            anno = torch.tensor(anno)
            
        elif self.mode in ['val']:
            img, anno, valid_region = multi_scale_test(img, anno, scale=[1.0], crop_size=self.crop_size)
            img_meta['valid_region'] = valid_region[0]
            img = self.transform(img[0])
            anno = torch.tensor(anno[0])
            
        else:
            img, anno, valid_region = multi_scale_test(img, anno, scale=self.test_multi_scale, crop_size=self.crop_size)
            img_meta['valid_region'] = valid_region
            if isinstance(img, list):
                img = [self.transform(i) for i in img]
                anno = [torch.tensor(a) for a in anno]
            else:
                img = self.transform(img)
                anno = torch.tensor(anno)
                
        return img, anno, img_meta

    def __len__(self):
        return len(self.img_list)
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser("CellSeg training argument parser.")
    parser.add_argument('--image_dir',default='./data/fix_boundary/images')
    parser.add_argument('--anno_dir',default='./data/fix_boundary/labels')
    parser.add_argument('--test_image_dir',default='./data/Val_Labeled_3class/images')
    parser.add_argument('--test_anno_dir',default='./data/Val_Labeled_3class/labels')
    parser.add_argument('--split_info', default='/home/zby/Cellseg/data/split_info.json')
    
    parser.add_argument('--scale_range', default=(0.5,2.0))
    parser.add_argument('--crop_size', default=(512,512))
    parser.add_argument('--rand_flip', default=0.5, help="Horizonal and Vertical filp, 0 for unchange")
    parser.add_argument('--rand_rotate', default=False, type=bool)
    args = parser.parse_args()
    
    train_dataset = CellSeg(args,mode='train')
    val_dataset = CellSeg(args,mode='val')
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    for ii, item in enumerate(train_dataloader):
        print("The {}/{} batches...".format(ii,len(train_dataloader)),end='\r')
    for ii, item in enumerate(val_dataloader):
        print("The {}/{} batches...".format(ii,len(val_dataloader)),end='\r')