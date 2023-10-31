import os
import sys
from turtle import pos
sys.path.append('/home/zby/Cellseg')
import cv2
import json
import tifffile as tif
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor,ColorJitter
from datasets.utils import *

class WSCellSeg(Dataset):
    def __init__(self, args, mode='test'):
        super(WSCellSeg,self).__init__()
        # Import the data paths
        self.args = args
        self.data_root = args.data_root
        self.split_info = json.load(open(args.split_info,'r'))

        # Initialize the training list
        if mode == 'train_weak':
            self.img_list = [train_item['img_path'] for train_item in self.split_info['train']]
            self.anno_list = [train_item['weak_label_path'] for train_item in self.split_info['train']]
            self.gt_list = [train_item['gt_path'] for train_item in self.split_info['train']]
            self.semantic_list = [train_item['semantic_path'] for train_item in self.split_info['train']]
            self.heat_list = [train_item['weak_heat_path'] for train_item in self.split_info['train']]
            self.point_list = [train_item['point_path'] for train_item in self.split_info['train']]
        elif mode == 'train_full':
            self.img_list = [train_item['img_path'] for train_item in self.split_info['train']]
            self.anno_list = [train_item['full_label_path'] for train_item in self.split_info['train']]
            self.gt_list = [train_item['gt_path'] for train_item in self.split_info['train']]
            self.semantic_list = [train_item['semantic_path'] for train_item in self.split_info['train']]
            self.heat_list = [train_item['full_heat_path'] for train_item in self.split_info['train']]
            self.point_list = [train_item['point_path'] for train_item in self.split_info['train']]
        elif mode == 'val':
            self.img_list = [val_item['img_path'] for val_item in self.split_info['val']]
            self.anno_list = [val_item['full_label_path'] for val_item in self.split_info['val']]
            self.gt_list = [val_item['gt_path'] for val_item in self.split_info['val']]
            self.semantic_list = [val_item['semantic_path'] for val_item in self.split_info['val']]
            self.heat_list = [val_item['full_heat_path'] for val_item in self.split_info['val']]
            self.point_list = [val_item['point_path'] for val_item in self.split_info['val']]
        elif mode == 'val_train':
            self.img_list = [val_item['img_path'] for val_item in self.split_info['train']]
            self.anno_list = [val_item['full_label_path'] for val_item in self.split_info['train']]
            self.gt_list = [val_item['gt_path'] for val_item in self.split_info['train']]
            self.semantic_list = [val_item['semantic_path'] for val_item in self.split_info['train']]
            self.heat_list = [val_item['full_heat_path'] for val_item in self.split_info['train']]
            self.point_list = [val_item['point_path'] for val_item in self.split_info['train']]
        elif mode=='test':
            # Test dataset
            self.test_image_dir = args.test_image_dir
            self.test_anno_dir = args.test_anno_dir
            image_names = sorted(os.listdir(self.test_image_dir))
            anno_names = [image_name for image_name in image_names]
            self.img_list = [os.path.join(self.test_image_dir, image_name) for image_name in image_names]
            self.anno_list = [os.path.join(self.test_anno_dir, anno_name) for anno_name in anno_names]
            self.gt_list = self.anno_list
            self.semantic_list = self.anno_list
            self.heat_list = self.anno_list
            self.point_list = self.anno_list
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
        gt_path = self.gt_list[index]
        semantic_path = self.semantic_list[index]
        heat_path = self.heat_list[index]
        point_path = self.point_list[index]
        vor_path = anno_path.replace('weak','weak_distmap').replace('full','full_distmap')
        
        if img_path.endswith('.png'):
            img = cv2.imread(img_path)
        elif img_path.endswith('.tif') or img_path.endswith('.tiff'):
            img = tif.imread(img_path)
        
        anno = cv2.imread(anno_path,flags=0) if os.path.exists(anno_path) else np.ones_like(img[:,:,0])*255
        gt = tif.imread(gt_path).astype(np.int16) if os.path.exists(gt_path) else np.zeros_like(img[:,:,0]).astype(np.int16)
        semantic = cv2.imread(semantic_path,flags=0) if os.path.exists(semantic_path) else np.zeros_like(img[:,:,0])
        heat = cv2.imread(heat_path, flags=0)/255 if os.path.exists(heat_path) else np.zeros_like(img[:,:,0])
        point_dict = json.load(open(point_path, 'r')) if os.path.exists(point_path) else None
        vor = cv2.imread(vor_path, flags=0) if os.path.exists(vor_path) else np.zeros_like(img[:,:,0])
        valid_label = 1 if os.path.exists(anno_path) and os.path.exists(gt_path) and os.path.exists(semantic_path) and os.path.exists(heat_path) and os.path.exists(point_path) and os.path.exists(vor_path) else 0
        # print(valid_label)
        img_meta = {'img_path': img_path, 'valid_label': valid_label, 'ori_shape':img.shape,}
        
        # pesudo labels
        pesudo_rand = random.random()
        if pesudo_rand < self.args.pesudo_rate and self.mode in ['train_weak', 'train_full', 'finetune']:
            anno_path = os.path.join(self.args.pesudo_class_dir, os.path.basename(anno_path))
            anno = cv2.imread(anno_path, flags=0)
            heat_path = os.path.join(self.args.pesudo_heat_dir, os.path.basename(heat_path))
            heat = cv2.imread(heat_path, flags=0)/255
            use_pesudo = True
        else:
            use_pesudo = False
        
        point_dict = pre_point_dict(point_dict)
        # vis_point(img.copy(),anno.copy(),point_dict.copy(), save_path='./1.png')
        if self.mode in ['train_weak', 'train_full', 'finetune', 'all_full', 'all_weak']:
            if self.scale_range:
                img, anno, heat, vor, point_dict = random_scale(img, anno, heat, vor, self.scale_range, self.crop_size, point_dict)
                # vis_point(img.copy(),anno.copy(),point_dict.copy(), save_path='./2.png')
            if self.crop_size:
                img, anno, heat, vor, point_dict = random_crop(img, anno, heat, vor, self.crop_size, point_dict)
                # vis_point(img.copy(),anno.copy(),point_dict.copy(), save_path='./3.png')
            if self.flip:
                img, anno, heat, vor, point_dict = random_flip(img, anno, heat, vor, self.flip, point_dict)
                # vis_point(img.copy(),anno.copy(),point_dict.copy(), save_path='./4.png')
            if self.rotate:
                img, anno, heat, vor, point_dict = random_rotate(img, anno, heat, vor, point_dict)
                # vis_point(img.copy(),anno.copy(),point_dict.copy(),'./5.png')
            
            if self.args.degree_version in ['v1','v4']:
                deg_dict = gen_deg_n_dist(img, point_dict, self.args.degree_neighbour)
            elif self.args.degree_version == 'v2':
                deg_dict = gen_deg_n_dist_v2(img, point_dict, anno.copy(), self.args.degree_neighbour)
            elif self.args.degree_version == 'v3':
                deg_dict = gen_deg_n_dist_v3(img, point_dict, anno.copy(), self.args.degree_neighbour)
            elif self.args.degree_version in ['v5']:
                deg_dict = gen_deg_n_dist_v5(img, point_dict, rand_init=True)
            elif self.args.degree_version in ['v6']:
                deg_dict = gen_deg_n_dist_v5(img, point_dict, rand_init=False)
            elif self.args.degree_version in ['v7']:
                deg_dict = gen_deg_n_dist_v7(img, point_dict, anno.copy(), use_pesudo, self.args.degree_neighbour)
            elif self.args.degree_version in ['v8']:
                deg_dict = gen_deg_n_dist_v8(img, point_dict, anno.copy(), use_pesudo, self.args.degree_neighbour)
            elif self.args.degree_version in ['v9']:
                deg_dict = gen_deg_n_dist_v9(img, point_dict)
            elif self.args.degree_version in ['v10']:
                deg_dict = gen_deg_n_dist_v10(img, point_dict, vor.copy())
            else:
                deg_dict = {'deg_map': np.zeros(), 'dist_map':dist_map, 'dist_mask': dist_mask}
            
            count = gen_counting_map(img, point_dict, kernel_size=5)
            
            img = self.transform(img)
            img = ColorJitter(brightness=self.bright, contrast=self.contrast,saturation=self.saturation,hue=self.hue)(img)
            anno = torch.tensor(anno).long()
            heat = torch.tensor(heat, dtype=torch.float)
            mask = torch.zeros(anno.shape, dtype=torch.float)
            mask[anno != 255] = 1
            mask[anno == 255] = 0
            deg_map = torch.tensor(deg_dict['deg_map']/360, dtype=torch.float)
            dist_map = torch.tensor(deg_dict['dist_map']/self.args.distance_scale, dtype=torch.float)
            dist_mask = torch.tensor(deg_dict['dist_mask'])
            count = torch.tensor(count*self.args.count_scale, dtype=torch.float)
            vor[vor==255]=1
            img_meta['heat'] = heat
            img_meta['mask'] = mask
            img_meta['deg_map'] = deg_map
            img_meta['dist_map'] = dist_map
            img_meta['dist_mask'] = dist_mask
            img_meta['count'] = count
            img_meta['vor'] = vor
            
        elif self.mode in ['val', 'val_train']:
            img, anno, valid_region = multi_scale_test(img, anno, scale=self.test_multi_scale, crop_size=self.crop_size)
            img_meta['valid_region'] = valid_region
            if isinstance(img, list):
                img = [self.transform(i) for i in img]
                anno = [torch.tensor(a) for a in anno]
                deg_map = [torch.ones(i.shape[1:])*self.args.test_degree/360 for i in img]
            else:
                img = self.transform(img)
                anno = torch.tensor(anno)
                deg_map = torch.ones(img.shape[1:])*self.args.test_degree/360
            img_meta['gt'] = torch.tensor(gt)
            img_meta['semantic'] = torch.tensor(semantic)
            img_meta['deg_map'] = deg_map
        else:
            raise NotImplementedError("Dataset mode {} is not implemented!".format(self.mode))
                
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
    
    train_dataset = WSCellSeg(args,mode='train')
    val_dataset = WSCellSeg(args,mode='val')
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    for ii, item in enumerate(train_dataloader):
        print("The {}/{} batches...".format(ii,len(train_dataloader)),end='\r')
    for ii, item in enumerate(val_dataloader):
        print("The {}/{} batches...".format(ii,len(val_dataloader)),end='\r')