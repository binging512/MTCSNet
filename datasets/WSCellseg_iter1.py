import os
import sys
sys.path.append('/home/zby/Cellseg')
import cv2
import json
import tifffile as tif
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor,ColorJitter
from datasets.utils import *

class WSCellSeg_iter1(Dataset): 
    def __init__(self, args, mode) -> None:
        super(WSCellSeg_iter1, self).__init__()
        self.args = args
        self.data_root = args.data_root
        self.split_info = json.load(open(args.split_info,'r'))
        if mode == 'train_weak':
            self.img_list = [train_item['img_path'] for train_item in self.split_info['train']]
            self.anno_list = [train_item['weak_label_path'] for train_item in self.split_info['train']]
            self.gt_list = [train_item['gt_path'] for train_item in self.split_info['train']]
            self.semantic_list = [train_item['semantic_path'] for train_item in self.split_info['train']]
            self.heat_list = [train_item['weak_heat_path'] for train_item in self.split_info['train']]
            self.point_list = [train_item['point_path'] for train_item in self.split_info['train']]
            self.bbox_list = [train_item['bbox_path'] for train_item in self.split_info['train']]
            self.bbox_dict_list = []
            json_list = [json.load(open(json_path,'r')) for json_path in self.bbox_list]
            for ii, bbox_dict in enumerate(json_list):
                for bbox_idx, bbox_prop in bbox_dict.items():
                    self.bbox_dict_list.append({'img_path': self.img_list[ii], 
                                                'anno_path': self.anno_list[ii],
                                                'gt_path': self.gt_list[ii],
                                                'semantic_path': self.semantic_list[ii],
                                                'heat_path': self.heat_list[ii],
                                                'point_path': self.point_list[ii],
                                                'pt1': bbox_prop['pt1'], 'pt2': bbox_prop['pt2']})
        elif mode == 'val_train':
            self.img_list = [val_item['img_path'] for val_item in self.split_info['train']]
            self.anno_list = [val_item['full_label_path'] for val_item in self.split_info['train']]
            self.gt_list = [val_item['gt_path'] for val_item in self.split_info['train']]
            self.semantic_list = [val_item['semantic_path'] for val_item in self.split_info['train']]
            self.heat_list = [val_item['full_heat_path'] for val_item in self.split_info['train']]
            self.point_list = [val_item['point_path'] for val_item in self.split_info['train']]
            self.bbox_list = [val_item['bbox_path'] for val_item in self.split_info['train']]
            self.bbox_dict_list = []
            json_list = [json.load(open(json_path,'r')) for json_path in self.bbox_list]
            for ii, bbox_dict in enumerate(json_list):
                for bbox_idx, bbox_prop in bbox_dict.items():
                    self.bbox_dict_list.append({'img_path': self.img_list[ii], 
                                                'anno_path': self.anno_list[ii],
                                                'gt_path': self.gt_list[ii],
                                                'semantic_path': self.semantic_list[ii],
                                                'heat_path': self.heat_list[ii],
                                                'point_path': self.point_list[ii],
                                                'pt1': bbox_prop['pt1'], 'pt2': bbox_prop['pt2']})
        elif mode == 'val':
            self.img_list = [val_item['img_path'] for val_item in self.split_info['val']]
            self.anno_list = [val_item['full_label_path'] for val_item in self.split_info['val']]
            self.gt_list = [val_item['gt_path'] for val_item in self.split_info['val']]
            self.semantic_list = [val_item['semantic_path'] for val_item in self.split_info['val']]
            self.heat_list = [val_item['full_heat_path'] for val_item in self.split_info['val']]
            self.point_list = [val_item['point_path'] for val_item in self.split_info['val']]
            self.bbox_list = [val_item['bbox_path'] for val_item in self.split_info['val']]
            self.bbox_dict_list = []
            json_list = [json.load(open(json_path,'r')) for json_path in self.bbox_list]
            for ii, bbox_dict in enumerate(json_list):
                for bbox_idx, bbox_prop in bbox_dict.items():
                    self.bbox_dict_list.append({'img_path': self.img_list[ii], 
                                                'anno_path': self.anno_list[ii],
                                                'gt_path': self.gt_list[ii],
                                                'semantic_path': self.semantic_list[ii],
                                                'heat_path': self.heat_list[ii],
                                                'point_path': self.point_list[ii],
                                                'bbox_index': bbox_idx,
                                                'pt1': bbox_prop['pt1'], 'pt2': bbox_prop['pt2']})
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
            self.bbox_list = self.anno_list
            self.bbox_dict_list = self.anno_list
        else:
            raise NotImplementedError()
        
        self.mode = mode
        self.resize = args.crop_size
        self.flip = args.rand_flip
        self.rotate = args.rand_rotate
        self.bright = args.rand_bright
        self.contrast = args.rand_contrast
        self.saturation = args.rand_saturation
        self.hue = args.rand_hue
        self.test_multi_scale = args.test_multi_scale
        self.transform = ToTensor()
        
    def __getitem__(self, index):
        if self.mode in ['train_weak']:
            item_dict = self.bbox_dict_list[index]
            img_path = item_dict['img_path']
            anno_path = item_dict['anno_path']
            gt_path = item_dict['gt_path']
            semantic_path = item_dict['semantic_path']
            heat_path = item_dict['heat_path']
            point_path = item_dict['point_path']
            
            if img_path.endswith('.png'):
                img = cv2.imread(img_path)
            elif img_path.endswith('.tif') or img_path.endswith('.tiff'):
                img = tif.imread(img_path)
                
            anno = cv2.imread(anno_path,flags=0) if os.path.exists(anno_path) else np.ones_like(img[:,:,0])*255
            gt = tif.imread(gt_path).astype(np.int16) if os.path.exists(gt_path) else np.zeros_like(img[:,:,0]).astype(np.int16)
            semantic = cv2.imread(semantic_path,flags=0) if os.path.exists(semantic_path) else np.zeros_like(img[:,:,0])
            heat = cv2.imread(heat_path, flags=0)/255 if os.path.exists(heat_path) else np.zeros_like(img[:,:,0])
            point_dict = json.load(open(point_path, 'r')) if os.path.exists(point_path) and self.args.net_degree else None
            valid_label = 1 if os.path.exists(anno_path) else 0
            
            img_meta = {'img_path': img_path, 'valid_label': valid_label, 'ori_shape':img.shape,
                        'gt': torch.tensor(gt), 'semantic': torch.tensor(semantic)}
            
            point_dict = pre_point_dict(point_dict)
            pt1 = item_dict['pt1']
            pt2 = item_dict['pt2']
            
            img, anno, heat, point_dict = img_crop(img, anno, heat, point_dict, pt1, pt2)
            img, anno, heat, point_dict, valid_region = img_resize(img, anno, heat, point_dict, self.args.crop_size)
            img_meta['valid_region'] = valid_region
            img_meta['ori_pt1'] = pt1 
            img_meta['ori_pt2'] = pt2
            if self.args.rand_flip:
                img, anno, heat, point_dict = random_flip(img, anno, heat, self.args.rand_flip, point_dict)
            if self.args.rand_rotate:
                img, anno, heat, point_dict = random_rotate(img, anno, heat, point_dict)
            
            if self.args.degree_version in ['v5']:
                deg_dict = gen_deg_n_dist_v5(img, point_dict, rand_init=True)
            img = torch.tensor(img/255).permute(2,0,1)
            img = ColorJitter(brightness=self.bright, contrast=self.contrast,saturation=self.saturation,hue=self.hue)(img)
            anno = torch.tensor(anno).long()
            heat = torch.tensor(heat, dtype=torch.float)
            mask = torch.zeros(anno.shape, dtype=torch.float)
            mask[anno != 255] = 1
            mask[anno == 255] = 0
            deg_map = torch.tensor(deg_dict['deg_map']/360, dtype=torch.float)
            dist_map = torch.tensor(deg_dict['dist_map']/self.args.distance_scale, dtype=torch.float)
            dist_mask = torch.tensor(deg_dict['dist_mask'])
            img_meta['heat'] = heat
            img_meta['mask'] = mask
            img_meta['deg_map'] = deg_map
            img_meta['dist_map'] = dist_map
            img_meta['dist_mask'] = dist_mask
            
        elif self.mode in ['val', 'val_train']:
            img_path = self.img_list[index]
            anno_path = self.anno_list[index]
            gt_path = self.gt_list[index]
            semantic_path = self.semantic_list[index]
            heat_path = self.heat_list[index]
            point_path = self.point_list[index]
            bbox_path = self.bbox_list[index]
            
            if img_path.endswith('.png'):
                img = cv2.imread(img_path)
            elif img_path.endswith('.tif') or img_path.endswith('.tiff'):
                img = tif.imread(img_path)
                
            anno = cv2.imread(anno_path,flags=0) if os.path.exists(anno_path) else np.ones_like(img[:,:,0])*255
            gt = tif.imread(gt_path).astype(np.int16) if os.path.exists(gt_path) else np.zeros_like(img[:,:,0]).astype(np.int16)
            semantic = cv2.imread(semantic_path,flags=0) if os.path.exists(semantic_path) else np.zeros_like(img[:,:,0])
            heat = cv2.imread(heat_path, flags=0)/255 if os.path.exists(heat_path) else np.zeros_like(img[:,:,0])
            point_dict = json.load(open(point_path, 'r')) if os.path.exists(point_path) and self.args.net_degree else None
            # bbox_dict = json.load(open(bbox_path, 'r')) if os.path.exists(bbox_dict) else None
            valid_label = 1 if os.path.exists(anno_path) else 0
            
            img_meta = {'img_path': img_path, 'valid_label': valid_label, 'ori_shape':img.shape,
                        'gt': torch.tensor(gt), 'semantic': torch.tensor(semantic), 'bbox_path': bbox_path}
            
            img = torch.tensor(img/255).permute(2,0,1)
            anno = torch.tensor(anno).long()
            deg_map = torch.ones(img.shape[1:])*self.args.test_degree/360
            img_meta['deg_map'] = deg_map
            
        else:
            raise NotImplementedError("Dataset mode {} is not implemented!".format(self.mode))
        
        return img, anno, img_meta
    
    def __len__(self):
        if self.mode in ['train_weak']:
            return len(self.bbox_dict_list)
        else:
            return len(self.img_list)
    
if __name__=="__main__":
    pass