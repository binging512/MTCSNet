from skimage.segmentation import slic,mark_boundaries
from skimage import io
import numpy as np
import cv2
import os
import tifffile as tif
import json
from skimage.color import label2rgb

def get_distance(pt1, pt2):
    dist = np.sqrt(np.power(pt1[0]-pt2[0],2)+np.power(pt1[1]-pt2[1],2))
    return dist

def get_min_distance(fg_point_dict):
    select_points = [v['select_point'] for k,v in fg_point_dict.items()]
    if len(select_points)>=2:
        min_dist_list = []
        for i, pt1 in enumerate(select_points):
            dist_list = []
            for ii, pt2 in enumerate(select_points):
                dist_list.append(get_distance(pt1, pt2))
            dist_list.remove(0)
            min_dist_list.append(min(dist_list))
        min_dist = min(min_dist_list)
    else:
        min_dist = 100
    min_dist = min(100,min_dist)
    return min_dist

def get_adjacent_superpixel(superpixel_value_list, superpixel):
    temp = np.zeros(superpixel.shape)
    for superpixel_value in superpixel_value_list:
        temp[superpixel == superpixel_value] = 1
    temp = cv2.morphologyEx(temp,cv2.MORPH_DILATE, kernel=np.ones((3,3)))
    superpixel = superpixel*temp
    adj_superpixel = list(map(int,list(np.unique(superpixel))))
    adj_superpixel.remove(0)
    for superpixel_value in superpixel_value_list:
        adj_superpixel.remove(superpixel_value)
    return adj_superpixel

def get_superpixel_feature(img, superpixel, superpixel_value):
    H,W,C = img.shape
    mask = np.zeros(img.shape[:2])
    mask[superpixel==superpixel_value] = 1
    num_pixel = np.sum(mask)
    superpixel_color = np.sum(img*mask[:,:,np.newaxis],axis=(0,1))/num_pixel/255
    _, _, _, superpixel_centroid = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    superpixel_position = [superpixel_centroid[1][0]/H, superpixel_centroid[1][1]/W]
    superpixel_feature = [superpixel_color[0], superpixel_color[1], superpixel_color[2], superpixel_position[0], superpixel_position[1]]
    return superpixel_feature

def feature_dist(feat1,feat2):
    return np.linalg.norm(np.array(feat1)-np.array(feat2))

def gen_super_pixel(img_dir, output_dir):
    img_list = sorted(os.listdir(img_dir))
    for ii,img_name in enumerate(img_list):
        print("Processing the {}/{} images...".format(ii, len(img_list)),end='\r')
        img_path = os.path.join(img_dir, img_name)
        img = io.imread(img_path)
        H,W,C = img.shape
        n_segment = 2500 #round(H*W/200)
        n_segment = round((50*W/1024)*(50*H/1024))*16
        seg = slic(img, n_segments=n_segment, compactness=10, sigma=3, max_num_iter=5, slic_zero=True,)
        save_path = os.path.join(output_dir, img_name)
        vis = mark_boundaries(img,seg)
        tif.imwrite(save_path.replace('.png','.tiff'), seg)
        cv2.imwrite(save_path.replace('.png','_vis.png'), vis*255)

def gen_superpixel_adaptive(img_dir, point_dir, output_dir):
    img_list = sorted(os.listdir(img_dir))
    for ii,img_name in enumerate(img_list):
        print("Processing the {}/{} images...".format(ii, len(img_list)),end='\r')
        img_path = os.path.join(img_dir, img_name)
        img = tif.imread(img_path)
        H,W,C = img.shape
        # find the minimum distance between the foreground
        point_path = os.path.join(point_dir, img_name.replace('.tif','.json'))
        point_dict = json.load(open(point_path,'r'))
        fg_points_dict = point_dict['foreground']
        min_dist = get_min_distance(fg_points_dict)
        min_dist = max(20, min_dist)
        h_num = H/min_dist
        w_num = W/min_dist
        n_segment = h_num*w_num
        seg = slic(img, n_segments=n_segment, compactness=10, sigma=3, max_num_iter=5, slic_zero=True,)
        save_path = os.path.join(output_dir, img_name)
        vis = mark_boundaries(img,seg)
        tif.imwrite(save_path.replace('.tif','.tiff'), seg)
        cv2.imwrite(save_path.replace('.tif','_vis.png'), vis*255)

def merge_superpixel(img_dir, superpixel_dir, output_dir):
    for ii, img_name in enumerate(sorted(os.listdir(img_dir))):
        print("Merging the {}/{} superpixel images...".format(ii,len(os.listdir(img_dir))), end='\r')
        img_path = os.path.join(img_dir,img_name)
        img = cv2.imread(img_path)
        superpixel_path = os.path.join(superpixel_dir, img_name.replace('.png','.tiff'))
        superpixel = tif.imread(superpixel_path)
        merged_superpixels = []
        flaged_superpixels = []
        H,W,C = img.shape
        # Collecting superpixels
        for i in range(1, np.max(superpixel)+1):
            superpixel_feature = get_superpixel_feature(img, superpixel, i)
            adj_superpixel_list = get_adjacent_superpixel([i], superpixel)
            merging_superpixel = []
            for j in adj_superpixel_list:
                adj_superpixel_feature = get_superpixel_feature(img, superpixel, j)
                feat_dist = feature_dist(superpixel_feature, adj_superpixel_feature) 
                if feat_dist <= 0.1:
                    merging_superpixel.append(j)
            if not i in flaged_superpixels:
                merged_superpixels.append([i,])
                flaged_superpixels.append(i)
            
            for merged_superpixel in merged_superpixels:
                if i in merged_superpixel:
                    for j in merging_superpixel:
                        if j not in flaged_superpixels:
                            merged_superpixel.append(j)
                            flaged_superpixels.append(j)
        # Merge the pixels
        seg = np.zeros((H,W),dtype=np.uint16)
        for idx, merged_superpixel in enumerate(merged_superpixels):
            for pixel in merged_superpixel:
                seg[superpixel == pixel] = idx+1
        save_path = os.path.join(output_dir, img_name)
        vis = mark_boundaries(img, seg)
        tif.imwrite(save_path.replace('.png','.tiff'), seg)
        cv2.imwrite(save_path.replace('.png','_vis.png'), vis*255)

def get_img_feat(img):
    H,W,C = img.shape
    h_pos = np.stack([np.arange(H)]*W, axis=1)/H
    w_pos = np.stack([np.arange(W)]*H, axis=0)/W
    pos_embed = np.stack((h_pos,w_pos), axis=2)
    img_feat = np.concatenate((img/255, pos_embed), axis=2)
    return img_feat

def gen_cluster_pixels(img_dir, point_dir, output_dir):
    img_list = sorted(os.listdir(img_dir))
    for ii,img_name in enumerate(img_list):
        print("Processing the {}/{} images...".format(ii, len(img_list)),end='\r')
        img_path = os.path.join(img_dir, img_name)
        img = tif.imread(img_path)
        H,W,C = img.shape
        img_feat = get_img_feat(img)
        # find the minimum distance between the foreground
        point_path = os.path.join(point_dir, img_name.replace('.tif','.json'))
        point_dict = json.load(open(point_path,'r'))
        fg_points_list = [point['select_point'] for point_idx, point in point_dict['foreground'].items()]
        bg_points_list = [[point['x'], point['y']] for point_idx, point in point_dict['background'].items()]
        
        fg_feats = [np.concatenate((img[fg[1], fg[0]]/255, [fg[1]/H], [fg[0]/W]), axis=0) for fg in fg_points_list]
        bg_feats = [np.concatenate((img[bg[1], bg[0]]/255, [bg[1]/H], [bg[0]/W]), axis=0) for bg in bg_points_list]
        n_fg = len(fg_feats)
        n_bg = len(bg_feats)
        
        label = np.zeros((H,W))
        img_feat = np.reshape(img_feat, (-1, 5))
        img_feat = np.concatenate([img_feat[np.newaxis,:,:]]*(n_fg+n_bg), axis=0)
        proto_feats = np.concatenate((fg_feats, bg_feats), axis=0)
        feat_dists = np.linalg.norm(img_feat - proto_feats[:,np.newaxis,:], axis=2)
        
        feat_label = np.argmax(feat_dists, axis=0)
        feat_label = np.reshape(feat_label,(H,W))
        label[feat_label>= n_fg] = 0
        label = label+ (feat_label< n_fg)*(feat_label+1)
        # label[feat_label< n_fg] = feat_label+1
        
        vis = label2rgb(label)*255
        vis = vis.astype('uint8')
        save_path = os.path.join(output_dir, img_name.replace('.tif', '.png'))
        cv2.imwrite(save_path, vis)
        
def gen_semantic_label(gt_dir, output_dir):
    for ii, gt_name in enumerate(sorted(os.listdir(gt_dir))):
        print("Processing the {}/{} images...".format(ii, len(os.listdir(gt_dir))),end='\r')
        gt_path = os.path.join(gt_dir,gt_name)
        gt = tif.imread(gt_path)
        semantic = np.zeros(gt.shape)
        semantic[gt>0]= 128
        output_path = os.path.join(output_dir,gt_name.replace('_label.tiff','.png'))
        cv2.imwrite(output_path, semantic)
    pass

if __name__ =="__main__":
    img_dir = './data/NeurIPS2022_CellSeg/images'
    output_dir = './data/NeurIPS2022_CellSeg/super_pixel_1'
    # img_dir = "/home/zby/WSCellseg/data/NeurIPS2022_CellSeg/temp/images"
    # output_dir = '/home/zby/WSCellseg/data/NeurIPS2022_CellSeg/temp/results'
    os.makedirs(output_dir, exist_ok=True)
    # gen_super_pixel(img_dir, output_dir)
    
    img_dir = './data/MoNuSeg/train/images'
    point_dir = './data/MoNuSeg/train/points_v5'
    output_dir = './data/MoNuSeg/train/superpixel_v5_L'
    os.makedirs(output_dir, exist_ok=True)
    gen_superpixel_adaptive(img_dir, point_dir, output_dir)
    
    img_dir = './data/NeurIPS2022_CellSeg/images'
    superpxiel_dir = './data/NeurIPS2022_CellSeg/superpixel_adaptive'
    output_dir = './data/NeurIPS2022_CellSeg/superpixel_merged'
    os.makedirs(output_dir, exist_ok=True)
    # merge_superpixel(img_dir, superpxiel_dir, output_dir)
    
    gt_dir = './data/NeurIPS2022_CellSeg/gts'
    output_dir = './data/NeurIPS2022_CellSeg/semantics'
    os.makedirs(output_dir,exist_ok=True)
    # gen_semantic_label(gt_dir, output_dir)