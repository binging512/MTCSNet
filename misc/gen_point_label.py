import os
import shutil
import numpy as np
import cv2
import json
import random
import tifffile as tif
from distancemap import distance_map_from_binary_matrix
import threading

def gen_point_supervision(gt_paths, out_point_dir):
    for i, gt_path in enumerate(gt_paths):
        print("Processing the {}/{} items, gt path: {}".format(i,len(gt_paths), gt_path),end='\r')
        point_dict = {"background":{}, "foreground":{}}
        gt_name = os.path.basename(gt_path)
        gt = tif.imread(gt_path)
        
        # select the foreground points
        num_cells = np.max(gt)
        for ii in range(1,num_cells+1):
            cell = np.zeros(gt.shape)
            cell[gt==ii] = 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cell.astype(np.uint8),connectivity=8)
            x,y,w,h,s = stats[1]
            centroid = centroids[1]
            centroid = list(map(int,centroid))
            
            # point_x, point_y = centroid
            # if cell[point_y, point_x] == 0:
            #     print("False point in image:{}".format(gt_path))
            flag = 0
            attempt_num = 0
            point_range = 0.1
            while flag == 0:
                if attempt_num == 20:
                    point_range += 0.05
                    attempt_num = 0
                offset_ratio_x = ((random.random()-0.5)/0.5)*point_range      #[-0.25,0.25]
                offset_ratio_y = ((random.random()-0.5)/0.5)*point_range      #[-0.25,0.25]
                offset_x = round(w*offset_ratio_x)
                offset_y = round(h*offset_ratio_y)
                point_x = centroid[0] + offset_x
                point_y = centroid[1] + offset_y
                if cell[point_y, point_x] == 1:
                    flag = 1
                else:
                    attempt_num += 1
                if point_range > 0.3:
                    print("False point in image:{}".format(gt_path))
                    
            point_dict['foreground'][str(ii)] = {'x':int(x),
                                                'y':int(y),
                                                'w':int(w),
                                                'h':int(h),
                                                's':int(s),
                                                'centroid':centroid,
                                                "select_point":[point_x, point_y]}
            
        # select the background points
        background = np.zeros(gt.shape)
        H,W = background.shape
        background[gt>0] = 1
        background_points= []
        need_point_num = max(10,round(num_cells))
        while len(background_points)<need_point_num:
            x = random.randint(0, W-1)
            y = random.randint(0, H-1)
            if background[y][x] == 0:
                background_points.append([x,y])
        for ii, point in enumerate(background_points):
            point_dict['background'][str(ii+1)] = {'x':int(point[0]),
                                                 'y':int(point[1])}
            
        json.dump(point_dict, open(os.path.join(out_point_dir, gt_name.replace('_label.tiff','.json')),'w'), indent=2)
    pass

if __name__=="__main__":
    img_dir = './data/NeurIPS2022_CellSeg/images'
    gt_dir = './data/NeurIPS2022_CellSeg/gts'
    out_point_dir = './data/NeurIPS2022_CellSeg/points_v2'
    os.makedirs(out_point_dir,exist_ok=True)
    gt_paths = [os.path.join(gt_dir, gt_name) for gt_name in sorted(os.listdir(gt_dir))]
    # thread_num = 8
    # databins = []
    # for i in range(thread_num):
    #     databins.append([])
    # for ii, gt_path in enumerate(gt_paths):
    #     databins[ii%thread_num].append(gt_path)
    # [print(len(databin)) for databin in databins]
    # thread_bins = []
    # for i in range(thread_num):
    #     thread_bin = threading.Thread(target=gen_point_supervision,args=(databins[i],out_point_dir))
    #     thread_bins.append(thread_bin)

    # for i, thread_bin in enumerate(thread_bins):
    #     print('Thread {} started!'.format(i))
    #     thread_bin.start()

    gen_point_supervision(gt_paths, out_point_dir)
