import os
import cv2
import tifffile as tif
import json
import math
import pickle
import numpy as np
from skimage import segmentation, morphology, measure
from skimage.segmentation import watershed
from skimage.transform import rescale
import celldetection as cd


def mc_distance_postprocessing(cell_prediction, th_cell, seeds, downsample):
    """ Post-processing for distance label (cell + neighbor) prediction.

    :param cell_prediction: 就是heatmap,0-1
    :type cell_prediction:
    :param th_cell:
    :type th_cell: float
    :param th_seed:
    :type th_seed: float
    :param downsample:
    :type downsample: bool

    :return: Instance segmentation mask.
    """

    min_area = 64  # keep only seeds larger than threshold area

    # Instance segmentation (use summed up channel 0)
    # sigma_cell = 0.5
    # cell_prediction[0] = gaussian_filter(cell_prediction[0], sigma=sigma_cell)  # slight smoothing
    mask = cell_prediction > th_cell  # get binary mask by thresholding distance prediction  比0.5大的mask（应该是看作细胞

    # seeds = measure.label(cell_prediction > th_seed, background=0)  # get seeds  初始种子用0.7作为阈值
    # props = measure.regionprops(seeds)  # Remove very small seeds  去除面积过小的种子

    # region_miu = np.mean([prop.area for prop in props if prop.area > min_area])
    # region_sigma = np.sqrt(np.var([prop.area for prop in props if prop.area > min_area]))
    # region_range = [region_miu - 2 * region_sigma, region_miu + 2 * region_sigma]

    # for idx, prop in enumerate(props):
    #     if prop.area < min_area or prop.area < region_range[0]:
    #         seeds[seeds == prop.label] = 0  # 但是这样就进入争议区域了

    # if len(props) <= 50:  # 小的基，可以用腐蚀或者膨胀
    #     seeds = cv2.dilate((seeds > 0).astype(np.uint16), np.ones((5, 5), np.uint8), iterations=1)

    seeds = measure.label(seeds, background=0)
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)  # 确定0.5-0.7之间属于哪个instance

    if downsample:
        # Downsample instance segmentation
        prediction_instance = rescale(prediction_instance,
                                      scale=0.8,
                                      order=0,
                                      preserve_range=True,
                                      anti_aliasing=False).astype(np.int32)

    prediction = prediction_instance
    return mask, seeds, prediction.astype(np.int32)

def get_degree_n_distance(pt1,pt2):
    """Caculate the degree and distance between two points

    Args:
        pt1 (_type_): the first point (x1, y1)
        pt2 (_type_): the second point (x2, y2)

    Returns:
        _type_: _description_
    """
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = math.sqrt(dx**2 + dy**2)
    if dist == 0:
        deg = 0
    else:
        sin = dy/dist
        cos = dx/dist
        if sin>= 0 and cos>=0:
            rad = math.asin(sin)
        elif sin>=0 and cos<0:
            rad = math.pi - math.asin(sin)
        elif sin<0 and cos<0:
            rad = math.pi - math.asin(sin)
        elif sin<0 and cos>=0:
            rad = 2*math.pi+ math.asin(sin)
        deg = rad*360/(2*math.pi)
    return deg, dist

def gen_distmap(inst_map, degree=0):
    cell_mask = np.zeros(inst_map.shape)
    cell_mask[inst_map>0] = 1
    radian = math.radians(degree)
    H,W = cell_mask.shape

    boundary = segmentation.find_boundaries(inst_map, mode='outer')*1
    boundary = morphology.dilation(boundary, morphology.disk(1))
    cell_mask[boundary==1]= 0
    cell_points = np.array(np.where(cell_mask==1)).T
    dist_map = np.zeros((H,W))
    for cell_point in cell_points:
        mask = np.zeros((H,W))
        if math.sin(radian)>=0 and math.cos(radian)>=0:
            margin_x = W-cell_point[1]
            margin_y = H-cell_point[0]
        elif math.sin(radian)>=0 and math.cos(radian)<0:
            margin_x = cell_point[1]
            margin_y = H-cell_point[0]
        elif math.sin(radian)<0 and math.cos(radian)<0:
            margin_x = cell_point[1]
            margin_y = cell_point[0]
        elif math.sin(radian)<0 and math.cos(radian)>=0:
            margin_x = W-cell_point[1]
            margin_y = cell_point[0]
        dx = math.cos(radian)
        dy = math.sin(radian)
        step_x = margin_x/(abs(dx)+1e-8)
        step_y = margin_y/(abs(dy)+1e-8)
        step = min(step_x,step_y)
        dest_x = min(max(0,round(cell_point[1]+ dx*step)),W)
        dest_y = min(max(0,round(cell_point[0]+ dy*step)),H)
        mask = cv2.line(mask,(cell_point[1],cell_point[0]), (dest_x, dest_y), color=1, thickness=1)
        mask = boundary*mask
        bnd_points = np.array(np.where(mask==1)).T
        if len(bnd_points)==0:
            bnd_points = [np.array([dest_y, dest_x])]
        min_dist = min([np.linalg.norm(cell_point-bnd_point) for bnd_point in bnd_points])
        dist_map[cell_point[0], cell_point[1]] = min_dist
    return dist_map

def gen_distmap_v2(inst_map, degree=0):
    dist_map = np.zeros(inst_map.shape[:2])
    radian = math.radians(degree)
    H,W = inst_map.shape
    
    dilate_mask = morphology.dilation(inst_map, morphology.disk(1))
    boundary = dilate_mask-inst_map
    cell_num = np.max(dilate_mask)
    for inst_idx in range(1, cell_num+1):
        cell_points = np.array(np.where(inst_map==inst_idx)).T
        bnd_points = np.array(np.where(boundary==inst_idx)).T
        for cell_point in cell_points:
            deg_list = []
            dist_list = []
            for bnd_point in bnd_points:
                deg, dist = get_degree_n_distance([cell_point[1],cell_point[0]], [bnd_point[1], bnd_point[0]])
                deg_list.append(deg)
                dist_list.append(dist)
            if len(deg_list)==0:
                dist_map[cell_point[0], cell_point[1]] = 0
            else:
                deg_idx = np.argmin(np.abs(degree-np.array(deg_list)))
                min_dist = dist_list[deg_idx]
                dist_map[cell_point[0], cell_point[1]] = min_dist
    return dist_map

def gen_pesudo_label(img_dir, score_dir, point_dir, output_dir):
    class_label_dir = os.path.join(output_dir, 'class_labels')
    heat_label_dir = os.path.join(output_dir,'heat_labels')
    degree_label_dir = os.path.join(output_dir, 'degree_labels')
    vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(class_label_dir, exist_ok=True)
    os.makedirs(heat_label_dir, exist_ok=True)
    os.makedirs(degree_label_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    th_cell = 0.3
    score_names = sorted(os.listdir(score_dir))
    for ii, score_name in enumerate(score_names):
        print("Generating the {}/{} images....".format(ii, len(score_names)), end='\r')
        img_path = os.path.join(img_dir,score_name.replace('.png','.tif'))
        img = tif.imread(img_path)
        score_path = os.path.join(score_dir, score_name)
        score = cv2.imread(score_path, flags=0)/255
        point_path = os.path.join(point_dir, score_name.replace('.png', '.json'))
        point_dict = json.load(open(point_path, 'r'))
        # set the seed with gt
        seeds = np.zeros(score.shape)
        for p_idx, p_prop in point_dict['foreground'].items():
            select_point = p_prop['select_point']
            seeds[select_point[1], select_point[0]] = 1
            
        mask, seeds, prediction = mc_distance_postprocessing(score, th_cell, seeds, downsample=False)
        # generate class label
        class_label = np.zeros(prediction.shape)
        boundary = segmentation.find_boundaries(prediction,mode='inner')
        boundary = morphology.dilation(boundary, morphology.disk(1))
        class_label[prediction>0] = 1
        class_label[boundary==1] = 0
        
        # generate heat label
        heat_label = np.zeros(prediction.shape)
        heat_label = morphology.erosion(class_label, morphology.disk(1))
        heat_label = cv2.GaussianBlur(class_label, (5,5), sigmaX=-1)
        heat_label = heat_label/np.max(heat_label)*255
        
        
        # visualization
        dist_maps = []
        degree_list = [0,45,90,135,180,225,270,315]
        for degree in degree_list:
            dist_map = gen_distmap(prediction, degree=degree)
            dist_maps.append(dist_map)
            save_vis_path = os.path.join(vis_dir,score_name.replace('.png','_deg{}.png'.format(degree)))
            cv2.imwrite(save_vis_path, dist_map)
            
        dist_maps = np.stack(dist_maps,axis=0)
        
        # save the files
        save_class_label_path = os.path.join(class_label_dir, score_name)
        save_heat_label_path = os.path.join(heat_label_dir, score_name)
        save_degree_label_path = os.path.join(degree_label_dir, score_name.replace('.png','.pkl'))
        cv2.imwrite(save_class_label_path, class_label)
        cv2.imwrite(save_heat_label_path, heat_label)
        pickle.dump(dist_maps, save_degree_label_path)
        
    return

def gen_pesudo_label_v2(img_dir, score_dir, point_dir, voronoi_dir, output_dir):
    class_label_dir = os.path.join(output_dir, 'class_labels')
    heat_label_dir = os.path.join(output_dir,'heat_labels')
    vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(class_label_dir, exist_ok=True)
    os.makedirs(heat_label_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    th_cell = 0.3
    score_names = sorted(os.listdir(score_dir))
    for ii, score_name in enumerate(score_names):
        print("Generating the {}/{} images....".format(ii, len(score_names)), end='\r')
        img_path = os.path.join(img_dir,score_name.replace('.png','.tif'))
        img = tif.imread(img_path)
        score_path = os.path.join(score_dir, score_name)
        score = cv2.imread(score_path, flags=0)/255
        point_path = os.path.join(point_dir, score_name.replace('.png', '.json'))
        point_dict = json.load(open(point_path, 'r'))
        # set the seed with gt
        seeds = np.zeros(score.shape)
        for p_idx, p_prop in point_dict['foreground'].items():
            select_point = p_prop['select_point']
            seeds[select_point[1], select_point[0]] = 1
            
        mask, seeds, prediction = mc_distance_postprocessing(score, th_cell, seeds, downsample=False)
        
        # generate class label
        num_cells = np.max(prediction)
        class_label = np.zeros(prediction.shape)
        # for cell_idx in range(1, num_cells+1):
            
        boundary = segmentation.find_boundaries(prediction,mode='inner')
        boundary = morphology.dilation(boundary, morphology.disk(1))
        class_label[prediction>0] = 1
        class_label[boundary==1] = 0
        
        # generate heat label
        heat_label = np.zeros(prediction.shape)
        heat_label = morphology.erosion(class_label, morphology.disk(1))
        heat_label = cv2.GaussianBlur(class_label, (5,5), sigmaX=-1)
        heat_label = heat_label/np.max(heat_label)*255
        
        
        # visualization
        # dist_map = gen_distmap(prediction, degree=45)
        # dist_map = gen_distmap_v2(prediction, degree=90)
        
        # save the files
        save_class_label_path = os.path.join(class_label_dir, score_name)
        save_heat_label_path = os.path.join(heat_label_dir, score_name)
        save_dist_label_path = os.path.join(vis_dir, score_name)
        cv2.imwrite(save_class_label_path, class_label)
        cv2.imwrite(save_heat_label_path, heat_label)
        # cv2.imwrite(save_dist_label_path, dist_map)

if __name__=="__main__":
    img_dir = './data/MoNuSeg/train/images'
    score_dir = './workspace/pt_v5/Mo_Mo_unet50r_cls2_1head_ep300_b4_crp512_iter0_rw4/results_test/score'
    point_dir = './data/MoNuSeg/train/points_v5'
    output_dir = './workspace/pt_v5/Mo_Mo_unet50r_cls2_1head_ep300_b4_crp512_iter0_rw4/results_test/pesudo'
    os.makedirs(output_dir, exist_ok=True)
    
    gen_pesudo_label(img_dir, score_dir, point_dir, output_dir)