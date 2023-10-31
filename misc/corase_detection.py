import os
import cv2
import json
import tifffile as tif
import numpy as np
from skimage import measure
from skimage.segmentation import watershed
from skimage.transform import rescale

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

def get_corase_detection(img_dir, score_dir, point_dir, output_dir):
    vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    th_cell = 0.3
    score_names = sorted(os.listdir(score_dir))
    for ii, score_name in enumerate(score_names):
        print("Cropping the {}/{} images....".format(ii, len(score_names)), end='\r')
        img_path = os.path.join(img_dir,score_name.replace('.png','.tif'))
        img = tif.imread(img_path)
        score_path = os.path.join(score_dir, score_name)
        score = cv2.imread(score_path, flags=0)/255
        point_path = os.path.join(point_dir, score_name.replace('.png', '.json'))
        point_dict = json.load(open(point_path, 'r'))
        # set the seed with gt
        seeds = np.zeros(score.shape)
        for p_idx,p_prop in point_dict['foreground'].items():
            select_point = p_prop['select_point']
            seeds[select_point[1], select_point[0]] = 1
            
        mask, seeds, prediction = mc_distance_postprocessing(score, th_cell, seeds, downsample=False)
        cv2.imwrite('./1.jpg', prediction)
        props = measure.regionprops(prediction)
        bboxes = [p.bbox for p in props if p.area>=64]
        
        # H,W,C = img.shape
        # new_bboxes = []
        # for bbox in bboxes:
        #     y1,x1,y2,x2 = bbox
        #     h = y2-y1
        #     w = x2-x1
        #     new_x1 = max(0, round(x1-w*0.2))
        #     new_y1 = max(0, round(y1-h*0.2))
        #     new_x2 = min(W-1, round(x2+w*0.2))
        #     new_y2 = min(H-1, round(y2+h*0.2))
        #     new_bboxes.append([new_y1,new_x1,new_y2,new_x2])
        # bboxes = new_bboxes
        
        # visualization
        score = score*255
        score = np.stack((score,score,score), axis=-1)
        for bbox in bboxes:
            y1,x1,y2,x2 = bbox
            img = cv2.rectangle(img, (x1,y1), (x2,y2), color=(255,0,0), thickness=1)
            score = cv2.rectangle(score, (x1,y1), (x2,y2), color=(255,0,0), thickness=1)
        for p_idx,p_prop in point_dict['foreground'].items():
            select_point = p_prop['select_point']
            img = cv2.circle(img, select_point, radius=2, color=(0,255,0))
            score = cv2.circle(score, select_point, radius=2, color=(0,255,0))
        vis = np.hstack((img,score))
        vis_save_path = os.path.join(vis_dir, score_name)
        cv2.imwrite(vis_save_path, vis)
        # save the file
        bbox_dict = {}
        for idx, bbox in enumerate(bboxes):
            bbox_dict[idx] = {'pt1':[bbox[0],bbox[1]], 'pt2':[bbox[2], bbox[3]]}
        save_path = os.path.join(output_dir, score_name.replace('.png','.json'))
        json.dump(bbox_dict, open(save_path,'w'), indent=2)

def mc_distance_postprocessing_val(cell_prediction, th_cell, th_seed, downsample):
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

    seeds = measure.label(cell_prediction > th_seed, background=0)  # get seeds  初始种子用0.7作为阈值
    props = measure.regionprops(seeds)  # Remove very small seeds  去除面积过小的种子

    region_miu = np.mean([prop.area for prop in props if prop.area > min_area])
    region_sigma = np.sqrt(np.var([prop.area for prop in props if prop.area > min_area]))
    region_range = [region_miu - 2 * region_sigma, region_miu + 2 * region_sigma]

    for idx, prop in enumerate(props):
        if prop.area < min_area or prop.area < region_range[0]:
            seeds[seeds == prop.label] = 0  # 但是这样就进入争议区域了

    if len(props) <= 50:  # 小的基，可以用腐蚀或者膨胀
        seeds = cv2.dilate((seeds > 0).astype(np.uint16), np.ones((5, 5), np.uint8), iterations=1)

    seeds = measure.label(seeds, background=0)
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)  # 确定0.5-0.7之间属于哪个instance
    
    # # Semantic segmentation / classification
    # prediction_class = np.zeros_like(prediction_instance)
    # for idx in range(1, prediction_instance.max()+1):
    #     # Get sum of distance prediction of selected cell for each class (class 0 is sum of the other classes)
    #     pix_vals = cell_prediction[1:][:, prediction_instance == idx]
    #     cell_layer = np.sum(pix_vals, axis=1).argmax() + 1  # +1 since class 0 needs to be considered for argmax
    #     prediction_class[prediction_instance == idx] = cell_layer

    if downsample:
        # Downsample instance segmentation
        prediction_instance = rescale(prediction_instance,
                                      scale=0.8,
                                      order=0,
                                      preserve_range=True,
                                      anti_aliasing=False).astype(np.int32)

        # Downsample semantic segmentation
        # prediction_class = rescale(prediction_class,
        #                            scale=0.8,
        #                            order=0,
        #                            preserve_range=True,
        #                            anti_aliasing=False).astype(np.uint16)

    # Combine instance segmentation and semantic segmentation results
    # prediction = np.concatenate((prediction_instance[np.newaxis, ...], prediction_class[np.newaxis, ...]), axis=0)
    prediction = prediction_instance
    return mask, seeds, prediction.astype(np.int32)

def get_corase_detection_val(img_dir, score_dir, output_dir):
    vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    th_cell = 0.3
    th_seed = 0.7
    score_names = sorted(os.listdir(score_dir))
    for ii, score_name in enumerate(score_names):
        print("Cropping the {}/{} images....".format(ii, len(score_names)), end='\r')
        img_path = os.path.join(img_dir,score_name.replace('.png','.tif'))
        img = tif.imread(img_path)
        score_path = os.path.join(score_dir, score_name)
        score = cv2.imread(score_path, flags=0)/255
            
        mask, seeds, prediction = mc_distance_postprocessing_val(score, th_cell, th_seed, downsample=False)
        # cv2.imwrite('./1.jpg', prediction)
        props = measure.regionprops(prediction)
        bboxes = [p.bbox for p in props if p.area>=64]
        # save the file
        bbox_dict = {}
        for idx, bbox in enumerate(bboxes):
            bbox_dict[idx] = {'pt1':[bbox[0],bbox[1]], 'pt2':[bbox[2], bbox[3]]}
        save_path = os.path.join(output_dir, score_name.replace('.png','.json'))
        json.dump(bbox_dict, open(save_path,'w'), indent=2)
        
        # visualization
        score = score*255
        score = np.stack((score,score,score), axis=-1)
        for bbox in bboxes:
            y1,x1,y2,x2 = bbox
            img = cv2.rectangle(img, (x1,y1), (x2,y2), color=(255,0,0), thickness=1)
            score = cv2.rectangle(score, (x1,y1), (x2,y2), color=(255,0,0), thickness=1)
        vis = np.hstack((img,score))
        vis_save_path = os.path.join(vis_dir, score_name)
        cv2.imwrite(vis_save_path, vis)
        

if __name__=="__main__":
    img_dir = './data/MoNuSeg/train/images'
    score_dir = './workspace/pt_v5/Mo_Mo_unet50_cls2_1head_ep200_b4_crp512_iter0/results_test/score'
    point_dir = './data/MoNuSeg/train/points_v5'
    output_dir = './workspace/pt_v5/Mo_Mo_unet50_cls2_1head_ep200_b4_crp512_iter0/results_test/det'
    os.makedirs(output_dir, exist_ok=True)
    get_corase_detection(img_dir, score_dir, point_dir, output_dir)
    
    img_dir = './data/MoNuSeg/test/images'
    score_dir = './workspace/pt_v5/Mo_Mo_unet50_cls2_1head_ep200_b4_crp512_iter0/results_test/score'
    output_dir = './workspace/pt_v5/Mo_Mo_unet50_cls2_1head_ep200_b4_crp512_iter0/results_test/det'
    os.makedirs(output_dir, exist_ok=True)
    get_corase_detection_val(img_dir, score_dir, output_dir)