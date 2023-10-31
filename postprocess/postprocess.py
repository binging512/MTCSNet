import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage import measure
from skimage.transform import rescale
import tifffile as tif
from skimage.color import label2rgb

'''
宾语给的78.08的后处理
'''

def postprocess(img_dir, heat_dir, output_dir):
    heat_list = sorted(os.listdir(heat_dir))
    heat_paths = [os.path.join(heat_dir,heat_name) for heat_name in heat_list]
    for ii,heat_path in enumerate(heat_paths):
        print("Processing the {}/{} images....".format(ii,len(heat_paths)),end='\r')
        img_path = heat_path.replace(heat_dir,img_dir)
        heatmap = cv2.imread(heat_path,flags=0)/255
        th_cell = 0.5
        th_seed = 0.8
        mask, seeds, new_inst_map = mc_distance_postprocessing(heatmap, th_cell, th_seed, downsample=False)
        os.makedirs(os.path.join(output_dir, 'seg'), exist_ok=True)
        tif.imwrite(os.path.join(output_dir, 'seg', os.path.basename(heat_path.replace('.png','.tiff'))), new_inst_map, compression='zlib')
        # 可视化
        # color labels
        clr_labels = label2rgb(new_inst_map, bg_label=0)
        clr_labels *= 255
        clr_labels = clr_labels.astype('uint8')
        os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, 'vis', os.path.basename(heat_path)),clr_labels)

        # 可视化cell, seed, 和争议部分
        mask = mask.astype(np.uint8)
        seeds = (seeds > 0).astype(np.uint8)
        bbd = mask - seeds
        hc_img = np.concatenate((mask, seeds, bbd), axis=1)  # H W
        hc_img = (hc_img * 255).astype(np.uint8)
        os.makedirs(os.path.join(output_dir, 'vis_bbd'), exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, 'vis_bbd', os.path.basename(heat_path)),hc_img)

def mc_distance_postprocessing(cell_prediction, th_cell, th_seed, downsample):
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

def mc_distance_postprocessing_count(cell_prediction, th_cell, seeds, downsample):
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
    mask = cell_prediction > th_cell  # get binary mask by thresholding distance prediction  比0.5大的mask（应该是看作细胞
    seeds = measure.label(seeds, background=0)
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)  # 确定0.5-0.7之间属于哪个instance
    prediction = prediction_instance
    return mask, seeds, prediction.astype(np.int32)


if __name__=="__main__":
    img_dir = './data/MoNuSeg/train/images'
    heat_dir = './workspace/CellSeg_ablation_new/CS_CS_unet50pprvdc_cls2_1head_ep300_b16_crp512_iter0_cn/results_val/score'
    output_dir = heat_dir.replace('score', 'postprocessed')
    os.makedirs(output_dir,exist_ok=True)
    postprocess(img_dir, heat_dir, output_dir)