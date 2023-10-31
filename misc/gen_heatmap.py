import os
import cv2
import numpy as np
from skimage import segmentation, morphology

def gen_heatmap(label_dir, output_dir):
    label_names= os.listdir(label_dir)
    for ii, label_name in enumerate(label_names):
        print("Generating the {}/{} images...".format(ii, label_name), end='\r')
        label_path = os.path.join(label_dir, label_name)
        label = cv2.imread(label_path ,flags=0)
        mask= np.zeros(label.shape)
        mask[label==1] = 1
        fg_mask = morphology.dilation(mask,morphology.disk(1))
        boundary = segmentation.find_boundaries(fg_mask, mode='inner')
        
        label[label==255]=0
        label[boundary] = 1
        label = label.astype(np.float32)
        label = cv2.GaussianBlur(label, (5,5), sigmaX=-1)
        save_path = os.path.join(output_dir,label_name)
        label = label/np.max(label)
        cv2.imwrite(save_path, label*255)
    pass

if __name__=="__main__":
    label_dir = './data/MoNuSeg/train/labels/weak'
    output_dir = './data/MoNuSeg/train/labels/weak_heatmap_new'
    os.makedirs(output_dir,exist_ok=True)
    gen_heatmap(label_dir, output_dir)