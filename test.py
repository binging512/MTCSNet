import os
import cv2
import numpy as np
from PIL import Image
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from skimage import measure
from datasets.WSCellseg import WSCellSeg

from models.unet import UNet
from models.unetplusplus import NestedUNet
from models.unet_parallel import UNet_parallel
from models.FCT import FCT
from models.CISNet import CISNet
from models.loss import DiceLoss, FocalLoss

from utils.slide_infer import slide_inference
from postprocess.postprocess import mc_distance_postprocessing
from utils.f1_score import compute_af1_results
from utils.miou import eval_metrics
from utils.tools import resize
from metrics.instance_metrics import get_fast_aji_plus, get_fast_pq


def test_iter0(args, logger):
    test_dataset = WSCellSeg(args,mode=args.test_mode)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    
    if args.net_name.lower() == 'unet':
        print("Using Model: unet")
        model = UNet(args)
    elif args.net_name.lower() == 'unetplusplus':
        print("Using Model: unetplusplus")
        model = NestedUNet(args)
    elif args.net_name.lower() == 'unet_par':
        print("Using Model: unet_parallel")
        model = UNet_parallel(args)
    elif args.net_name.lower() in ['fct']:
        print("Using Model: FCT")
        model = FCT(args)
    elif args.net_name.lower() in ['cisnet']:
        print("Using Model: cisnet")
        model = CISNet(args)
    else:
        raise NotImplementedError("Model {} is not implemented!".format(args.net_name.lower()))

    logger.info("Loading checkpoint from: {}".format(os.path.join(args.workspace,args.checkpoint)))
    model_weights = torch.load(os.path.join(args.workspace,args.checkpoint))
    new_dict = {}
    for k,v in model_weights.items():
        if k in list(model.state_dict().keys()):
            new_dict[k] = v
    model.load_state_dict(new_dict,strict=True)
    
    model = model.cuda()
    model.eval()
    logger.info("============== Testing ==============")
    post_pred_f1_results, post_vor_f1_results, post_heat_f1_results = [], [], []
    post_IoU, post_Dice, vor_post_IoU, vor_post_Dice, heat_post_IoU, heat_post_Dice = [], [], [], [], [], []
    
    AJI_list, DQ_list, SQ_list, PQ_list = [], [], [], []
    vor_AJI_list, vor_DQ_list, vor_SQ_list, vor_PQ_list = [], [], [], []
    heat_AJI_list, heat_DQ_list, heat_SQ_list, heat_PQ_list = [], [], [], []
    
    for ii, item in enumerate(test_dataloader):
        if ii%10 == 0:
            logger.info("Testing the {}/{} images...".format(ii,len(test_dataloader)))
        imgs, annos, img_meta = item
        preds_list = []
        preds_vor_list = []
        certs_list = []
        heats_list = []
        degs_list = []
        for idx, img in enumerate(imgs):
            img = img.cuda()
            valid_region = img_meta['valid_region'][idx]
            
            with torch.no_grad():
                preds, preds_vor, certs, heats, degs = slide_inference(model, img, img_meta, rescale=True, args=args, valid_region=valid_region)
                
            # Classification
            preds_list.append(preds.detach().cpu())
            preds_vor_list.append(preds_vor.detach().cpu())
            certs_list.append(certs.detach().cpu())
            heats_list.append(heats.detach().cpu())
            degs_list.append(degs.detach().cpu())
        # Fusion the results from different scales
        if args.test_fusion =='mean':
            fused_preds = torch.mean(torch.stack(preds_list,dim=0), dim=0)
            fused_preds_vor = torch.mean(torch.stack(preds_vor_list, dim=0), dim=0)
            fused_certs = torch.mean(torch.stack(certs_list,dim=0), dim=0)
            fused_heats = torch.mean(torch.stack(heats_list,dim=0), dim=0)
            fused_degs = torch.mean(torch.stack(degs_list,dim=0), dim=0)
        if args.test_fusion == 'max':
            fused_preds,_ = torch.max(torch.stack(preds_list,dim=0), dim=0)
            fused_preds_vor,_ = torch.max(torch.stack(preds_vor_list, dim=0), dim=0)
            fused_certs,_ = torch.max(torch.stack(certs_list,dim=0), dim=0)
            fused_heats,_ = torch.max(torch.stack(heats_list,dim=0), dim=0)
            fused_degs,_ = torch.max(torch.stack(degs_list,dim=0), dim=0)
        
        fused_preds = torch.softmax(fused_preds, dim=1)
        pred_cls = torch.argmax(fused_preds.detach().cpu(),dim=1).squeeze().numpy()
        pred_scores = fused_preds[:,1,:,:].squeeze().numpy()
        fused_preds_vor = torch.softmax(fused_preds_vor, dim=1)
        pred_vor_cls = torch.argmax(fused_preds_vor.detach().cpu(), dim=1).squeeze().numpy()
        pred_vor_scores = fused_preds_vor[:,1,:,:].squeeze().numpy()
        cert_scores = fused_certs[:,0,:,:].squeeze().numpy()
        heat_scores = torch.sigmoid(fused_heats[:,0,:,:]).squeeze().numpy()
        deg_scores = fused_degs[:,0,:,:].squeeze().numpy()*args.distance_scale
        
        # classifier head & vor head
        _, post_pred_seeds, post_pred_seg = mc_distance_postprocessing(pred_scores, args.infer_threshold, args.infer_seed, downsample=False, min_area=args.infer_min_area)
        _, post_heat_seeds, post_heat_seg = mc_distance_postprocessing(heat_scores, args.infer_threshold, args.infer_seed, downsample=False, min_area=args.infer_min_area)
        _, post_pred_seeds, post_pred_vor_seg = mc_distance_postprocessing(pred_vor_scores, args.infer_threshold, args.infer_seed, downsample=False, min_area=args.infer_min_area)
        
        # Rescale to the original scale, using the first scale factor
        pred_cls = resize(pred_cls, img_meta['ori_shape'][:2], mode='nearest')
        pred_scores = resize(pred_scores, img_meta['ori_shape'][:2], mode='bilinear')
        pred_vor_cls = resize(pred_vor_cls, img_meta['ori_shape'][:2], mode='nearest')
        pred_vor_scores = resize(pred_vor_scores, img_meta['ori_shape'][:2], mode='bilinear')
        cert_scores = resize(cert_scores, img_meta['ori_shape'][:2], mode='bilinear')
        heat_scores = resize(heat_scores, img_meta['ori_shape'][:2], mode='bilinear')
        deg_scores = resize(deg_scores, img_meta['ori_shape'][:2], mode='bilinear', if_deg=True)
        post_pred_seg = resize(post_pred_seg, img_meta['ori_shape'][:2], mode='nearest')
        post_heat_seg = resize(post_heat_seg, img_meta['ori_shape'][:2], mode='nearest')
        post_pred_vor_seg = resize(post_pred_vor_seg, img_meta['ori_shape'][:2], mode='nearest')

        # Calculate the metrics
        gt = img_meta['gt'].squeeze().numpy()
        # F1 score from post classification
        f1_list = np.array(compute_af1_results(gt, post_pred_seg, 0, 0))
        post_pred_f1_results.append(f1_list[0])
        # F1 score from post vor
        f1_list = np.array(compute_af1_results(gt, post_pred_vor_seg, 0, 0))
        post_vor_f1_results.append(f1_list[0])
        # F1 score from post heatmap
        f1_list = np.array(compute_af1_results(gt, post_heat_seg, 0, 0))
        post_heat_f1_results.append(f1_list[0])
        
        # IoU
        semantic = img_meta['semantic'].squeeze().numpy()
        semantic[semantic==128] = 1
        if args.net_num_classes == 3:
            semantic[semantic==255] = 2
        semantic_new = np.zeros(semantic.shape)
        semantic_new[gt>0] = 1
        
        post_pred_cls = np.zeros_like(post_pred_seg)
        post_pred_cls[post_pred_seg>0]=1
        post_pred_vor_cls = np.zeros_like(post_pred_vor_seg)
        post_pred_vor_cls[post_pred_vor_seg>0]=1
        post_pred_heat_cls = np.zeros_like(post_heat_seg)
        post_pred_heat_cls[post_heat_seg>0] = 1
        
        ret_metrics = eval_metrics(post_pred_cls, semantic_new, args.net_num_classes, ignore_index=255, metrics=['mIoU','mDice'])
        post_IoU.append(ret_metrics['IoU'][1])
        post_Dice.append(ret_metrics['Dice'][1])
        ret_metrics = eval_metrics(post_pred_vor_cls, semantic_new, args.net_num_classes, ignore_index=255, metrics=['mIoU','mDice'])
        vor_post_IoU.append(ret_metrics['IoU'][1])
        vor_post_Dice.append(ret_metrics['Dice'][1])
        ret_metrics = eval_metrics(post_pred_heat_cls, semantic_new, args.net_num_classes, ignore_index=255, metrics=['mIoU','mDice'])
        heat_post_IoU.append(ret_metrics['IoU'][1])
        heat_post_Dice.append(ret_metrics['Dice'][1])
        
        # AJI,PQ
        pred_aji = get_fast_aji_plus(gt, post_pred_seg)
        pred_qs, _= get_fast_pq(gt, post_pred_seg)
        pred_dq, pred_sq, pred_pq = pred_qs
        AJI_list.append(pred_aji)
        DQ_list.append(pred_dq)
        SQ_list.append(pred_sq)
        PQ_list.append(pred_pq)
        
        pred_aji = get_fast_aji_plus(gt, post_pred_vor_seg)
        pred_qs, _= get_fast_pq(gt, post_pred_vor_seg)
        pred_dq, pred_sq, pred_pq = pred_qs
        vor_AJI_list.append(pred_aji)
        vor_DQ_list.append(pred_dq)
        vor_SQ_list.append(pred_sq)
        vor_PQ_list.append(pred_pq)
        
        pred_aji = get_fast_aji_plus(gt, post_heat_seg)
        pred_qs, _= get_fast_pq(gt, post_heat_seg)
        pred_dq, pred_sq, pred_pq = pred_qs
        heat_AJI_list.append(pred_aji)
        heat_DQ_list.append(pred_dq)
        heat_SQ_list.append(pred_sq)
        heat_PQ_list.append(pred_pq)
        
        # Visualization
        pred_cls[pred_cls==1] = 128
        pred_cls[pred_cls==2] = 255
        pred_vor_cls[pred_vor_cls==1]=128
        pred_vor_cls[pred_vor_cls==2]=255
        img_name = os.path.basename(img_meta['img_path'][0]).split('.')[0]+'.png'
        save_path = os.path.join(args.workspace,args.results_test,'pred', img_name)
        save_score_path = os.path.join(args.workspace,args.results_test, 'score', img_name)
        vor_save_path = os.path.join(args.workspace, args.results_val,'vor', img_name)
        vor_score_save_path = os.path.join(args.workspace, args.results_val,'vor_score', img_name)
        heat_save_path = os.path.join(args.workspace, args.results_test, 'heat', img_name)
        deg_save_path = os.path.join(args.workspace, args.results_test,'deg', img_name.replace('.png','_deg{}.pkl'.format(args.test_degree)))
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        os.makedirs(os.path.dirname(save_score_path),exist_ok=True)
        os.makedirs(os.path.dirname(vor_save_path),exist_ok=True)
        os.makedirs(os.path.dirname(vor_score_save_path),exist_ok=True)
        os.makedirs(os.path.dirname(heat_save_path),exist_ok=True)
        os.makedirs(os.path.dirname(deg_save_path),exist_ok=True)
        cv2.imwrite(save_path, pred_cls)
        cv2.imwrite(save_score_path, pred_scores*255)
        cv2.imwrite(vor_save_path, pred_vor_cls)
        cv2.imwrite(vor_score_save_path, pred_vor_scores*255)
        cv2.imwrite(heat_save_path, heat_scores*255)
        if args.net_degree:
            pickle.dump(deg_scores, open(deg_save_path, 'wb'))
    
    
    logger.info("============== Final Metrics ==============")
    if args.net_celoss:
        logger.info("Post Classification Results:")
        logger.info("F1@0.5: {} ".format(np.mean(post_pred_f1_results)))
        logger.info("IoU: {}".format(np.mean(post_IoU)))
        logger.info("Dice: {}".format(np.mean(post_Dice)))
        logger.info("AJI: {}".format(np.mean(AJI_list)))
        logger.info("DQ: {:.5f}, SQ: {:.5f}, PQ: {:.5f}".format(np.mean(DQ_list), np.mean(SQ_list), np.mean(PQ_list)))
    if args.net_regression:
        logger.info("Post Regression Results:")
        logger.info("F1@0.5: {} ".format(np.mean(post_heat_f1_results)))
        logger.info("IoU: {}".format(np.mean(heat_post_IoU)))
        logger.info("Dice: {}".format(np.mean(heat_post_Dice)))
        logger.info("AJI: {}".format(np.mean(heat_AJI_list)))
        logger.info("DQ: {:.5f}, SQ: {:.5f}, PQ: {:.5f}".format(np.mean(heat_DQ_list), np.mean(heat_SQ_list), np.mean(heat_PQ_list)))
    if args.net_vorloss:
        logger.info("Post Voronoi Results:")
        logger.info("F1@0.5: {} ".format(np.mean(post_vor_f1_results)))
        logger.info("IoU: {}".format(np.mean(vor_post_IoU)))
        logger.info("Dice: {}".format(np.mean(vor_post_Dice)))
        logger.info("AJI: {}".format(np.mean(vor_AJI_list)))
        logger.info("DQ: {:.5f}, SQ: {:.5f}, PQ: {:.5f}".format(np.mean(vor_DQ_list), np.mean(vor_SQ_list), np.mean(vor_PQ_list)))
        
    logger.info("Test Complete!!!")

if __name__=="__main__":
    test_iter0()