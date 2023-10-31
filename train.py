import os
import random
import cv2
import numpy as np
from skimage import measure
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import ColorJitter

from datasets.WSCellseg import WSCellSeg

from models.unet import UNet
from models.unetplusplus import NestedUNet
from models.unet_parallel import UNet_parallel
from models.FCT import FCT
from models.CISNet import CISNet
from models.loss import *

from utils.slide_infer import slide_inference
from utils.f1_score import compute_af1_results
from utils.cutmix import CutMix
from utils.miou import eval_metrics
from utils.tools import resize
from postprocess.postprocess import mc_distance_postprocessing
from metrics.instance_metrics import get_fast_aji_plus, get_fast_pq

def seed_everything():
    seed = 512
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def validate(args, logger, model, dataloader):
    logger.info("============== Validating ==============")
    all_f1_results, vor_f1_results, heat_f1_results = [], [], []
    post_pred_f1_results, post_vor_f1_results, post_heat_f1_results = [], [], []
    aAcc, IoU, Acc, Dice, post_aAcc, post_IoU, post_Acc, post_Dice = [], [], [], [], [], [], [], []
    vor_aAcc, vor_IoU, vor_Acc, vor_Dice, vor_post_aAcc, vor_post_IoU, vor_post_Acc, vor_post_Dice = [], [], [], [], [], [], [], []
    
    AJI_list, DQ_list, SQ_list, PQ_list = [], [], [], []
    vor_AJI_list, vor_DQ_list, vor_SQ_list, vor_PQ_list = [], [], [], []
    heat_AJI_list, heat_DQ_list, heat_SQ_list, heat_PQ_list = [], [], [], []
    
    for ii, item in enumerate(dataloader):
        if ii%50 == 0:
            logger.info("Validating the {}/{} images...".format(ii,len(dataloader)))
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
        
        # classifier head
        _, post_pred_seeds, post_pred_seg = mc_distance_postprocessing(pred_scores, args.infer_threshold, args.infer_seed, downsample=False, min_area=args.infer_min_area)
        _, post_heat_seeds, post_heat_seg = mc_distance_postprocessing(heat_scores, args.infer_threshold, args.infer_seed, downsample=False, min_area=args.infer_min_area)
        # vor head
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
        # F1 score from classification
        seg = np.zeros_like(pred_cls)
        seg[pred_cls==1] = 1
        seg = measure.label(seg, background=0)
        gt = img_meta['gt'].squeeze().numpy()
        f1_list = np.array(compute_af1_results(gt, seg, 0, 0))
        if all_f1_results==[]:
            all_f1_results = f1_list
        else:
            all_f1_results += f1_list
        # F1 score from vor
        vor_seg = np.zeros_like(pred_vor_cls)
        vor_seg[pred_vor_cls==1]=1
        vor_seg = measure.label(vor_seg, background=0)
        f1_list = np.array((compute_af1_results(gt, vor_seg, 0, 0)))
        if vor_f1_results==[]:
            vor_f1_results = f1_list
        else:
            vor_f1_results += f1_list
        # F1 score from heatmap
        heat_seg = np.zeros_like(heat_scores)
        heat_seg[heat_scores >= args.infer_threshold] = 1
        heat_seg[heat_scores < args.infer_threshold] = 0
        seg = measure.label(heat_seg, background=0)
        f1_list = np.array(compute_af1_results(gt, seg, 0, 0))
        if heat_f1_results==[]:
            heat_f1_results = f1_list
        else:
            heat_f1_results += f1_list
        # F1 score from post classification
        f1_list = np.array(compute_af1_results(gt, post_pred_seg, 0, 0))
        if post_pred_f1_results==[]:
            post_pred_f1_results = f1_list
        else:
            post_pred_f1_results += f1_list
        # F1 score from post vor
        f1_list = np.array(compute_af1_results(gt, post_pred_vor_seg, 0, 0))
        if post_vor_f1_results==[]:
            post_vor_f1_results = f1_list
        else:
            post_vor_f1_results += f1_list
        # F1 score from post heatmap
        f1_list = np.array(compute_af1_results(gt, post_heat_seg, 0, 0))
        if post_heat_f1_results==[]:
            post_heat_f1_results = f1_list
        else:
            post_heat_f1_results += f1_list
        
        # IoU
        semantic = img_meta['semantic'].squeeze().numpy()
        semantic[semantic==128] = 1
        if args.net_num_classes == 3:
            semantic[semantic==255] = 2
        semantic_new = np.zeros(semantic.shape)
        semantic_new[gt>0] = 1
        # ret_metrics = eval_metrics(pred_cls, semantic, args.net_num_classes, ignore_index=255)
        ret_metrics = eval_metrics(pred_cls, semantic_new, args.net_num_classes, ignore_index=255, metrics=['mIoU','mDice'])
        aAcc.append(ret_metrics['aAcc'])
        IoU.append(ret_metrics['IoU'])
        Acc.append(ret_metrics['Acc'])
        Dice.append(ret_metrics['Dice'])
        ret_metrics = eval_metrics(pred_vor_cls, semantic_new, args.net_num_classes, ignore_index=255, metrics=['mIoU','mDice'])
        vor_aAcc.append(ret_metrics['aAcc'])
        vor_IoU.append(ret_metrics['IoU'])
        vor_Acc.append(ret_metrics['Acc'])
        vor_Dice.append(ret_metrics['Dice'])
        
        post_pred_cls = np.zeros_like(post_pred_seg)
        post_pred_cls[post_pred_seg>0]=1
        post_pred_vor_cls = np.zeros_like(post_pred_vor_seg)
        post_pred_vor_cls[post_pred_vor_seg>0]=1
        ret_metrics = eval_metrics(post_pred_cls, semantic_new, args.net_num_classes, ignore_index=255, metrics=['mIoU','mDice'])
            
        post_aAcc.append(ret_metrics['aAcc'])
        post_IoU.append(ret_metrics['IoU'])
        post_Acc.append(ret_metrics['Acc'])
        post_Dice.append(ret_metrics['Dice'])
        ret_metrics = eval_metrics(post_pred_vor_cls, semantic_new, args.net_num_classes, ignore_index=255, metrics=['mIoU','mDice'])
        vor_post_aAcc.append(ret_metrics['aAcc'])
        vor_post_IoU.append(ret_metrics['IoU'])
        vor_post_Acc.append(ret_metrics['Acc'])
        vor_post_Dice.append(ret_metrics['Dice'])
        
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
        save_path = os.path.join(args.workspace, args.results_val,'pred', img_name)
        score_save_path = os.path.join(args.workspace, args.results_val,'score', img_name)
        vor_save_path = os.path.join(args.workspace, args.results_val,'vor', img_name)
        vor_score_save_path = os.path.join(args.workspace, args.results_val,'vor_score', img_name)
        heat_save_path = os.path.join(args.workspace, args.results_val,'heat', img_name)
        deg_save_path = os.path.join(args.workspace, args.results_val,'deg', img_name.replace('.png','_deg{}.pkl'.format(args.test_degree)))
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        os.makedirs(os.path.dirname(score_save_path),exist_ok=True)
        os.makedirs(os.path.dirname(vor_save_path),exist_ok=True)
        os.makedirs(os.path.dirname(vor_score_save_path),exist_ok=True)
        os.makedirs(os.path.dirname(heat_save_path),exist_ok=True)
        os.makedirs(os.path.dirname(deg_save_path),exist_ok=True)
        cv2.imwrite(save_path, pred_cls)
        cv2.imwrite(score_save_path, pred_scores*255)
        cv2.imwrite(vor_save_path, pred_vor_cls)
        cv2.imwrite(vor_score_save_path, pred_vor_scores*255)
        cv2.imwrite(heat_save_path, heat_scores*255)
        if args.net_degree:
            pickle.dump(deg_scores, open(deg_save_path, 'wb'))
        
    all_f1_results = all_f1_results/len(dataloader)
    f1_scores={
        'F1@0.5': all_f1_results[0],
        'F1@0.75': all_f1_results[5],
        'F1@0.9': all_f1_results[8],
        'F1@0.5:1.0:0.05':np.mean(all_f1_results),
    }
    vor_f1_results = vor_f1_results/len(dataloader)
    vor_f1_scores = {
        'F1@0.5': vor_f1_results[0],
        'F1@0.75': vor_f1_results[5],
        'F1@0.9': vor_f1_results[8],
        'F1@0.5:1.0:0.05':np.mean(vor_f1_results),
    }
    heat_f1_results = heat_f1_results/len(dataloader)
    heat_f1_scores={
        'F1@0.5': heat_f1_results[0],
        'F1@0.75': heat_f1_results[5],
        'F1@0.9': heat_f1_results[8],
        'F1@0.5:1.0:0.05':np.mean(heat_f1_results),
    }
    post_pred_f1_results = post_pred_f1_results/len(dataloader)
    post_pred_f1_scores={
        'F1@0.5': post_pred_f1_results[0],
        'F1@0.75': post_pred_f1_results[5],
        'F1@0.9': post_pred_f1_results[8],
        'F1@0.5:1.0:0.05':np.mean(post_pred_f1_results),
    }
    post_vor_f1_results = post_vor_f1_results/len(dataloader)
    post_vor_f1_scores={
        'F1@0.5': post_vor_f1_results[0],
        'F1@0.75': post_vor_f1_results[5],
        'F1@0.9': post_vor_f1_results[8],
        'F1@0.5:1.0:0.05':np.mean(post_vor_f1_results),
    }
    post_heat_f1_results = post_heat_f1_results/len(dataloader)
    post_heat_f1_scores={
        'F1@0.5': post_heat_f1_results[0],
        'F1@0.75': post_heat_f1_results[5],
        'F1@0.9': post_heat_f1_results[8],
        'F1@0.5:1.0:0.05':np.mean(post_heat_f1_results),
    }
    
    aAcc = np.nanmean(aAcc)
    IoU = np.nanmean(np.stack(IoU), axis=0)
    Acc = np.nanmean(np.stack(Acc), axis=0)
    Dice = np.nanmean(np.stack(Dice), axis=0)
    vor_aAcc = np.nanmean(vor_aAcc)
    vor_IoU = np.nanmean(np.stack(vor_IoU), axis=0)
    vor_Acc = np.nanmean(np.stack(vor_Acc), axis=0)
    vor_Dice = np.nanmean(np.stack(vor_Dice), axis=0)
    post_aAcc = np.nanmean(post_aAcc)
    post_IoU = np.nanmean(np.stack(post_IoU), axis=0)
    post_Acc = np.nanmean(np.stack(post_Acc), axis=0)
    post_Dice = np.nanmean(np.stack(post_Dice), axis=0)
    vor_post_aAcc = np.nanmean(vor_post_aAcc)
    vor_post_IoU = np.nanmean(np.stack(vor_post_IoU), axis=0)
    vor_post_Acc = np.nanmean(np.stack(vor_post_Acc), axis=0)
    vor_post_Dice = np.nanmean(np.stack(vor_post_Dice), axis=0)

    logger.info("Validation Complete!!!")
    logger.info("============== Calculating Metrics ==============")
    if args.net_celoss:
        logger.info("Classification Results:")
        logger.info("F1@0.5: {} ".format(f1_scores['F1@0.5']))
        logger.info("F1@0.75: {} ".format(f1_scores['F1@0.75']))
        logger.info("F1@0.9: {} ".format(f1_scores['F1@0.9']))
        logger.info("F1@0.5:1.0:0.05: {} ".format(f1_scores['F1@0.5:1.0:0.05']))
    if args.net_regression:
        logger.info("Regression Results:")
        logger.info("F1@0.5: {} ".format(heat_f1_scores['F1@0.5']))
        logger.info("F1@0.75: {} ".format(heat_f1_scores['F1@0.75']))
        logger.info("F1@0.9: {} ".format(heat_f1_scores['F1@0.9']))
        logger.info("F1@0.5:1.0:0.05: {} ".format(heat_f1_scores['F1@0.5:1.0:0.05']))
    if args.net_vorloss:
        logger.info("Voronoi Results:")
        logger.info("F1@0.5: {} ".format(vor_f1_scores['F1@0.5']))
        logger.info("F1@0.75: {} ".format(vor_f1_scores['F1@0.75']))
        logger.info("F1@0.9: {} ".format(vor_f1_scores['F1@0.9']))
        logger.info("F1@0.5:1.0:0.05: {} ".format(vor_f1_scores['F1@0.5:1.0:0.05']))
    # mIoU
    if args.net_celoss:
        logger.info("Classification IoU Results:")
        logger.info("aAcc: {}".format(aAcc))
        logger.info("IoU: {}, mIoU: {}".format(IoU, np.mean(IoU)))
        logger.info("Acc: {}, mAcc: {}".format(Acc, np.mean(Acc)))
        logger.info("Dice: {}, mDice: {}".format(Dice, np.mean(Dice)))
    if args.net_vorloss:
        logger.info("Voronoi IoU Results:")
        logger.info("aAcc: {}".format(vor_aAcc))
        logger.info("IoU: {}, mIoU: {}".format(vor_IoU, np.mean(vor_IoU)))
        logger.info("Acc: {}, mAcc: {}".format(vor_Acc, np.mean(vor_Acc)))
        logger.info("Dice: {}, mDice: {}".format(vor_Dice, np.mean(vor_Dice)))
    # Postprocessed
    logger.info("============== Post Metrics ==============")
    if args.net_celoss:
        logger.info("Post classification Results:")
        logger.info("F1@0.5: {} ".format(post_pred_f1_scores['F1@0.5']))
        logger.info("AJI: {}".format(np.mean(AJI_list)))
        logger.info("DQ: {:.5f}, SQ: {:.5f}, PQ: {:.5f}".format(np.mean(DQ_list), np.mean(SQ_list), np.mean(PQ_list)))
        logger.info("IoU: {}".format(post_IoU[1]))
        logger.info("Dice: {}".format(post_Dice[1]))
    if args.net_regression:
        logger.info("Post regression Results:")
        logger.info("F1@0.5: {} ".format(post_heat_f1_scores['F1@0.5']))
        logger.info("AJI: {}".format(np.mean(heat_AJI_list)))
        logger.info("DQ: {:.5f}, SQ: {:.5f}, PQ: {:.5f}".format(np.mean(heat_DQ_list), np.mean(heat_SQ_list), np.mean(heat_PQ_list)))
    if args.net_vorloss:
        logger.info("Post voronoi Results:")
        logger.info("F1@0.5: {} ".format(post_vor_f1_scores['F1@0.5']))
        logger.info("AJI: {}".format(np.mean(vor_AJI_list)))
        logger.info("DQ: {:.5f}, SQ: {:.5f}, PQ: {:.5f}".format(np.mean(vor_DQ_list), np.mean(vor_SQ_list), np.mean(vor_PQ_list)))
        logger.info("IoU: {}".format(vor_post_IoU[1]))
        logger.info("Dice: {}, mDice: {}".format(vor_post_Dice, np.mean(vor_post_Dice)))
    # mIoU
    logger.info("Classification IoU Results:")

    results_dict = {
        'f1_score': f1_scores['F1@0.5'],
        'heat_f1_score': heat_f1_scores['F1@0.5'],
        'post_pred_f1_score': post_pred_f1_scores['F1@0.5'], 
        'post_heat_f1_score': post_heat_f1_scores['F1@0.5'],
        'aAcc': aAcc, 
        'mIoU': np.mean(IoU), 
        'mAcc': np.mean(Acc)
    }
    return results_dict

def train_iter0(args, logger):
    seed_everything()
    train_dataset = WSCellSeg(args, mode = args.train_mode)
    val_dataset = WSCellSeg(args, mode = 'val')
    # sampler = WeightedRandomSampler(train_dataset.weights, num_samples=round(len(train_dataset)/args.batch_size))
    # train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size, sampler=sampler, num_workers=args.num_worker)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    if args.net_name.lower() == 'unet':
        print("Using Model: unet")
        model = UNet(args)
    elif args.net_name.lower() == 'unet_par':
        print("Using Model: unet_parallel")
        model = UNet_parallel(args)
    elif args.net_name.lower() in ['unetplusplus','unetpp']:
        print("Using Model: unetplusplus")
        model = NestedUNet(args)
    elif args.net_name.lower() in ['fct']:
        print("Using Model: FCT")
        model = FCT(args)
    elif args.net_name.lower() in ['cisnet']:
        print("Using Model: cisnet")
        model = CISNet(args)
    else:
        raise NotImplementedError("Model {} is not implemented!".format(args.net_name.lower()))
    
    if os.path.isfile(args.net_resume):
        logger.info("Model training resume from: {}".format(args.net_resume))
        model.load_state_dict(torch.load(args.net_resume), strict=True)
    model = nn.DataParallel(model.cuda())

    ceLoss = nn.CrossEntropyLoss(ignore_index=255)
    diceloss = DiceLoss(ignore_index=255)
    focalloss = FocalLoss()
    mseloss = nn.MSELoss()
    ssimloss = SSIM(window_size=11)
    pixelcontrastloss = PixelContrastLoss()
    certaintyloss = CertaintyLoss(args, ignore_index=255)
    maskmseloss = MaskMSELoss()
    maskdistloss = MaskDistLoss()
    maskdistloss_v2 = MaskDistLoss_v2(args)
    maskiouloss = MaskIOULoss()
    blockmaeloss = BlockMAELoss()
    consistencyloss = ConsistencyLoss()

    # optimizer = AdamW(model.parameters(),lr=args.net_learning_rate)
    if args.net_name.lower() in ['unet','unet_par','unetplusplus', 'unetpp', 'cisnet']:
        optimizer = AdamW([{'params':list(model.module.pretrained.parameters()), 'lr':args.net_learning_rate/10},
                        {'params':list(model.module.new_added.parameters()), 'lr':args.net_learning_rate}])
    else:
        optimizer = AdamW(model.parameters(),lr=args.net_learning_rate)
        
    scheduler = CosineAnnealingLR(optimizer, T_max=args.net_num_epoches*len(train_dataloader), eta_min=1e-8)

    best_score = 0
    best_epoch = 0
    best_aAcc = 0
    best_mIoU = 0
    best_mAcc = 0
    best_mode = 'None'
    for ep in range(args.net_num_epoches):
        logger.info("==============Training {}/{} Epoches==============".format(ep,args.net_num_epoches))
        for ii, item in enumerate(train_dataloader):
            optimizer.zero_grad()
            img, anno, img_meta = item
            img = img.cuda()
            anno = anno.cuda()
            gt_heat = img_meta['heat'].cuda()
            mask = img_meta['mask'].cuda()
            deg_map = img_meta['deg_map'].cuda()
            dist_map = img_meta['dist_map'].cuda()
            dist_mask = img_meta['dist_mask'].cuda()
            vor = img_meta['vor'].cuda()
            
            if args.cutmix:
                img, anno, gt_heat, deg_map, dist_map, dist_mask, vor = CutMix(args, img, anno, gt_heat, deg_map, dist_map, dist_mask, vor)
            
            pred, pred_vor, cert, heat, deg = model(img)
            
            loss = 0
            loss_dict = {}
            # Calculate the losses
            pred = [F.interpolate(p, img.shape[-2:], mode='bilinear') for p in pred]
            pred_vor = [F.interpolate(v, img.shape[-2:], mode='bilinear') for v in pred_vor]
            cert = [torch.sigmoid(F.interpolate(c, img.shape[-2:], mode='bilinear')) for c in cert]
            heat = [torch.sigmoid(F.interpolate(h, img.shape[-2:], mode='bilinear')) for h in heat]
            deg = [F.interpolate(d, img.shape[-2:], mode='bilinear') for d in deg]
            for iii in range(len(pred)):
                if args.net_celoss:
                    loss_ce = ceLoss(pred[iii],anno.long()) 
                    loss += loss_ce
                    loss_dict["loss_ce"] = loss_ce
                if args.net_diceloss:
                    loss_dice = diceloss(pred[iii],anno.long()) 
                    loss += loss_dice
                    loss_dict["loss_dice"] = loss_dice
                if args.net_focalloss:
                    loss_focal = focalloss(pred[iii],anno.long())
                    loss += loss_focal
                    loss_dict["loss_focal"] = loss_focal
                if args.net_vorloss:
                    loss_vor = ceLoss(pred_vor[iii], vor.long())
                    loss += loss_vor
                    loss_dict['loss_vor'] = loss_vor
                if args.net_consistency:
                    loss_consistency = consistencyloss(pred[iii], pred_vor[iii])
                    loss += loss_consistency
                    loss_dict['loss_consis'] = loss_consistency
                if args.net_certainty:
                    loss_certrain = certaintyloss(pred[iii], cert[iii], anno.long())
                    loss += loss_certrain
                    loss_dict['loss_cert'] = loss_certrain
                if args.net_regression:
                    loss_heat_mse = maskmseloss(heat[iii], gt_heat, mask)*args.net_reg_weight
                    loss += loss_heat_mse
                    loss_dict['loss_mask_mse'] = loss_heat_mse
                if args.net_degree:
                    if args.degree_version in ['v4']:
                        loss_deg_mse = maskdistloss_v2(deg[iii], dist_map, deg_map, dist_mask)
                    elif args.degree_version in ['v9']:
                        loss_deg_mse = maskdistloss(deg[iii], dist_map, dist_mask)
                    elif args.degree_version in ['v10']:
                        loss_deg_mse = maskiouloss(deg[iii], dist_map, dist_mask)*0.1
                    else:
                        loss_deg_mse = maskdistloss(deg[iii], dist_map.unsqueeze(1), dist_mask.unsqueeze(1))
                    loss += loss_deg_mse
                    loss_dict['loss_deg_mse'] = loss_deg_mse
                    
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if ii%5 == 0:
                logger.info("Epoch:{}/{} || iter:{}/{} || loss:{:.4f} || lr:{}".format(ep, args.net_num_epoches,
                                                                                    ii, len(train_dataloader),
                                                                                    loss, scheduler.get_last_lr()[0]))
                logger.info("Loss details:{}".format(["{}:{:.8f}".format(k,v) for k,v in loss_dict.items()]))

        model.eval()
        if (ep+1)%args.val_interval == 0:
            results_dict = validate(args,logger,model,val_dataloader)
            save_model_path = os.path.join(args.workspace,'checkpoints/epoch_{}.pth'.format(ep+1))
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            torch.save(model.module.state_dict(), save_model_path)
            if results_dict['post_pred_f1_score']>best_score:
                best_score = results_dict['post_pred_f1_score']
                best_aAcc = results_dict['aAcc']
                best_mIoU = results_dict['mIoU']
                best_mAcc = results_dict['mAcc']
                best_epoch = ep
                best_mode = 'Classification'
                torch.save(model.module.state_dict(), os.path.join(args.workspace,'best_model.pth'))
                logger.info("Best mode: {}".format(best_mode))
                logger.info("Best model update: Epoch:{}, F1_score:{}, aAcc:{:.5f}, mIoU:{:.5f}, mAcc:{:.5f}".format(best_epoch, 
                                                                                                                     best_score, 
                                                                                                                     best_aAcc, 
                                                                                                                     best_mIoU,
                                                                                                                     best_mAcc))
                logger.info("Best model saved to: {}".format(os.path.join(args.workspace,'best_model.pth')))
            if results_dict['post_heat_f1_score']>best_score:
                best_score = results_dict['post_heat_f1_score']
                best_aAcc = results_dict['aAcc']
                best_mIoU = results_dict['mIoU']
                best_mAcc = results_dict['mAcc']
                best_epoch = ep
                best_mode = 'Regression'
                torch.save(model.module.state_dict(), os.path.join(args.workspace,'best_model.pth'))
                logger.info("Best mode: {}".format(best_mode))
                logger.info("Best model update: Epoch:{}, F1_score:{}, aAcc:{:.5f}, mIoU:{:.5f}, mAcc:{:.5f}".format(best_epoch, 
                                                                                                                     best_score, 
                                                                                                                     best_aAcc, 
                                                                                                                     best_mIoU,
                                                                                                                     best_mAcc))
                logger.info("Best model saved to: {}".format(os.path.join(args.workspace,'best_model.pth')))
            
        if (ep+1)%args.save_interval==0:
            torch.save(model.module.state_dict(), os.path.join(args.workspace,'epoch_{}.pth'.format(ep+1)))
            logger.info("Checkpoint saved to {}".format(os.path.join(args.workspace,'epoch_{}.pth'.format(ep+1))))
            
        logger.info("Best mode:{}, Best Epoch:{}, Best_F1_score:{}, Best_aAcc:{:.5f}, Best_mIoU:{:.5f}, Best_mAcc:{:.5f}".format(best_mode,
                                                                                                                                best_epoch, 
                                                                                                                                best_score, 
                                                                                                                                best_aAcc, 
                                                                                                                                best_mIoU,
                                                                                                                                best_mAcc))
        model.train()
    torch.save(model.module.state_dict(), os.path.join(args.workspace,'final.pth'))
