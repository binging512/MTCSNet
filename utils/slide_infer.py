import warnings
import torch
import torch.nn.functional as F

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def slide_inference(model, img, img_meta, rescale, args, valid_region=None):
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """

    h_stride, w_stride = args.infer_stride
    h_crop, w_crop = args.crop_size
    batch_size, _, h_img, w_img = img.size()

    num_classes = args.net_num_classes
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = img.new_zeros((batch_size, num_classes, h_img, w_img)).cpu().detach()
    preds_vor = img.new_zeros((batch_size, num_classes, h_img, w_img)).cpu().detach()
    certs = img.new_zeros((batch_size, 1, h_img, w_img)).cpu().detach()
    heats = img.new_zeros((batch_size, 1, h_img, w_img)).cpu().detach()
    if args.degree_version in ['v4']:
        degs = img.new_zeros((batch_size, 4*args.net_N+3, h_img, w_img)).cpu().detach()
    elif args.degree_version in ['v9', 'v10']:
        degs = img.new_zeros((batch_size, 8, h_img, w_img)).cpu().detach()
    else:
        degs = img.new_zeros((batch_size, 1, h_img, w_img)).cpu().detach()
    
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img)).cpu().detach()
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                pred, pred_vor, cert, heat, deg = model(crop_img)
                if isinstance(pred,list):
                    pred = torch.stack(pred, dim=0)
                    pred = torch.mean(pred, dim=0)
                if isinstance(pred_vor,list):
                    pred_vor = torch.stack(pred_vor, dim=0)
                    pred_vor = torch.mean(pred_vor, dim=0)
                if isinstance(cert,list):
                    cert = torch.stack(cert, dim=0)
                    cert = torch.mean(cert, dim=0)
                if isinstance(heat,list):
                    heat = torch.stack(heat, dim=0)
                    heat = torch.mean(heat, dim=0)
                if isinstance(deg,list):
                    deg = torch.stack(deg, dim=0)
                    deg = torch.mean(deg, dim=0)
                    
            pred = F.interpolate(pred, (h_crop, w_crop), mode='bilinear')
            preds += F.pad(pred.cpu().detach(), (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
            pred_vor = F.interpolate(pred_vor, (h_crop, w_crop), mode='bilinear')
            preds_vor += F.pad(pred_vor.cpu().detach(), (int(x1), int(preds_vor.shape[3] - x2), int(y1), int(preds_vor.shape[2] - y2)))
            cert = F.interpolate(cert, (h_crop, w_crop), mode='bilinear')
            certs += F.pad(cert.cpu().detach(), (int(x1), int(certs.shape[3] - x2), int(y1), int(certs.shape[2] - y2)))
            heat = F.interpolate(heat, (h_crop, w_crop), mode='bilinear')
            heats += F.pad(heat.cpu().detach(), (int(x1), int(heats.shape[3] - x2), int(y1), int(heats.shape[2] - y2)))
            deg = F.interpolate(deg, (h_crop, w_crop), mode='bilinear')
            degs += F.pad(deg.cpu().detach(), (int(x1), int(degs.shape[3] - x2), int(y1), int(degs.shape[2] - y2)))
            
            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    if torch.onnx.is_in_onnx_export():
        # cast count_mat to constant while exporting to ONNX
        count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
    preds = preds / count_mat
    preds_vor = preds_vor/count_mat
    certs = certs / count_mat
    heats = heats / count_mat
    degs = degs / count_mat
    
    if valid_region is not None:
        h_valid = valid_region[0]
        w_valid = valid_region[1]
        preds = preds[:, :, :h_valid, :w_valid]
        preds_vor = preds_vor[:, :, :h_valid, :w_valid]
        certs = certs[:, :, :h_valid, :w_valid]
        heats = heats[:, :, :h_valid, :w_valid]
        degs = degs[:, :, :h_valid, :w_valid]
        
    if rescale:
        preds = resize(preds, size=img_meta['seg_shape'][:2], mode='bilinear', align_corners=False, warning=False)
        preds_vor = resize(preds_vor, size=img_meta['seg_shape'][:2], mode='bilinear', align_corners=False, warning=False)
        certs = resize(certs, size=img_meta['seg_shape'][:2], mode='bilinear', align_corners=False, warning=False)
        heats = resize(heats, size=img_meta['seg_shape'][:2], mode='bilinear', align_corners=False, warning=False)
        deg_h, deg_w = degs.shape[-2:]
        ori_h, ori_w = img_meta['seg_shape'][:2]
        mag_ratio = ori_h/deg_h
        degs = resize(degs, size=img_meta['seg_shape'][:2], mode='bilinear', align_corners=False, warning=False)*mag_ratio

    return preds, preds_vor, certs, heats, degs


def bbox_inference(model, imgs, img_meta, bbox_dict, args):
    B, C, H, W = imgs.shape
    preds = torch.zeros((B, args.net_num_classes, H, W)).cpu().detach()
    certs = torch.zeros((B, 1, H, W)).cpu().detach()
    heats = torch.zeros((B, 1, H, W)).cpu().detach()
    if args.degree_version in ['v4']:
        degs = torch.zeros((B, 4*args.net_N+3, H, W)).cpu().detach()
    else:
        degs = torch.zeros((B, 1, H, W)).cpu().detach()
    counts = torch.zeros((B, 1, H, W)).cpu().detach()
    
    for bbox_idx, bbox_prop in bbox_dict.items():
        # crop the cell
        pt1 = bbox_prop['pt1']
        pt2 = bbox_prop['pt2']
        w = pt2[0]-pt1[0]
        h = pt2[1]-pt1[1]
        w_start = max(0,round(pt1[0]-w*0.2))
        h_start = max(0,round(pt1[1]-h*0.2))
        w_stop = min(W,round(pt2[0]+w*0.2))
        h_stop = min(H,round(pt2[1]+h*0.2))
        img_cropped = imgs[:, :, h_start:h_stop, w_start:w_stop]
        h,w = img_cropped.shape[2:]
        
        # resize the cell
        crop_size = args.crop_size
        canvs = torch.zeros([B, C, crop_size[1], crop_size[0]]).cuda()
        w_ratio = crop_size[0]/w
        h_ratio = crop_size[1]/h
        scale_ratio = min(w_ratio, h_ratio)
        dest_w = round(scale_ratio*w)
        dest_h = round(scale_ratio*h)
        img_cropped = F.interpolate(img_cropped, (dest_h, dest_w), mode='bilinear')
        canvs[:, :, :dest_h, :dest_w] = img_cropped
        
        # predict the cells
        pred, cert, heat, deg = model(canvs)
        if isinstance(pred,list):
            pred = torch.stack(pred, dim=0)
            pred = torch.mean(pred, dim=0)[:, :, :dest_h, :dest_w]
        if isinstance(cert,list):
            cert = torch.stack(cert, dim=0)
            cert = torch.mean(cert, dim=0)[:, :, :dest_h, :dest_w]
        if isinstance(heat,list):
            heat = torch.stack(heat, dim=0)
            heat = torch.mean(heat, dim=0)[:, :, :dest_h, :dest_w]
        if isinstance(deg,list):
            deg = torch.stack(deg, dim=0)
            deg = torch.mean(deg, dim=0)[:, :, :dest_h, :dest_w]
        
        pred = F.interpolate(pred, (h, w), mode='bilinear')
        preds += F.pad(pred.cpu().detach(), (int(w_start), int(preds.shape[3] - w_stop), int(h_start), int(preds.shape[2] - h_stop)))
        cert = F.interpolate(cert, (h, w), mode='bilinear')
        certs += F.pad(cert.cpu().detach(), (int(w_start), int(certs.shape[3] - w_stop), int(h_start), int(certs.shape[2] - h_stop)))
        heat = F.interpolate(heat, (h, w), mode='bilinear')
        heats += F.pad(heat.cpu().detach(), (int(w_start), int(heats.shape[3] - w_stop), int(h_start), int(heats.shape[2] - h_stop)))
        deg = F.interpolate(deg, (h, w), mode='bilinear')/scale_ratio
        degs += F.pad(deg.cpu().detach(), (int(w_start), int(degs.shape[3] - w_stop), int(h_start), int(degs.shape[2] - h_stop)))

        counts[:, :, h_start:h_stop, w_start:w_stop] += 1
        
    valid_region = torch.zeros((B, 1, H, W))
    valid_region[counts > 0] = 1
    
    counts[counts==0] = 1
    preds = preds / counts
    certs = certs / counts
    heats = heats / counts
    degs = degs / counts
    
    
    return preds, certs, heats, degs, valid_region
