# -*- coding=utf-8 -*-
'''
给出实例评价指标（语义分割指标），包括AJI、Diceobj、DQ（F1-score）、SQ、PQ、HD、、ACC、IOU、precision、sensitivity、
'''
from scipy.ndimage.measurements import center_of_mass
import numpy as np
import sklearn.metrics as metrics
from PIL import Image
import cv2
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff as hausdorff
import torch
import pickle
from tqdm import tqdm

class EvaluationInstance(object):
    # 测试类
    def __init__(self, is_save, predict, truth):
        self.is_save = is_save  # 是否存储预测图像
        # predict 和 truth转换，每张图不超过65536个细胞
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy().astype(np.uint16)  # 如果是torch先变成numpy
            truth = truth.detach().cpu().numpy().astype(np.uint16)
        else:
            predict = predict.astype(np.uint16)
            truth = truth.astype(np.uint16)
        self.predict = predict
        self.truth = truth

        # 创建mask list
        true_id_list = list(np.unique(truth))  # 预测列表和gt列表
        pred_id_list = list(np.unique(predict))
        assert max(true_id_list) == len(true_id_list) - 1
        assert max(pred_id_list) == len(pred_id_list) - 1  # 保证连续编号
        # 可以看到mask list的第一个元素是None（因为背景id=0，不需要mask）
        true_masks = [
            None,
        ]
        print("List All Masks!")
        for t in tqdm(true_id_list[1:]):
            t_mask = np.array(truth == t, np.uint8)  # binary mask
            # 压扁
            t_mask = np.max(t_mask, axis=2)  # h w
            true_masks.append(t_mask.astype(np.uint8))

        pred_masks = [
            None,
        ]
        for p in tqdm(pred_id_list[1:]):
            p_mask = np.array(predict == p, np.uint8)  # binary mask
            # 压扁而且还需要填充
            p_mask = np.max(p_mask, axis=2)  # h w
            p_mask = self._fill_(p_mask)  # 填充连通域的空洞
            # 是否考虑一下如果出现大于一个连通域的情况清除面积较少的连通域？？？？？？？
            pred_masks.append(p_mask.astype(np.uint8))

        self.true_id_list = true_id_list
        self.pred_id_list = pred_id_list
        self.true_masks = true_masks
        self.pred_masks = pred_masks



    def __call__(self, save_path):
        # predict truth 均为1000*1000*10
        assert self.predict.shape == self.truth.shape, "evaluation shape mismatch"

        [dq, sq, pq], _ = self._PQ_DQ_SQ_()
        aji_score = self._AJI_()
        Dice_obj, IoU_obj, Hausdorff = self._dice_iou_hausdorff_obj_()

        if self.is_save:
            predict = self.predict.copy()
            truth = self.truth.copy()
            h, w, c = predict.shape
            save_img = np.zeros((h, w, 3))
            predict[predict > 0] = 255
            truth[truth > 0] = 255
            save_img[:, :, 0] = predict[:, :, 0]
            save_img[:, :, 1] = truth[:, :, 0]
            Image.fromarray(save_img.astype(np.uint8)).convert("RGB").save(save_path)

        return dq, sq, pq, aji_score, Dice_obj, IoU_obj, Hausdorff

    def _PQ_DQ_SQ_(self, match_iou=0.5):
        """`match_iou` is the IoU threshold level to determine the pairing between
        GT instances `p` and prediction instances `g`. `p` and `g` is a pair
        if IoU > `match_iou`. However, pair of `p` and `g` must be unique
        (1 prediction instance to 1 GT instance mapping).

        If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
        in bipartite graphs) is caculated to find the maximal amount of unique pairing.

        If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
        the number of pairs is also maximal.

        Fast computation requires instance IDs are in contiguous orderding
        i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
        and `by_size` flag has no effect on the result.

        Returns:
            [dq, sq, pq]: measurement statistic

            [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                          pairing information to perform measurement

        """
        print("Is calculating PQ_DQ_SQ ...")

        assert match_iou >= 0.0, "Cant' be negative"

        true_id_list = self.true_id_list.copy()
        pred_id_list = self.pred_id_list.copy()
        true_masks = self.true_masks.copy()
        pred_masks = self.pred_masks.copy()
        pred = self.predict.copy()

        # prefill with value
        pairwise_iou = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )

        # caching pairwise iou

        for true_id in tqdm(true_id_list[1:]):  # 0-th is background  对于每一个gt
            t_mask = true_masks[true_id]
            pred_true_overlap = pred[t_mask > 0, :]  # 挖
            pred_true_overlap_id = np.unique(pred_true_overlap).astype(np.uint16)
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:  # 对于与之重叠的每一个pred，算他们的iou
                if pred_id == 0:  # ignore
                    continue  # overlaping background
                p_mask = pred_masks[pred_id]
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                iou = inter / (total - inter)
                pairwise_iou[true_id - 1, pred_id - 1] = iou
        #
        if match_iou >= 0.5:  # 走这个，设定的是0.5
            # 注意：这个基于的定理是
            # 当所有的prediction instance之间都没有overlapping，
            # 那么每一个GT instance都最多只存在一个IoU大于0.5的prediction instance与之配对。
            # 因此设置配准的阈值match_iou为0.5！！！！！但我们的多少有重叠，所以有点问题！！！！
            # 如果指标明显不对再改
            paired_iou = pairwise_iou[pairwise_iou > match_iou]  # 只有>0.5的才算匹配，才保留；否则清除
            pairwise_iou[pairwise_iou <= match_iou] = 0.0

            # 对于每一个gt（对于每一行），最多只有一个pred与之匹配，保留最大的那个，其他全部清零
            keep_id_list = np.argmax(pairwise_iou, axis=1)
            tmp_iou = np.zeros_like(pairwise_iou)
            tmp_iou[np.arange(tmp_iou.shape[0]), keep_id_list] = pairwise_iou[np.arange(tmp_iou.shape[0]), keep_id_list]
            pairwise_iou = tmp_iou

            paired_true, paired_pred = np.nonzero(pairwise_iou)  # 矩阵中所有匹配的点的坐标
            # assert len(paired_true) == 1, Exception("More than one IoUs exceed 0.5!")
            paired_iou = pairwise_iou[paired_true, paired_pred]  # 记录匹配的iou
            paired_true += 1  # index is instance id - 1
            paired_pred += 1  # hence return back to original
        else:  # * Exhaustive maximal unique pairing  这个就是只获得最大配对
            #### Munkres pairing with scipy library
            # the algorithm return (row indices, matched column indices)
            # if there is multiple same cost in a row, index of first occurence
            # is return, thus the unique pairing is ensure
            # inverse pair to get high IoU as minimum
            paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
            ### extract the paired cost and remove invalid pair
            paired_iou = pairwise_iou[paired_true, paired_pred]

            # now select those above threshold level
            # paired with iou = 0.0 i.e no intersection => FP or FN
            paired_true = list(paired_true[paired_iou > match_iou] + 1)
            paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
            paired_iou = paired_iou[paired_iou > match_iou]

        # get the actual FP and FN
        unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]  # 没配对的gt，就是FN
        unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]  # 没配对的预测就是FP
        # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

        #
        tp = len(paired_true)
        fp = len(unpaired_pred)
        fn = len(unpaired_true)
        # get the F1-score i.e DQ
        dq = tp / (tp + 0.5 * fp + 0.5 * fn)
        # get the SQ, no paired has 0 iou so not impact
        sq = paired_iou.sum() / (tp + 1.0e-6)

        # 返回DQ（F1）、SQ、PQ，还有详细的匹配信息
        return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]

    def _AJI_(self):
        print('Is calculating AJI ...')

        true_id_list = self.true_id_list.copy()
        pred_id_list = self.pred_id_list.copy()
        true_masks = self.true_masks.copy()
        pred_masks = self.pred_masks.copy()
        pred = self.predict.copy()
        # prefill with value
        pairwise_inter = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )  # 设计了两个关系矩阵
        pairwise_union = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )

        # caching pairwise

        for true_id in tqdm(true_id_list[1:]):  # 0-th is background
            t_mask = true_masks[true_id]  # 对于每一个truth
            pred_true_overlap = pred[t_mask > 0, :]  # 挖一下  hw 10
            pred_true_overlap_id = np.unique(pred_true_overlap).astype(np.uint16)  # 看看pred上有几个细胞和这个gt产生交集
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:  #
                if pred_id == 0:  # ignore
                    continue  # overlaping background
                p_mask = pred_masks[pred_id]  # 预测列表中取出来该mask
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                pairwise_inter[true_id - 1, pred_id - 1] = inter
                pairwise_union[true_id - 1, pred_id - 1] = total - inter
        # 这样算出来对于每个t_mask都会可能有多个对应的pred————————这里往后都没看了
        pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
        #### Munkres pairing to find maximal unique pairing 选最大匹配的
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]
        # now select all those paired with iou != 0.0 i.e have intersection
        paired_true = paired_true[paired_iou > 0.0]
        paired_pred = paired_pred[paired_iou > 0.0]
        paired_inter = pairwise_inter[paired_true, paired_pred]
        paired_union = pairwise_union[paired_true, paired_pred]
        paired_true = list(paired_true + 1)  # index to instance ID
        paired_pred = list(paired_pred + 1)
        overall_inter = paired_inter.sum()
        overall_union = paired_union.sum()
        # add all unpaired GT and Prediction into the union
        unpaired_true = np.array(
            [idx for idx in true_id_list[1:] if idx not in paired_true]
        )
        unpaired_pred = np.array(
            [idx for idx in pred_id_list[1:] if idx not in paired_pred]
        )
        for true_id in unpaired_true:
            # 这里有悖公式？？不应该加上的，但是别人评测代码里有没匹配上的gt这一项（而且都有），师姐可以去掉这一项看看影响（估计没有影响）
            overall_union += true_masks[true_id].sum()
        for pred_id in unpaired_pred:
            overall_union += pred_masks[pred_id].sum()
        #
        aji_score = overall_inter / overall_union
        return aji_score

    def _dice_iou_hausdorff_obj_(self, hausdorff_flag=True):
        """ Compute the object-level metrics between predicted and
            groundtruth: dice, iou, hausdorff """
        print('Is calculating dice, iou, hausdorff ...')

        pred = self.predict.copy()
        gt = self.truth.copy()
        gt_id_list = self.true_id_list.copy()
        pred_id_list = self.pred_id_list.copy()
        gt_masks = self.true_masks.copy()
        pred_masks = self.pred_masks.copy()


        # pred_labeled = label(pred, connectivity=2)
        Ns = len(pred_id_list) - 1  # 总共有多少个细胞
        # gt_labeled = label(gt, connectivity=2)
        Ng = len(gt_id_list) - 1

        # --- compute dice, iou, hausdorff --- #
        pred_objs_area = np.sum(pred > 0)  # total area of objects in image
        gt_objs_area = np.sum(gt > 0)  # total area of objects in groundtruth gt

        # compute how well groundtruth object overlaps its segmented object
        dice_g = 0.0
        iou_g = 0.0
        hausdorff_g = 0.0

        for i in tqdm(range(1, Ng + 1)):  # 对于每一个gt
            gt_i = gt_masks[i]  # hw 0-1
            overlap_parts = pred[gt_i > 0, :]  # 挖出来

            # get intersection objects numbers in image
            obj_no = np.unique(overlap_parts)
            obj_no = obj_no[obj_no != 0]  # 得到重复的id
            # 这里多少也有点问题！！！！！因为重叠，所以除以的总像素数应该是instmask的像素数而不是binary mask的
            gamma_i = float(np.sum(gt_i)) / gt_objs_area  # 按照该细胞像素数权重作为最后的加权系数（这个是按照公式来的）

            # show_figures((pred_labeled, gt_i, overlap_parts))

            if obj_no.size == 0:  # no intersection object  就是这个gt没有和预测产生交集
                dice_i = 0
                iou_i = 0

                # find nearest segmented object in hausdorff distance
                if hausdorff_flag:  # 就寻找最近的预测
                    min_haus = 1e3

                    # find overlap object in a window [-50, 50]
                    pred_cand_indices = self.find_candidates(gt_i, pred)  # 返回挖到的pred的indice列表

                    for j in pred_cand_indices:  # 对于每一个pred
                        pred_j = pred_masks[j]
                        seg_ind = np.argwhere(pred_j)  # 形成坐标点集
                        gt_ind = np.argwhere(gt_i)
                        haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])  # 计算这个pred和gt的hd距离

                        if haus_tmp < min_haus:
                            min_haus = haus_tmp
                    haus_i = min_haus  # 取最小的hd距离
            else:
                # find max overlap object
                obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
                seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number  取重叠最大的来计算指标
                pred_i = pred_masks[seg_obj]  # segmented object

                overlap_area = np.max(obj_areas)  # overlap area  重叠部分的像素数
                # 取匹配最大的来计算dice和iou指数，也就是说这里的dice和iou是每个gt中细胞的dice和iou
                # （下面就变成pred中的了，最后取一个平均）
                dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
                iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)
                if np.sum(pred_i) + np.sum(gt_i) - overlap_area == 0:
                    print(np.sum(pred_i), np.sum(gt_i), overlap_area)

                # compute hausdorff distance
                if hausdorff_flag:
                    seg_ind = np.argwhere(pred_i)
                    gt_ind = np.argwhere(gt_i)
                    haus_i = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

            dice_g += gamma_i * dice_i  # 这里可能会有一个问题，细胞的重叠可能导致gamma_i的和略大于1，以后再去解决！！！！！！
            iou_g += gamma_i * iou_i
            if hausdorff_flag:
                hausdorff_g += gamma_i * haus_i

        # compute how well segmented object overlaps its groundtruth object
        dice_s = 0.0
        iou_s = 0.0
        hausdorff_s = 0.0
        for j in tqdm(range(1, Ns + 1)):  # 和上面同理
            pred_j = pred_masks[j]
            overlap_parts = gt[pred_j > 0, :]

            # get intersection objects number in gt
            obj_no = np.unique(overlap_parts)
            obj_no = obj_no[obj_no != 0]

            # show_figures((pred_j, gt_labeled, overlap_parts))

            sigma_j = float(np.sum(pred_j)) / pred_objs_area  # 这里一样会有点问题！！！！
            # no intersection object
            if obj_no.size == 0:
                dice_j = 0
                iou_j = 0

                # find nearest groundtruth object in hausdorff distance
                if hausdorff_flag:
                    min_haus = 1e3

                    # find overlap object in a window [-50, 50]
                    gt_cand_indices = self.find_candidates(pred_j, gt)

                    for i in gt_cand_indices:
                        gt_i = gt_masks[i]
                        seg_ind = np.argwhere(pred_j)
                        gt_ind = np.argwhere(gt_i)
                        haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                        if haus_tmp < min_haus:
                            min_haus = haus_tmp
                    haus_j = min_haus
            else:
                # find max overlap gt
                gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
                gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
                gt_j = gt_masks[gt_obj]  # groundtruth object

                overlap_area = np.max(gt_areas)  # overlap area

                dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
                iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

                # compute hausdorff distance
                if hausdorff_flag:
                    seg_ind = np.argwhere(pred_j)
                    gt_ind = np.argwhere(gt_j)
                    haus_j = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

            dice_s += sigma_j * dice_j
            iou_s += sigma_j * iou_j
            if hausdorff_flag:
                hausdorff_s += sigma_j * haus_j
        # 返回dice、iou、hd距离指标
        return (dice_g + dice_s) / 2, (iou_g + iou_s) / 2, (hausdorff_g + hausdorff_s) / 2

    def find_candidates(self, obj_i, objects_labeled, radius=50):
        """
        find object indices in objects_labeled in a window centered at obj_i
        when computing object-level hausdorff distance
        """
        if radius > 400:
            return np.array([])

        h, w, _ = objects_labeled.shape  # h w 10
        x, y = center_of_mass(obj_i)
        x, y = int(x), int(y)
        r1 = x - radius if x - radius >= 0 else 0
        r2 = x + radius if x + radius <= h else h
        c1 = y - radius if y - radius >= 0 else 0
        c2 = y + radius if y + radius < w else w
        indices = np.unique(objects_labeled[r1:r2, c1:c2, :])  # 挖，继续挖
        indices = indices[indices != 0]  # 表示挖到的pred的indice列表

        if indices.size == 0:  # 如果没找到，扩大范围继续找
            indices = self.find_candidates(obj_i, objects_labeled, 2 * radius)

        return indices

    def _fill_(self, predict):
        # 填充连通域
        predict *= 255
        h, w = predict.shape
        pre3 = np.zeros((h, w, 3))
        contours, t = cv2.findContours(predict.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.drawContours(pre3, contours, i, (255, 0, 0), -1)
        return pre3[:, :, 0] / 255

    # def _dice2_(self, predict, truth):
    #     pass


if __name__ == '__main__':
    dat_file_dir = 'TCGA-IZ-8196-01A-01-BS1_pre_gt.dat'
    dat_file = open(dat_file_dir, "rb")
    dat_dict = pickle.load(dat_file)
    p, t = dat_dict['predict'], dat_dict['truth']
    eval = EvaluationInstance(is_save=False, predict=p, truth=t)
    dq, sq, pq, aji_score, Dice_obj, IoU_obj, Hausdorff = eval(save_path=None)
    print(f"DQ: {dq}, SQ: {sq}, PQ: {pq}, AJI: {aji_score}, \nDice_obj: {Dice_obj}, IoU_obj: {IoU_obj}, Hausdorff: {Hausdorff}")
