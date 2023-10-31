# -*- coding=utf-8 -*-
'''
给出像素级评价指标（语义分割指标），包括F1、IoU、dice（jaccard）、acc、recall（sensitivity）、precision
'''

import numpy as np
import sklearn.metrics as metrics
from PIL import Image
import cv2
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
import torch


class EvaluationPixel(object):
    # 测试类
    def __init__(self, is_save):
        self.is_save = is_save  # 是否存储预测图像

    def __call__(self, predict, truth, save_path):
        # predict truth 均为1000*1000*10
        assert predict.shape == truth.shape, "evaluation shape mismatch"

        dice1 = self._dice1_(predict, truth)

        if self.is_save:
            h, w, c = predict.shape
            save_img = np.zeros((h, w, 3))
            predict[predict > 0] = 255
            truth[truth > 0] = 255
            save_img[:, :, 0] = predict[:, :, 0]
            save_img[:, :, 1] = truth[:, :, 0]
            Image.fromarray(save_img.astype(np.uint8)).convert("RGB").save(save_path)

        return dice1

    def _dice1_(self, predict, truth):
        pre_dice = predict[:, :, 0].copy()  # 计算dice只需要语义分割结果
        truth_dice = truth[:, :, 0].copy()
        pre_dice[pre_dice > 0] = 1
        pre_dice[pre_dice < 1] = 0  # 实例预测结果二值化
        truth_dice[truth_dice > 0] = 1
        truth_dice[truth_dice < 1] = 0
        pre_dice = self._fill_(pre_dice)  # 填充连通域的空洞

        up = (pre_dice * truth_dice).sum()
        down = (pre_dice.sum() + truth_dice.sum())
        dice = 2 * up / down
        # print(up, down)
        return dice

    def _IoU_(self, predict, truth):
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy()  # 如果是torch先变成numpy
            truth = truth.detach().cpu().numpy()
        pre_iou = predict[:, :, 0].copy()  # 计算iou只需要语义分割结果
        truth_iou = truth[:, :, 0].copy()
        pre_iou = np.where(pre_iou > 0, np.ones_like(pre_iou), np.zeros_like(pre_iou))  # 预测和truth的二值化
        truth_iou = np.where(truth_iou > 0, np.ones_like(truth_iou), np.zeros_like(truth_iou))
        pre_iou = self._fill_(pre_iou)  # 填充连通域的空洞——这个我就照着加了！
        iou_pred = jaccard_similarity_score(truth_iou.astype(np.float32).reshape(-1),
                                            pre_iou.astype(np.float32).reshape(-1))
        return iou_pred  # 返回一张图的交并比，这里先不计算分类别的mIOU因为总共就两类没有太大意义

    def _F1_score_(self, predict, truth):
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy()  # 如果是torch先变成numpy
            truth = truth.detach().cpu().numpy()
        pre_iou = predict[:, :, 0].copy()  # 计算iou只需要语义分割结果
        truth_iou = truth[:, :, 0].copy()
        pre_iou = np.where(pre_iou > 0, np.ones_like(pre_iou), np.zeros_like(pre_iou))  # 预测和truth的二值化
        truth_iou = np.where(truth_iou > 0, np.ones_like(truth_iou), np.zeros_like(truth_iou))
        pre_iou = self._fill_(pre_iou)  # 填充连通域的空洞——这个我就照着加了！
        TP = truth_iou * pre_iou  # 预测正确的正例
        FP = pre_iou - TP  # 假阳性——预测错误的负例
        FN = truth_iou - TP  # 没预测出来的正例——假阴性
        precision = TP.sum() / (TP.sum() + FP.sum())
        recall = TP.sum() / (TP.sum() + FN.sum())
        F1_score = (2 * precision * recall) / (precision + recall)
        return F1_score

    def _acc_(self, predict, truth):
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy()  # 如果是torch先变成numpy
            truth = truth.detach().cpu().numpy()
        pre_iou = predict[:, :, 0].copy()  # 计算iou只需要语义分割结果
        truth_iou = truth[:, :, 0].copy()
        pre_iou = np.where(pre_iou > 0, np.ones_like(pre_iou), np.zeros_like(pre_iou))  # 预测和truth的二值化
        truth_iou = np.where(truth_iou > 0, np.ones_like(truth_iou), np.zeros_like(truth_iou))
        pre_iou = self._fill_(pre_iou)  # 填充连通域的空洞——这个我就照着加了！
        return accuracy_score(truth_iou.flatten(), pre_iou.flatten())

    def _recall_(self, predict, truth):
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy()  # 如果是torch先变成numpy
            truth = truth.detach().cpu().numpy()
        pre_iou = predict[:, :, 0].copy()  # 计算iou只需要语义分割结果
        truth_iou = truth[:, :, 0].copy()
        pre_iou = np.where(pre_iou > 0, np.ones_like(pre_iou), np.zeros_like(pre_iou))  # 预测和truth的二值化
        truth_iou = np.where(truth_iou > 0, np.ones_like(truth_iou), np.zeros_like(truth_iou))
        pre_iou = self._fill_(pre_iou)  # 填充连通域的空洞——这个我就照着加了！
        return recall_score(truth_iou.flatten(), pre_iou.flatten())

    def _precision_(self, predict, truth):
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy()  # 如果是torch先变成numpy
            truth = truth.detach().cpu().numpy()
        pre_iou = predict[:, :, 0].copy()  # 计算iou只需要语义分割结果
        truth_iou = truth[:, :, 0].copy()
        pre_iou = np.where(pre_iou > 0, np.ones_like(pre_iou), np.zeros_like(pre_iou))  # 预测和truth的二值化
        truth_iou = np.where(truth_iou > 0, np.ones_like(truth_iou), np.zeros_like(truth_iou))
        pre_iou = self._fill_(pre_iou)  # 填充连通域的空洞——这个我就照着加了！
        return precision_score(truth_iou.flatten(), pre_iou.flatten())

    def _fill_(self, predict):
        # 填充连通域
        predict *= 255
        h, w = predict.shape
        pre3 = np.zeros((h, w, 3))
        contours, t = cv2.findContours(predict.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.drawContours(pre3, contours, i, (255, 0, 0), -1)
        return pre3[:, :, 0] / 255

    def _dice2_(self, predict, truth):
        pass
