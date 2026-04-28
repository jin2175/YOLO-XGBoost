# Ultralytics YOLOv5 + XGBoost Hybrid Validation Script
"""
YOLOv5 + XGBoost 混合检测验证脚本
对比评估:
1. 原始YOLOv5性能
2. 混合检测器性能
3. 性能提升分析

Usage:
    $ python val_hybrid.py --weights yolov5s.pt --data coco128.yaml --xgb-model runs/xgboost_train/exp/xgb_model.json
"""

import argparse
import json
import joblib
import os
import sys

import time
try:
    from thop import profile
except ImportError:
    profile = None

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

'''from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    colorstr,
    increment_path,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
)
from utils.metrics import ap_per_class, box_iou, ConfusionMatrix
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, smart_inference_mode'''


# 添加 YOLOv8 的依赖
from ultralytics import YOLO
#from ultralytics.utils.ops import non_max_suppression, scale_boxes, xywh2xyxy
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes, xywh2xyxy
from ultralytics.utils.metrics import box_iou, ap_per_class
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.files import increment_path
from ultralytics.utils import LOGGER, colorstr
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'

# YOLOv8 的数据加载器专用导入
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_dataloader



# Import hybrid detector
from hybrid_detector import HybridDetector


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def _compute_prf(tp: int, fp: int, fn: int):
    """Compute precision/recall/F1 from counts."""
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


@smart_inference_mode()
def quantize_confidence_yolo(
    model,
    dataloader,
    device,
    iou_thres=0.45,
    max_det=300,
    half=True,
    conf_sweep=None,
    iou_eval=0.5,
    target_recall=0.995,
    save_dir: Optional[Path] = None,
):
    """
    Quantize/justify confidence threshold by sweeping NMS confidence and measuring P/R/F1 + candidate counts.

    This is meant to justify why low_conf_thres (e.g., 0.1) is chosen for the "keep candidates then refine" stage.
    """
    model.eval()
    assert 0.0 < iou_eval <= 1.0
    if conf_sweep is None:
        # Include 0.1 explicitly and provide dense sampling around low thresholds.
        conf_sweep = (
            [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.09, 0.1]
            + [0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        )
    conf_sweep = [float(x) for x in conf_sweep]

    # Precompute a single IoU threshold tensor for correctness calculation.
    iouv = torch.tensor([iou_eval], device=device)

    # Accumulators per threshold
    per_thr = {
        thr: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_images": 0,
            "n_labels": 0,
            "n_det": 0,  # after NMS
            "conf_all": [],  # for confidence distribution (sampled)
        }
        for thr in conf_sweep
    }

    # Iterate dataset once and run NMS for each threshold on the same raw predictions
    pbar = tqdm(dataloader, desc="Quantizing confidence (YOLOv5)", bar_format=TQDM_BAR_FORMAT)
    for _, (im, targets, _, shapes) in enumerate(pbar):
        im = im.to(device, non_blocking=True)
        targets = targets.to(device)
        im = im.half() if half else im.float()
        im /= 255
        nb, _, height, width = im.shape

        # Inference (raw preds)
        preds_raw = model(im)

        # scale targets to pixels
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)

        # Run NMS for each confidence threshold
        for thr in conf_sweep:
            preds = non_max_suppression(
                preds_raw,
                conf_thres=thr,
                iou_thres=iou_thres,
                labels=[],
                multi_label=True,
                agnostic=False,
                max_det=max_det,
            )

            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl = labels.shape[0]
                npr = pred.shape[0]

                # Convert to native-space for IoU correctness calculation
                shape = orig_shapes[si]
                if npr:
                    predn = pred.clone()
                    scale_boxes(im[si].shape[1:], predn[:, :4], shape)
                else:
                    predn = pred

                # Labels to native space
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_boxes(im[si].shape[1:], tbox, shape)
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                else:
                    labelsn = labels

                # Compute TP/FP/FN at iou_eval
                if npr and nl:
                    correct = process_batch(predn, labelsn, iouv)  # (npr, 1)
                    tp = int(correct.any(1).sum().item())
                    fp = int(npr - tp)
                    # For FN, count GT boxes that were not matched by any prediction of correct class at IoU>=thr.
                    # We approximate FN as nl - unique matched labels in process_batch matching.
                    iou = box_iou(labelsn[:, 1:], predn[:, :4])
                    correct_class = labelsn[:, 0:1] == predn[:, 5]
                    matched = (iou >= iou_eval) & correct_class
                    fn = int(nl - matched.any(1).sum().item())
                elif npr and not nl:
                    tp, fp, fn = 0, int(npr), 0
                elif (not npr) and nl:
                    tp, fp, fn = 0, 0, int(nl)
                else:
                    tp, fp, fn = 0, 0, 0

                acc = per_thr[thr]
                acc["tp"] += tp
                acc["fp"] += fp
                acc["fn"] += fn
                acc["n_images"] += 1
                acc["n_labels"] += int(nl)
                acc["n_det"] += int(npr)
                if npr:
                    # store confidences (sampling guard: keep at most ~200k values)
                    if len(acc["conf_all"]) < 200000:
                        acc["conf_all"].extend(pred[:, 4].detach().float().cpu().tolist())

    # Summarize
    rows = []
    best_f1 = -1.0
    best_thr = conf_sweep[0]
    max_recall = 0.0
    for thr in conf_sweep:
        acc = per_thr[thr]
        precision, recall, f1 = _compute_prf(acc["tp"], acc["fp"], acc["fn"])
        max_recall = max(max_recall, recall)
        det_per_img = acc["n_det"] / max(acc["n_images"], 1)
        rows.append(
            {
                "conf_thres": thr,
                "precision_iou{:.2f}".format(iou_eval): precision,
                "recall_iou{:.2f}".format(iou_eval): recall,
                "f1_iou{:.2f}".format(iou_eval): f1,
                "det_per_image": det_per_img,
                "n_images": acc["n_images"],
                "n_labels": acc["n_labels"],
                "tp": acc["tp"],
                "fp": acc["fp"],
                "fn": acc["fn"],
            }
        )
        # Choose best by F1, then by fewer detections (for efficiency)
        if (f1 > best_f1 + 1e-12) or (abs(f1 - best_f1) <= 1e-12 and det_per_img < per_thr[best_thr]["n_det"] / max(per_thr[best_thr]["n_images"], 1)):
            best_f1, best_thr = f1, thr

    # Recall-constrained recommended threshold: smallest thr achieving target_recall * max_recall
    recall_target = float(target_recall) * max_recall
    thr_recall = conf_sweep[0]
    for thr in conf_sweep:
        r = next(x for x in rows if x["conf_thres"] == thr)["recall_iou{:.2f}".format(iou_eval)]
        if r >= recall_target:
            thr_recall = thr
            break

    result = {
        "iou_eval": float(iou_eval),
        "conf_sweep": conf_sweep,
        "max_recall": float(max_recall),
        "target_recall_ratio": float(target_recall),
        "recall_target": float(recall_target),
        "best_f1_thr": float(best_thr),
        "recall_constrained_thr": float(thr_recall),
        "rows": rows,
    }

    # Confidence histogram for selected thresholds (0.1 and recall_constrained_thr)
    hist = {}
    for key_thr in sorted(set([0.1, thr_recall, best_thr])):
        # pick nearest existing sweep threshold
        nearest = min(conf_sweep, key=lambda x: abs(x - key_thr))
        confs = np.array(per_thr[nearest]["conf_all"], dtype=np.float32)
        if confs.size:
            bins = np.linspace(0.0, 1.0, 51)
            counts, edges = np.histogram(confs, bins=bins)
            hist[str(nearest)] = {"bin_edges": edges.tolist(), "counts": counts.tolist(), "n": int(confs.size)}
        else:
            hist[str(nearest)] = {"bin_edges": [], "counts": [], "n": 0}
    result["confidence_hist"] = hist

    # Save outputs
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        csv_path = save_dir / "confidence_sweep.csv"
        headers = list(rows[0].keys()) if rows else []
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for r in rows:
                f.write(",".join(str(r[h]) for h in headers) + "\n")

        # JSON summary
        json_path = save_dir / "confidence_quantization.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"[ConfidenceQuant] Saved: {csv_path}")
        LOGGER.info(f"[ConfidenceQuant] Saved: {json_path}")

    # Print a compact recommendation for paper
    LOGGER.info(
        f"[ConfidenceQuant] iou={iou_eval:.2f} max_recall={max_recall:.4f}, "
        f"best_F1_thr={best_thr}, recall_constrained_thr={thr_recall} "
        f"(target={target_recall:.3f} of max)"
    )
    return result


def apply_soft_nms(bboxes, scores, labels, conf_thres=0.1, sigma=1.0, method='gaussian', linear_iou_thres=0.3):
    """
    改进版 Soft-NMS: 支持更温和的 Gaussian 和 Linear 衰减
    """
    keep_boxes = []
    keep_scores = []
    keep_labels = []
    
    unique_labels = labels.unique()
    for cls in unique_labels:
        mask = (labels == cls)
        c_boxes = bboxes[mask]
        c_scores = scores[mask].clone()
        
        while c_boxes.shape[0] > 0:
            max_idx = torch.argmax(c_scores)
            max_box = c_boxes[max_idx].unsqueeze(0)
            max_score = c_scores[max_idx].item()
            
            keep_boxes.append(max_box)
            keep_scores.append(max_score)
            keep_labels.append(cls.item())
            
            if c_boxes.shape[0] == 1:
                break
                
            ious = box_iou(max_box, c_boxes)[0]
            
            # --- 核心修改：两种更温和的惩罚机制 ---
            if method == 'gaussian':
                # 调高了 sigma (默认为1.0)，衰减变得更平滑
                decay = torch.exp(-(ious * ious) / sigma)
            elif method == 'linear':
                # 线性衰减：只有当 IoU 大于 linear_iou_thres 时才开始惩罚
                decay = torch.ones_like(ious)
                decay_mask = ious > linear_iou_thres
                decay[decay_mask] = 1.0 - ious[decay_mask]
            else:
                decay = torch.ones_like(ious)
            
            c_scores = c_scores * decay
            
            # 移除已保存的最大框
            c_scores[max_idx] = 0.0
            
            # 过滤掉低于阈值的框 (此时衰减没那么狠了，能保住更多 0.1~0.2 的真实目标)
            keep_mask = c_scores > conf_thres
            c_boxes = c_boxes[keep_mask]
            c_scores = c_scores[keep_mask]
            
    if len(keep_boxes) == 0:
        return torch.zeros((0, 6), device=bboxes.device)
        
    keep_boxes = torch.cat(keep_boxes, dim=0)
    keep_scores = torch.tensor(keep_scores, device=bboxes.device, dtype=torch.float32).unsqueeze(1)
    keep_labels = torch.tensor(keep_labels, device=bboxes.device, dtype=torch.float32).unsqueeze(1)
    
    return torch.cat([keep_boxes, keep_scores, keep_labels], dim=1)


@smart_inference_mode()
def validate_soft_nms(
    model, dataloader, device, conf_thres=0.1, max_det=300, sigma=0.5, nc=80, names=None, half=True
):
    """
    验证 Soft-NMS 后处理的性能
    """
    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    seen = 0
    stats = []
    
    
    # --- 新增：计时变量 ---
    total_time = 0.0
    total_images = 0
    
    pbar = tqdm(dataloader, desc="Validating Soft-NMS", bar_format=TQDM_BAR_FORMAT)
    for batch_i, batch in enumerate(pbar):
        im = batch['img'].to(device, non_blocking=True)
        im = im.half() if half else im.float()
        im /= 255
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1).to(device)
        orig_shapes = batch['ori_shape']
        
        
        
        
        
        
        nb, _, height, width = im.shape
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        
        # --- 新增：计时开始 ---
        if device.type != 'cpu':
            torch.cuda.synchronize(device)
        t_start = time.time()
        
        
        preds = model(im)
        
        # 魔法：设置 iou_thres=1.0，绕过标准 NMS 的抑制机制，提取所有有效候选框
        raw_preds = non_max_suppression(
            preds, conf_thres, iou_thres=1.0, labels=[], multi_label=True, agnostic=False, max_det=10000
        )
        
        
        # --- 新增：计时结束 ---
        if device.type != 'cpu':
            torch.cuda.synchronize(device)
        total_time += (time.time() - t_start)
        total_images += nb
        
        # 应用 Soft-NMS
        soft_preds = []
        for det in raw_preds:
            if len(det) > 0:
                soft_det = apply_soft_nms(det[:, :4], det[:, 4], det[:, 5], conf_thres=conf_thres, method='linear', linear_iou_thres=0.45)
                if len(soft_det) > 0:
                    soft_det = soft_det[soft_det[:, 4].argsort(descending=True)[:max_det]]
                soft_preds.append(soft_det)
            else:
                soft_preds.append(det)

        for si, pred in enumerate(soft_preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            shape = orig_shapes[si]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1
            
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue
            
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape)
            
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shape)
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)
            
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
    
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    if len(stats) and stats[0].any():
        # 注意：这里和之前一样加上了 , *rest 防止解包报错
        tp, fp, p, r, f1, ap, ap_class, *rest = ap_per_class(*stats, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
    
    
    # --- 新增：计算 FPS ---
    fps = total_images / total_time if total_time > 0 else 0.0
    
    return {'precision': mp, 'recall': mr, 'mAP50': map50, 'mAP50-95': map, 'seen': seen, 'fps': fps}



@smart_inference_mode()
def validate_yolo_only(
    model,
    dataloader,
    device,
    conf_thres=0.1,#0.001
    iou_thres=0.45,#0.6
    max_det=300,
    nc=80,
    names=None,
    compute_loss=None,
    half=True,
):
    """
    验证原始YOLOv5性能
    """
    model.eval()
    
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    
    seen = 0
    stats = []
    
    total_time = 0.0
    total_images = 0
    
    pbar = tqdm(dataloader, desc="Validating YOLOv5", bar_format=TQDM_BAR_FORMAT)
    
    for batch_i, batch in enumerate(pbar):
        im = batch['img'].to(device, non_blocking=True)
        im = im.half() if half else im.float()
        im /= 255
        nb, _, height, width = im.shape
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1).to(device)
        orig_shapes = batch['ori_shape']
        
        # --- 新增：计时开始 ---
        if device.type != 'cpu':
            torch.cuda.synchronize(device)
        t_start = time.time()
        
        
        # Inference
        preds = model(im)
        
        
        # NMS - 确保targets在正确的设备上
        targets = targets.to(device)
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        #3preds = non_max_suppression(preds, conf_thres, iou_thres, max_det=max_det)
         # 使用与val.py相同的NMS参数，确保结果一致
        '''4lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]  # for autolabelling
        preds = non_max_suppression(
            preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False, max_det=max_det
        )'''
        
        
        preds = non_max_suppression(    
            preds, conf_thres, iou_thres,     
            labels=[],  # 空列表，不使用真实标签    
            multi_label=True,  # 与val.py一致    
            agnostic=False,     
            max_det=max_det)
            
        # --- 新增：计时结束 ---
        if device.type != 'cpu':
            torch.cuda.synchronize(device)
        total_time += (time.time() - t_start)
        total_images += nb
        
        

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            shape = orig_shapes[si]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1
            
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue
            
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape)
            
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shape)
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)
            
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
    
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class, *rest = ap_per_class(*stats, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
    
    
    # --- 新增：计算 FPS ---
    fps = total_images / total_time if total_time > 0 else 0.0
    
    return {
        'precision': mp,
        'recall': mr,
        'mAP50': map50,
        'mAP50-95': map,
        'seen': seen,
        'fps': fps,
    }


@smart_inference_mode()
def validate_hybrid(
    hybrid_detector,
    dataloader,
    device,
    imgsz=640,
    max_det=300,
    nc=80,
    names=None,
    half=True,
    ml_type='xgboost',  # <--- 新增
    ml_model=None,      # <--- 新增
    scaler=None,        # <--- 新增
):
    """
    验证混合检测器性能
    """
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    
    seen = 0
    stats = []
    
    # 统计信息
    total_yolo_high = 0
    total_xgb_refined = 0
    total_xgb_kept = 0
    
    # --- 新增：计时变量 ---
    total_time = 0.0
    total_images = 0
    
    pbar = tqdm(dataloader, desc="Validating Hybrid", bar_format=TQDM_BAR_FORMAT)
    
    for batch_i, batch in enumerate(pbar):
        im = batch['img'].to(device, non_blocking=True)
        im = im.half() if half else im.float()
        im /= 255
        nb, _, height, width = im.shape
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1).to(device)
        orig_shapes = batch['ori_shape']
        
        # 确保targets在正确的设备上
        targets = targets.to(device)
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        
        
        # ==================== 性能优化与计时开始 ====================
        if device.type != 'cpu':
            torch.cuda.synchronize(device)
        t_start = time.time()
        
        # 1. 批量并行推理 YOLO (移出了单图循环)
        preds_raw = hybrid_detector.yolo_model(im)
        
        # 2. 批量第一轮 NMS
        batch_all_dets = non_max_suppression(
            preds_raw,
            hybrid_detector.low_conf_thres,
            hybrid_detector.iou_thres,
            max_det=max_det
        )
        
        # 对每张图像进行混合检测
        for si in range(nb):
            labels = targets[targets[:, 0] == si, 1:]
            shape = orig_shapes[si]
            
            
            # 直接从批量结果中取单图候选框
            all_det = batch_all_dets[si]
            
            
            
            
            
            
            
            
            if len(all_det) == 0:
                seen += 1
                if labels.shape[0]:
                    stats.append((torch.zeros((0, niou), dtype=torch.bool, device=device), 
                                torch.zeros(0, device=device),
                                torch.zeros(0, device=device),
                                labels[:, 0]))
                continue
            
            # 分流
            conf = all_det[:, 4]
            high_conf_mask = conf >= hybrid_detector.mid_conf_thres
            low_conf_mask = (conf >= hybrid_detector.low_conf_thres) & (conf < hybrid_detector.mid_conf_thres)
            
            high_conf_dets = all_det[high_conf_mask]
            low_conf_dets = all_det[low_conf_mask]
            
            total_yolo_high += len(high_conf_dets)
            total_xgb_refined += len(low_conf_dets)
            
            # XGBoost处理低置信度检测
            if len(low_conf_dets) > 0 and hybrid_detector.xgb_model is not None:
                # 先缩放到原图尺寸再提取特征
                low_conf_scaled = low_conf_dets.clone()
                scale_boxes(im.shape[2:], low_conf_scaled[:, :4], shape) # <--- 修改这里
                
                
                
                # ======= 新增：支持多模型分流推理 =======
                if ml_type == 'xgboost':
                    refined_dets = hybrid_detector.xgboost_refinement(low_conf_scaled, shape)
                else:
                    # 1. 提取特征 (调用 HybridDetector 内部特征提取方法)
                    features = hybrid_detector.extract_features(low_conf_scaled, shape)
                    
                    # 转换为 numpy 数组以便 sklearn 处理
                    if isinstance(features, torch.Tensor):
                        features = features.cpu().numpy()
                    
                    # 2. 如果是 SVM，必须通过相同的 Scaler 进行标准化
                    if ml_type == 'svm' and scaler is not None:
                        features = scaler.transform(features)
                    
                    # 3. sklearn 模型推理，获取正类(真实目标)的概率
                    probs = ml_model.predict_proba(features)[:, 1]
                    
                    # 4. 更新置信度
                    refined_dets = low_conf_scaled.clone()
                    refined_dets[:, 4] = torch.tensor(probs, device=device, dtype=refined_dets.dtype)
                # ==========================================
                
                # 过滤
                keep_mask = refined_dets[:, 4] >= hybrid_detector.high_conf_thres
                refined_dets = refined_dets[keep_mask]
                total_xgb_kept += len(refined_dets)
                
                
    
                
                # 缩放回模型尺寸用于NMS
                if len(refined_dets) > 0:
                    # 已经在原图尺寸，不需要再缩放
                    pass
            else:
                refined_dets = torch.zeros((0, 6), device=device)
            
            # 融合
            if len(high_conf_dets) > 0:
                high_conf_scaled = high_conf_dets.clone()
                scale_boxes(im.shape[2:], high_conf_scaled[:, :4], shape) # <--- 修改这里
                
                
            else:
                high_conf_scaled = torch.zeros((0, 6), device=device)
            
            if len(high_conf_scaled) > 0 and len(refined_dets) > 0:
                merged_dets = torch.cat([high_conf_scaled, refined_dets], dim=0)
            elif len(high_conf_scaled) > 0:
                merged_dets = high_conf_scaled
            elif len(refined_dets) > 0:
                merged_dets = refined_dets
            else:
                merged_dets = torch.zeros((0, 6), device=device)
            
            # 最终NMS
            if len(merged_dets) > 0:
                final_dets = hybrid_detector._final_nms(merged_dets, hybrid_detector.iou_thres, max_det)
            else:
                final_dets = merged_dets
            
            
            
            # ==================== 计时结束 ====================
            if device.type != 'cpu':
                torch.cuda.synchronize(device)
            total_time += (time.time() - t_start)
            total_images += nb
            
            
            seen += 1
            npr = len(final_dets)
            nl = labels.shape[0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue
            
            # 评估
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im.shape[2:], tbox, shape)  # <--- 修改这里
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(final_dets, labelsn, iouv)
            
            stats.append((correct, final_dets[:, 4], final_dets[:, 5], labels[:, 0]))
    
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class, *rest = ap_per_class(*stats, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
    
    
    
    # --- 新增：计算 FPS ---
    fps = total_images / total_time if total_time > 0 else 0.0
    
    
    return {
        'precision': mp,
        'recall': mr,
        'mAP50': map50,
        'mAP50-95': map,
        'seen': seen,
        'yolo_high_conf': total_yolo_high,
        'xgb_refined': total_xgb_refined,
        'xgb_kept': total_xgb_kept,
        'fps': fps,
    }


@smart_inference_mode()
def run_validation(
    weights=ROOT / "yolov5s.pt",
    data=ROOT / "data/coco.yaml",
    xgb_model=None,
    imgsz=640,
    batch_size=32,
    device="",
    workers=8,
    low_conf_thres=0.1,
    mid_conf_thres=0.2,
    high_conf_thres=0.2,#0.25
    iou_thres=0.45,#0.6
    max_det=300,
    half=True,
    project=ROOT / "runs/val_hybrid",
    name="exp",
    exist_ok=False,
    plots=True,
    quantize_conf=False,
    conf_sweep=None,
    conf_iou_eval=0.5,
    conf_target_recall=0.995,
    ml_type="xgboost",
):
    """
    运行混合检测验证
    """
    # 创建保存目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备选择
    device = select_device(device)
    
    # 加载数据
    '''data = check_dataset(data)
    
    # 加载YOLOv5模型
    yolo_model = DetectMultiBackend(weights, device=device, data=data, fp16=half)
    stride = yolo_model.stride
    names = yolo_model.names
    nc = len(names)
    imgsz = check_img_size(imgsz, s=stride)
    
    # 创建混合检测器
    hybrid_detector = HybridDetector(
        weights=weights,
        xgb_model_path=xgb_model,
        device=device,
        data=data,
        half=half,
        low_conf_thres=low_conf_thres,
        mid_conf_thres=mid_conf_thres,
        high_conf_thres=high_conf_thres,
        iou_thres=iou_thres,
    )
    
    # 创建数据加载器
    dataloader = create_dataloader(
        data['val'],
        imgsz,
        batch_size,
        stride,
        single_cls=False,
        pad=0.5,
        rect=True,
        workers=workers,
        prefix=colorstr("val: "),
    )[0]'''
    
    # 1. 初始化 YOLOv8 模型 (去除了所有的 self)
    v8_model = YOLO(weights)
    yolo_model = v8_model.model.to(device)
    if half:
        yolo_model.half()
    yolo_model.eval()
    
    stride = int(yolo_model.stride.max()) if hasattr(yolo_model, 'stride') else 32
    names = v8_model.names
    nc = len(names)
    
    
    
    # 新增：计算模型参数量(M)与运算量(G)
    # ============================================
    params_m = sum(p.numel() for p in yolo_model.parameters()) / 1e6
    gflops = 0.0
    if profile is not None:
        try:
            dummy_im = torch.zeros((1, 3, imgsz, imgsz), device=device)
            dummy_im = dummy_im.half() if half else dummy_im.float()
            macs, _ = profile(yolo_model, inputs=(dummy_im, ), verbose=False)
            gflops = (macs * 2) / 1e9
        except Exception as e:
            LOGGER.warning(f"Failed to compute FLOPs: {e}")
    else:
        LOGGER.warning("thop missing. pip install thop to calculate GFLOPs.")
    
    
    # 2. 准备数据集 (已修正为加载 val 验证集)
    data_info = check_det_dataset(data)
    val_path = data_info['val']  
    
    dataset = YOLODataset(
        img_path=val_path,
        imgsz=imgsz,
        batch_size=batch_size,
        augment=False,  
        data=data_info,
        rect=True
    )
    dataloader = build_dataloader(dataset, batch=batch_size, workers=workers, shuffle=False)

    # 3. 恢复被注释掉的混合检测器实例化
    hybrid_detector = HybridDetector(
        weights=weights,
        xgb_model_path=xgb_model,
        device=device,
        data=data,
        half=half,
        low_conf_thres=low_conf_thres,
        mid_conf_thres=mid_conf_thres,
        high_conf_thres=high_conf_thres,
        iou_thres=iou_thres,
    )
    # 强制将 hybrid_detector 内部的模型替换为我们刚才加载好的纯血 YOLOv8 模型
    hybrid_detector.yolo_model = yolo_model
    
    #拦截并加载 SVM/RF 模型
    ml_model = None
    scaler = None
    if xgb_model and os.path.exists(xgb_model):
        if ml_type == 'xgboost':
            ml_model = hybrid_detector.xgb_model
        else:
            LOGGER.info(f"Loading {ml_type.upper()} model from {xgb_model}...")
            sklearn_data = joblib.load(xgb_model)
            if ml_type == 'svm':
                ml_model = sklearn_data['model']
                scaler = sklearn_data['scaler']
            elif ml_type == 'rf':
                ml_model = sklearn_data
            
            # 强行挂载到 detector 上，欺骗后续代码的非空检查
            hybrid_detector.xgb_model = ml_model
    
    

    # ============================================
    # 0. 置信度量化 (用于解释 low_conf_thres=0.1 的依据)
    # ============================================
    if quantize_conf:
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("Quantizing confidence threshold (sweep) ...")
        LOGGER.info("=" * 60)
        quantize_confidence_yolo(
            yolo_model,
            dataloader,
            device,
            iou_thres=iou_thres,
            max_det=max_det,
            half=half,
            conf_sweep=conf_sweep,
            iou_eval=conf_iou_eval,
            target_recall=conf_target_recall,
            save_dir=save_dir,
        )
    
    # ============================================
    # 1. 验证原始YOLOv5性能
    # ============================================
    LOGGER.info("\n" + "="*60)
    LOGGER.info("Validating Original YOLOv5...")
    LOGGER.info("="*60)
    
    yolo_results = validate_yolo_only(
        yolo_model,
        dataloader,
        device,
        conf_thres=0.1,  # 标准验证使用低阈值 0.001
        iou_thres=iou_thres,
        max_det=max_det,
        nc=nc,
        names=names,
        half=half,
    )
    
    LOGGER.info(f"\nYOLOv5 Results:")
    LOGGER.info(f"  Precision: {yolo_results['precision']:.4f}")
    LOGGER.info(f"  Recall: {yolo_results['recall']:.4f}")
    LOGGER.info(f"  mAP@0.5: {yolo_results['mAP50']:.4f}")
    LOGGER.info(f"  mAP@0.5:0.95: {yolo_results['mAP50-95']:.4f}")
    
    
    
    # ============================================
    # 1.5 验证主流对比方法: Soft-NMS
    # ============================================
    LOGGER.info("\n" + "="*60)
    LOGGER.info("Validating Baseline: Soft-NMS...")
    LOGGER.info("="*60)
    
    soft_nms_results = validate_soft_nms(
        yolo_model, dataloader, device, 
        conf_thres=0.1, max_det=max_det, sigma=0.5, nc=nc, names=names, half=half
    )
    
    LOGGER.info(f"\nSoft-NMS Results:")
    LOGGER.info(f"  Precision: {soft_nms_results['precision']:.4f}")
    LOGGER.info(f"  Recall: {soft_nms_results['recall']:.4f}")
    LOGGER.info(f"  mAP@0.5: {soft_nms_results['mAP50']:.4f}")
    LOGGER.info(f"  mAP@0.5:0.95: {soft_nms_results['mAP50-95']:.4f}")
    
    
    
    
    
    
    # ============================================
    # 2. 验证混合检测器性能
    # ============================================
    LOGGER.info("\n" + "="*60)
    LOGGER.info("Validating Hybrid Detector (YOLOv5 + XGBoost)...")
    LOGGER.info("="*60)
    
    if xgb_model and os.path.exists(xgb_model):
        hybrid_results = validate_hybrid(
            hybrid_detector,
            dataloader,
            device,
            imgsz=imgsz,
            max_det=max_det,
            nc=nc,
            names=names,
            half=half,
            ml_type=ml_type,     # 这三个参数应该在这里！
            ml_model=ml_model,   # 这三个参数应该在这里！
            scaler=scaler,       # 这三个参数应该在这里！
        )
        
        LOGGER.info(f"\nHybrid Detector Results:")
        LOGGER.info(f"  Precision: {hybrid_results['precision']:.4f}")
        LOGGER.info(f"  Recall: {hybrid_results['recall']:.4f}")
        LOGGER.info(f"  mAP@0.5: {hybrid_results['mAP50']:.4f}")
        LOGGER.info(f"  mAP@0.5:0.95: {hybrid_results['mAP50-95']:.4f}")
        LOGGER.info(f"\nDetection Statistics:")
        LOGGER.info(f"  High confidence detections (YOLOv5): {hybrid_results['yolo_high_conf']}")
        LOGGER.info(f"  Low confidence detections refined by XGBoost: {hybrid_results['xgb_refined']}")
        LOGGER.info(f"  XGBoost kept after refinement: {hybrid_results['xgb_kept']}")
    else:
        LOGGER.warning("XGBoost model not found, skipping hybrid validation")
        hybrid_results = None
    
    
    # ============================================
    # 3. 性能对比分析 (消融实验三方对比)
    # ============================================
    LOGGER.info("\n" + "="*80)
    LOGGER.info("Ablation Study: Performance Comparison")
    LOGGER.info("="*80)
    
    comparison = {
        'yolo': yolo_results,
        'soft_nms': soft_nms_results,
        'hybrid': hybrid_results,
    }
    
    if hybrid_results:
        # 打印三方对比表格
        LOGGER.info(f"{'Metric':<15} {'Orig YOLO':<15} {'Soft-NMS':<15} {'Hybrid(Ours)':<15} {'Ours vs Orig':<15}")
        LOGGER.info("-" * 80)
        
        
        # 1. 打印参数量和运算量
        LOGGER.info(f"{'Params (M)':<15} {params_m:<15.2f} {params_m:<15.2f} {params_m:<15.2f} -")
        # 备注：由于 XGBoost 推理在 CPU 上且仅产生微量判断操作，GFLOPs 增量极小，可视作相等
        LOGGER.info(f"{'GFLOPs (G)':<15} {gflops:<15.2f} {gflops:<15.2f} ~{gflops:<14.2f} -")
        
        # 2. 打印 FPS
        fps_orig = yolo_results.get('fps', 0)
        fps_soft = soft_nms_results.get('fps', 0)
        fps_ours = hybrid_results.get('fps', 0)
        LOGGER.info(f"{'FPS':<15} {fps_orig:<15.1f} {fps_soft:<15.1f} {fps_ours:<15.1f} -")
        LOGGER.info("-" * 80)

        # 3. 打印精度指标
        
        
        metrics = [
            ('Precision', 'precision'), 
            ('Recall', 'recall'), 
            ('mAP@0.5', 'mAP50'), 
            ('mAP@0.5:0.95', 'mAP50-95')
        ]
        
        for name, key in metrics:
            orig = yolo_results[key]
            soft = soft_nms_results[key]
            ours = hybrid_results[key]
            imp = (ours - orig) / max(orig, 1e-6) * 100
            
            LOGGER.info(f"{name:<15} {orig:<15.4f} {soft:<15.4f} {ours:<15.4f} {imp:+.2f}%")
        LOGGER.info("-" * 80)
    
    # 保存结果
    results_file = save_dir / "results.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    comparison = convert_to_serializable(comparison)
    
    with open(results_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    LOGGER.info(f"\nResults saved to {colorstr('bold', save_dir)}")
    
    return comparison


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov8m.pt", help="YOLOv5 model path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco.yaml", help="dataset.yaml path")
    parser.add_argument("--xgb-model", type=str, default=None, help="XGBoost model path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")#32
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument("--low-conf-thres", type=float, default=0.1, help="low confidence threshold")
    parser.add_argument("--mid-conf-thres", type=float, default=0.2, help="mid confidence threshold")
    parser.add_argument("--high-conf-thres", type=float, default=0.2, help="high confidence threshold")#0.25
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")#0.6
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--project", default=ROOT / "runs2/val_hybrid6", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--plots", action="store_true", help="save plots")
    # Confidence quantization (for paper justification of low_conf_thres, e.g. 0.1)
    parser.add_argument("--quantize-conf", action="store_true", help="sweep conf_thres and save P/R/F1 vs threshold")
    parser.add_argument("--ml-type", type=str, default="xgboost", choices=["xgboost", "svm", "rf"], help="Type of ML model loaded")
    parser.add_argument(
        "--conf-sweep",
        type=str,
        default="",
        help="comma-separated confidence thresholds, e.g. '0.001,0.01,0.05,0.1,0.2,0.3'",
    )
    parser.add_argument("--conf-iou-eval", type=float, default=0.5, help="IoU used in sweep metrics (default 0.5)")
    parser.add_argument(
        "--conf-target-recall",
        type=float,
        default=0.995,
        help="recommend smallest threshold achieving this ratio of max recall (default 0.995)",
    )
    opt = parser.parse_args()
    if opt.conf_sweep:
        opt.conf_sweep = [float(x.strip()) for x in opt.conf_sweep.split(",") if x.strip()]
    else:
        opt.conf_sweep = None
    return opt


def main(opt):
    #check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run_validation(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

