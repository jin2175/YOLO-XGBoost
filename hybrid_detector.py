# Ultralytics YOLOv5 + XGBoost Hybrid Detection System
"""
YOLOv5 + XGBoost 混合检测系统
根据置信度分流检测:
- 置信度 0.1-0.2: 使用XGBoost重新评估和优化
- 置信度 > 0.2: 直接使用YOLOv5结果
- 融合两者结果以提升mAP等指标

Usage:
    $ python hybrid_detector.py --weights yolov5s.pt --source data/images --xgb-model xgb_model.json
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_requirements,
    colorstr,
    increment_path,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
    xywh2xyxy,
)
from utils.torch_utils import select_device, smart_inference_mode


class HybridDetector:
    """
    YOLOv5 + XGBoost 混合检测器
    
    核心思想:
    1. 使用YOLOv5进行初始检测，获取所有候选框
    2. 根据置信度分流：
       - 低置信度(low_conf_thres ~ mid_conf_thres): 用XGBoost重新评估
       - 高置信度(> mid_conf_thres): 直接保留YOLOv5结果
    3. 融合两部分结果，应用NMS得到最终检测
    """
    
    def __init__(
        self,
        weights='yolov5s.pt',
        xgb_model_path=None,
        device='',
        data='data/coco128.yaml',
        half=False,
        dnn=False,
        low_conf_thres=0.1,      # XGBoost处理的最低置信度
        mid_conf_thres=0.2,      # XGBoost和YOLOv5的分界线
        high_conf_thres=0.2,    # 最终输出的置信度阈值,0.25
        iou_thres=0.45,          # NMS IoU阈值
        xgb_boost_factor=1.5,    # XGBoost提升因子
    ):
        """
        初始化混合检测器
        
        Args:
            weights: YOLOv5模型权重路径
            xgb_model_path: XGBoost模型路径 (如果为None则只使用YOLOv5)
            device: 计算设备
            data: 数据集yaml路径
            half: 是否使用FP16
            dnn: 是否使用OpenCV DNN
            low_conf_thres: XGBoost处理的最低置信度阈值
            mid_conf_thres: 置信度分流阈值 (低于此值用XGBoost，高于此值用YOLOv5)
            high_conf_thres: 最终输出的置信度阈值
            iou_thres: NMS IoU阈值
            xgb_boost_factor: XGBoost置信度提升因子
        """
        self.device = select_device(device)
        self.half = half
        self.low_conf_thres = low_conf_thres
        self.mid_conf_thres = mid_conf_thres
        self.high_conf_thres = high_conf_thres
        self.iou_thres = iou_thres
        self.xgb_boost_factor = xgb_boost_factor
        
        # 加载YOLOv5模型
        self.yolo_model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride = self.yolo_model.stride
        self.names = self.yolo_model.names
        self.pt = self.yolo_model.pt
        
        # 加载XGBoost模型
        self.xgb_model = None
        if xgb_model_path and os.path.exists(xgb_model_path):
            self._load_xgboost_model(xgb_model_path)
        
        LOGGER.info(f"HybridDetector initialized:")
        LOGGER.info(f"  - Low conf threshold (XGBoost): {low_conf_thres}")
        LOGGER.info(f"  - Mid conf threshold (split): {mid_conf_thres}")
        LOGGER.info(f"  - High conf threshold (output): {high_conf_thres}")
        LOGGER.info(f"  - XGBoost model: {'Loaded' if self.xgb_model else 'Not loaded'}")
    
    def _load_xgboost_model(self, model_path):
        """加载XGBoost模型"""
        try:
            import xgboost as xgb
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(model_path)
            LOGGER.info(f"XGBoost model loaded from {model_path}")
        except Exception as e:
            LOGGER.warning(f"Failed to load XGBoost model: {e}")
            self.xgb_model = None
    
    def extract_features(self, detections, img_shape, raw_output=None):
        """
        从检测结果中提取XGBoost所需的特征
        
        Args:
            detections: 检测结果 [x1, y1, x2, y2, conf, cls, ...]
            img_shape: 图像尺寸 (H, W)
            raw_output: YOLOv5原始输出 (可选，用于提取更多特征)
        
        Returns:
            features: numpy array of shape (N, num_features)
        """
        if len(detections) == 0:
            return np.array([]).reshape(0, 12)
        
        det = detections.cpu().numpy() if isinstance(detections, torch.Tensor) else detections
        h, w = img_shape[:2]
        
        features = []
        for d in det:
            x1, y1, x2, y2, conf, cls = d[:6]
            
            # 边界框特征
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            aspect_ratio = box_w / (box_h + 1e-6)
            
            # 归一化位置特征
            center_x = (x1 + x2) / 2 / w
            center_y = (y1 + y2) / 2 / h
            norm_w = box_w / w
            norm_h = box_h / h
            norm_area = box_area / (w * h)
            
            # 置信度和类别特征
            conf_feature = conf
            cls_feature = cls
            
            # 边界特征 (检测框是否接近图像边缘)
            edge_dist = min(x1/w, y1/h, (w-x2)/w, (h-y2)/h)
            
            feat = [
                conf_feature,       # 0: 原始置信度
                cls_feature,        # 1: 类别
                center_x,           # 2: 中心x (归一化)
                center_y,           # 3: 中心y (归一化)
                norm_w,             # 4: 宽度 (归一化)
                norm_h,             # 5: 高度 (归一化)
                norm_area,          # 6: 面积 (归一化)
                aspect_ratio,       # 7: 宽高比
                edge_dist,          # 8: 距边缘距离
                box_w,              # 9: 原始宽度
                box_h,              # 10: 原始高度
                box_area,           # 11: 原始面积
            ]
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def xgboost_refinement(self, detections, img_shape):
        """
        使用XGBoost对低置信度检测进行重新评估
        
        Args:
            detections: 低置信度检测结果
            img_shape: 图像尺寸
        
        Returns:
            refined_detections: 经XGBoost重新评估后的检测结果
        """
        if self.xgb_model is None or len(detections) == 0:
            return detections
        
        import xgboost as xgb
        
        # 提取特征
        features = self.extract_features(detections, img_shape)
        
        # XGBoost预测
        dmatrix = xgb.DMatrix(features)
        xgb_probs = self.xgb_model.predict(dmatrix)
        
        # 融合原始置信度和XGBoost预测
        det = detections.clone() if isinstance(detections, torch.Tensor) else torch.tensor(detections)
        original_conf = det[:, 4].cpu().numpy()
        
        # 置信度融合策略: 结合原始置信度和XGBoost预测
        # 如果XGBoost认为是真正目标 (高概率)，则提升置信度
        refined_conf = original_conf * (1 + (xgb_probs - 0.5) * self.xgb_boost_factor)
        refined_conf = np.clip(refined_conf, 0, 1)
        
        det[:, 4] = torch.tensor(refined_conf, device=det.device)
        
        return det
    
    @smart_inference_mode()
    def detect(self, img, imgsz=640, augment=False, max_det=1000):
        """
        执行混合检测
        
        Args:
            img: 输入图像 (BGR numpy array 或 预处理后的tensor)
            imgsz: 推理尺寸
            augment: 是否使用数据增强推理
            max_det: 最大检测数量
        
        Returns:
            detections: 最终检测结果 [x1, y1, x2, y2, conf, cls]
        """
        # 图像预处理
        if isinstance(img, np.ndarray):
            im0 = img.copy()
            img = self._preprocess(img, imgsz)
        else:
            im0 = None
            
        img = img.to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        
        if len(img.shape) == 3:
            img = img[None]
        
        # YOLOv5推理
        pred = self.yolo_model(img, augment=augment)
        
        # 使用低阈值进行NMS，获取所有候选检测
        all_detections = non_max_suppression(
            pred, 
            self.low_conf_thres,  # 使用低阈值
            self.iou_thres, 
            max_det=max_det
        )[0]
        
        if len(all_detections) == 0:
            return torch.zeros((0, 6), device=self.device)
        
        # 根据置信度分流
        conf = all_detections[:, 4]
        
        # 高置信度检测 (直接保留)
        high_conf_mask = conf >= self.mid_conf_thres
        high_conf_dets = all_detections[high_conf_mask]
        
        # 低置信度检测 (需要XGBoost处理)
        low_conf_mask = (conf >= self.low_conf_thres) & (conf < self.mid_conf_thres)
        low_conf_dets = all_detections[low_conf_mask]
        
        # 使用XGBoost重新评估低置信度检测
        if len(low_conf_dets) > 0 and self.xgb_model is not None:
            img_shape = im0.shape if im0 is not None else (img.shape[2], img.shape[3])
            refined_dets = self.xgboost_refinement(low_conf_dets, img_shape)
            
            # 过滤掉XGBoost评估后仍然低于阈值的检测
            refined_mask = refined_dets[:, 4] >= self.high_conf_thres
            refined_dets = refined_dets[refined_mask]
        else:
            refined_dets = torch.zeros((0, 6), device=self.device)
        
        # 融合高置信度和XGBoost优化后的检测
        if len(high_conf_dets) > 0 and len(refined_dets) > 0:
            merged_dets = torch.cat([high_conf_dets, refined_dets], dim=0)
        elif len(high_conf_dets) > 0:
            merged_dets = high_conf_dets
        elif len(refined_dets) > 0:
            merged_dets = refined_dets
        else:
            return torch.zeros((0, 6), device=self.device)
        
        # 对融合结果进行最终NMS
        final_dets = self._final_nms(merged_dets, self.iou_thres, max_det)
        
        return final_dets
    
    def _preprocess(self, img, imgsz):
        """图像预处理"""
        # Letterbox resize
        img = letterbox(img, imgsz, stride=int(self.stride))[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img)
    
    def _final_nms(self, detections, iou_thres, max_det):
        """对融合后的检测结果进行NMS"""
        if len(detections) == 0:
            return detections
        
        boxes = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5]
        
        # Class-wise NMS
        max_wh = 7680
        c = classes * max_wh
        boxes_offset = boxes + c.unsqueeze(1)
        
        keep = torchvision.ops.nms(boxes_offset, scores, iou_thres)
        keep = keep[:max_det]
        
        return detections[keep]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


@smart_inference_mode()
def run_hybrid_detection(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    xgb_model=None,
    data=ROOT / "data/coco.yaml",
    imgsz=(640, 640),
    low_conf_thres=0.1,
    mid_conf_thres=0.2,
    high_conf_thres=0.25,#
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_conf=False,
    nosave=False,
    project=ROOT / "runs/hybrid_detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    half=False,
):
    """
    运行混合检测
    """
    from ultralytics.utils.plotting import Annotator, colors
    
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    
    if is_file:
        source = check_file(source)
    
    # 目录设置
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化混合检测器
    detector = HybridDetector(
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
    
    imgsz = check_img_size(imgsz, s=detector.stride)
    
    # 数据加载
    dataset = LoadImages(source, img_size=imgsz, stride=detector.stride, auto=detector.pt)
    
    # 运行推理
    seen = 0
    dt = (Profile(device=detector.device), Profile(device=detector.device), Profile(device=detector.device))
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(detector.device)
            im = im.half() if detector.half else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
        
        # 混合检测
        with dt[1]:
            det = detector.detect(im0s, imgsz=imgsz[0], max_det=max_det)
        
        seen += 1
        p, im0 = Path(path), im0s.copy()
        save_path = str(save_dir / p.name)
        txt_path = str(save_dir / "labels" / p.stem)
        
        s += f"{im.shape[2]}x{im.shape[3]} "
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        
        annotator = Annotator(im0, line_width=line_thickness, example=str(detector.names))
        
        if len(det):
            # 缩放框到原始图像尺寸
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            
            # 打印结果
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()
                s += f"{n} {detector.names[int(c)]}{'s' * (n > 1)}, "
            
            # 写入结果
            for *xyxy, conf, cls in reversed(det):
                if save_txt:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    with open(f"{txt_path}.txt", "a") as f:
                        f.write(("%g " * len(line)).rstrip() % line + "\n")
                
                if save_img or view_img:
                    c = int(cls)
                    label = f"{detector.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
        
        im0 = annotator.result()
        
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)
        
        if save_img:
            cv2.imwrite(save_path, im0)
        
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")
    
    # 打印结果
    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="YOLOv5 model path")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob")
    parser.add_argument("--xgb-model", type=str, default=None, help="XGBoost model path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--low-conf-thres", type=float, default=0.1, help="low confidence threshold for XGBoost")
    parser.add_argument("--mid-conf-thres", type=float, default=0.2, help="mid confidence threshold (split point)")
    parser.add_argument("--high-conf-thres", type=float, default=0.25, help="high confidence threshold for output")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--project", default=ROOT / "runs/hybrid_detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run_hybrid_detection(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

