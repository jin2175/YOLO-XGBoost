# Ultralytics YOLOv5 + XGBoost Training Script
"""
XGBoost 模型训练脚本
用于训练XGBoost模型来优化YOLOv5低置信度检测

工作流程:
1. 使用YOLOv5对训练集进行推理，获取所有低置信度检测
2. 与真实标签匹配，标记TP/FP
3. 提取特征并训练XGBoost二分类模型
4. 保存模型用于混合检测

Usage:
    $ python train_xgboost.py --weights yolov5s.pt --data data/coco128.yaml --epochs 100
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

#from models.common import DetectMultiBackend
#from utils.callbacks import Callbacks
#from utils.dataloaders import create_dataloader
'''from utils.general import (
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
)'''
#from utils.metrics import box_iou
#from utils.torch_utils import select_device, smart_inference_mode




# 添加 YOLOv8 的依赖
from ultralytics import YOLO
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



class XGBoostTrainer:
    """XGBoost模型训练器"""
    
    def __init__(
        self,
        weights='yolov5s.pt',
        data='data/coco.yaml',
        device='',
        half=False,
        low_conf_thres=0.1,
        mid_conf_thres=0.2,
        iou_thres_match=0.45,  # 用于匹配TP/FP的IoU阈值
    ):
        self.device = select_device(device)
        self.half = half
        self.low_conf_thres = low_conf_thres
        self.mid_conf_thres = mid_conf_thres
        self.iou_thres_match = iou_thres_match
        
        # 加载YOLOv5模型
        #self.model = DetectMultiBackend(weights, device=self.device, data=data, fp16=half)
        #self.stride = self.model.stride
        #self.names = self.model.names
        
        # 初始化 YOLOv8 权重并提取底层的 PyTorch 模型
        v8_model = YOLO(weights)
        self.model = v8_model.model.to(self.device)
        if self.half:
            self.model.half()
        self.model.eval()
        
        # 兼容旧版的属性获取
        self.stride = int(self.model.stride.max()) if hasattr(self.model, 'stride') else 32
        self.names = v8_model.names
        
        
        self.nc = len(self.names)
        
        LOGGER.info(f"XGBoost Trainer initialized:")
        LOGGER.info(f"  - Low conf threshold: {low_conf_thres}")
        LOGGER.info(f"  - Mid conf threshold: {mid_conf_thres}")
        LOGGER.info(f"  - IoU threshold for matching: {iou_thres_match}")
    
    def extract_features(self, detections, img_shape):
        """
        从检测结果中提取特征
        
        Args:
            detections: 检测结果 [x1, y1, x2, y2, conf, cls]
            img_shape: 图像尺寸 (H, W)
        
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
            
            # 边界特征
            edge_dist = min(x1/w, y1/h, (w-x2)/w, (h-y2)/h)
            
            feat = [
                conf_feature,
                cls_feature,
                center_x,
                center_y,
                norm_w,
                norm_h,
                norm_area,
                aspect_ratio,
                edge_dist,
                box_w,
                box_h,
                box_area,
            ]
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def match_detections_to_labels(self, detections, labels, iou_thres=0.5):
        """
        将检测结果与真实标签匹配，确定TP/FP
        
        Args:
            detections: 检测结果 [x1, y1, x2, y2, conf, cls]
            labels: 真实标签 [cls, x1, y1, x2, y2]
            iou_thres: IoU阈值
        
        Returns:
            is_tp: bool array, True if detection is TP
        """
        if len(detections) == 0:
            return np.array([], dtype=bool)
        
        if len(labels) == 0:
            return np.zeros(len(detections), dtype=bool)
        
        det = detections.cpu() if isinstance(detections, torch.Tensor) else torch.tensor(detections)
        lab = labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels)
        
        # 计算IoU
        det_boxes = det[:, :4]
        lab_boxes = lab[:, 1:5]
        iou = box_iou(lab_boxes, det_boxes)
        
        # 检查类别匹配
        det_cls = det[:, 5]
        lab_cls = lab[:, 0]
        correct_class = lab_cls.unsqueeze(1) == det_cls.unsqueeze(0)
        
        # 找到每个检测的最佳匹配
        is_tp = np.zeros(len(detections), dtype=bool)
        matched_labels = set()
        
        # 按置信度排序检测
        conf_order = det[:, 4].argsort(descending=True)
        
        for det_idx in conf_order:
            det_idx = det_idx.item()
            # 找到与该检测匹配的标签 (IoU > 阈值 且 类别匹配)
            valid_matches = (iou[:, det_idx] >= iou_thres) & correct_class[:, det_idx]
            
            if valid_matches.any():
                # 找到IoU最高的未匹配标签
                for lab_idx in iou[:, det_idx].argsort(descending=True):
                    lab_idx = lab_idx.item()
                    if valid_matches[lab_idx] and lab_idx not in matched_labels:
                        is_tp[det_idx] = True
                        matched_labels.add(lab_idx)
                        break
        
        return is_tp
    
    @smart_inference_mode()
    def collect_training_data(
        self,
        dataloader,
        imgsz=640,
        iou_thres_nms=0.45,
        max_det=1000,
    ):
        """
        收集XGBoost训练数据
        
        Args:
            dataloader: 数据加载器
            imgsz: 推理尺寸
            iou_thres_nms: NMS IoU阈值
            max_det: 最大检测数
        
        Returns:
            features: 所有低置信度检测的特征
            labels: 对应的TP/FP标签
        """
        self.model.eval()
        
        all_features = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc="Collecting training data", bar_format=TQDM_BAR_FORMAT)
        
        '''for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            # 准备图像
            im = im.to(self.device, non_blocking=True)
            im = im.half() if self.half else im.float()
            im /= 255
            nb, _, height, width = im.shape
            
            # YOLOv5推理
            pred = self.model(im)'''
            
        for batch_i, batch in enumerate(pbar):
            # 1. 提取图像
            im = batch['img'].to(self.device, non_blocking=True)
            im = im.half() if self.half else im.float()
            im /= 255
            nb, _, height, width = im.shape
            
            # 2. 构造 targets 格式: [batch_idx, cls, x, y, w, h]
            targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1).to(self.device)
            
            # 3. 提取原图尺寸用于缩放
            orig_shapes = batch['ori_shape']
            
            # -------- 核心前向传播 --------
            pred = self.model(im)
            # -----------------------------    
            
            
            
            
            # NMS with low threshold to get all candidates
            pred = non_max_suppression(
                pred,
                self.low_conf_thres,
                iou_thres_nms,
                max_det=max_det
            )
            
            # 处理目标标签
            targets = targets.to(self.device)
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)
            
            # 处理每张图像
            for si, det in enumerate(pred):
                labels_i = targets[targets[:, 0] == si, 1:]
                
                if len(det) == 0:
                    continue
                
                # 只处理低置信度检测 (用于XGBoost训练)
                conf = det[:, 4]
                low_conf_mask = (conf >= self.low_conf_thres) & (conf < self.mid_conf_thres)
                low_conf_det = det[low_conf_mask]
                
                if len(low_conf_det) == 0:
                    continue
                
                # 缩放到原图尺寸
                shape = orig_shapes[si]
                scale_boxes(im[si].shape[1:], low_conf_det[:, :4], shape)
                
                # 转换标签格式
                if len(labels_i) > 0:
                    tbox = xywh2xyxy(labels_i[:, 1:5])
                    scale_boxes(im[si].shape[1:], tbox, shape)
                    labels_n = torch.cat((labels_i[:, 0:1], tbox), 1)
                else:
                    labels_n = torch.zeros((0, 5), device=self.device)
                
                # 匹配TP/FP
                is_tp = self.match_detections_to_labels(
                    low_conf_det, labels_n, self.iou_thres_match
                )
                
                # 提取特征
                features = self.extract_features(low_conf_det, shape)
                
                all_features.append(features)
                all_labels.append(is_tp)
        
        # 合并所有数据
        if len(all_features) > 0:
            all_features = np.vstack(all_features)
            all_labels = np.concatenate(all_labels)
        else:
            all_features = np.array([]).reshape(0, 12)
            all_labels = np.array([], dtype=bool)
        
        LOGGER.info(f"Collected {len(all_features)} samples")
        LOGGER.info(f"  - True Positives: {all_labels.sum()}")
        LOGGER.info(f"  - False Positives: {(~all_labels).sum()}")
        
        return all_features, all_labels
    
    def train_xgboost(
        self,
        features,
        labels,
        save_path='xgb_model.json',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        early_stopping_rounds=10,
        test_size=0.2,
    ):
        """
        训练XGBoost模型
        
        Args:
            features: 特征数据
            labels: 标签数据 (True=TP, False=FP)
            save_path: 模型保存路径
            n_estimators: 树的数量
            max_depth: 最大深度
            learning_rate: 学习率
            early_stopping_rounds: 早停轮数
            test_size: 测试集比例
        
        Returns:
            model: 训练好的XGBoost模型
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        if len(features) == 0:
            LOGGER.warning("No training data available!")
            return None
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels.astype(int), test_size=test_size, random_state=42, stratify=labels
        )
        
        LOGGER.info(f"Training set: {len(X_train)} samples")
        LOGGER.info(f"Test set: {len(X_test)} samples")
        
        # 计算类别权重 (处理不平衡)
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)
        
        LOGGER.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # 创建XGBoost数据集
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # 设置参数
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'scale_pos_weight': scale_pos_weight,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
        }
        
        # 训练模型
        LOGGER.info("Training XGBoost model...")
        evals = [(dtrain, 'train'), (dtest, 'eval')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10,
        )
        
        # 评估模型
        y_pred_prob = model.predict(dtest)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_prob) if len(np.unique(y_test)) > 1 else 0
        
        LOGGER.info(f"\nXGBoost Model Evaluation:")
        LOGGER.info(f"  - Accuracy: {accuracy:.4f}")
        LOGGER.info(f"  - Precision: {precision:.4f}")
        LOGGER.info(f"  - Recall: {recall:.4f}")
        LOGGER.info(f"  - F1 Score: {f1:.4f}")
        LOGGER.info(f"  - AUC: {auc:.4f}")
        
        # 特征重要性
        importance = model.get_score(importance_type='gain')
        feature_names = [
            'confidence', 'class', 'center_x', 'center_y',
            'norm_w', 'norm_h', 'norm_area', 'aspect_ratio',
            'edge_dist', 'box_w', 'box_h', 'box_area'
        ]
        
        LOGGER.info("\nFeature Importance (gain):")
        for i, name in enumerate(feature_names):
            key = f'f{i}'
            imp = importance.get(key, 0)
            LOGGER.info(f"  - {name}: {imp:.4f}")
        
        # 保存模型
        model.save_model(save_path)
        LOGGER.info(f"\nModel saved to {save_path}")
        
        return model
        
    def train_svm(self, features, labels, save_path='svm_model.joblib', test_size=0.2):
        """训练支持向量机(SVM)模型"""
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        if len(features) == 0:
            LOGGER.warning("No training data available!")
            return None
            
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels.astype(int), test_size=test_size, random_state=42, stratify=labels
        )
        
        max_svm_samples = 40000  # 限制最多使用4万个样本训练
        if len(X_train) > max_svm_samples:
            LOGGER.info(f"Subsampling SVM training data from {len(X_train)} to {max_svm_samples} to save time...")
            from sklearn.utils import resample
            X_train, y_train = resample(X_train, y_train, n_samples=max_svm_samples, random_state=42, stratify=y_train)
        
        
        # SVM 对特征尺度极度敏感，必须进行标准化 (Standardization)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        LOGGER.info("Training SVM model...")
        # 使用 rbf 核，开启 probability=True 以便后续做概率过滤，设置 class_weight 处理正负样本不平衡
        model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        # 评估模型
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        LOGGER.info(f"\nSVM Model Evaluation:")
        LOGGER.info(f"  - Accuracy: {accuracy:.4f}")
        LOGGER.info(f"  - Precision: {precision:.4f}")
        LOGGER.info(f"  - Recall: {recall:.4f}")
        LOGGER.info(f"  - F1 Score: {f1:.4f}")
        
        # 将 scaler 和 model 打包一起保存，因为推理时也需要用同一个 scaler 进行预处理
        full_model = {'scaler': scaler, 'model': model}
        joblib.dump(full_model, save_path)
        LOGGER.info(f"\nSVM Model saved to {save_path}")
        
        return full_model

    def train_rf(self, features, labels, save_path='rf_model.joblib', n_estimators=100, max_depth=6, test_size=0.2):
        """训练随机森林(Random Forest)模型"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import joblib
        
        if len(features) == 0:
            LOGGER.warning("No training data available!")
            return None
            
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels.astype(int), test_size=test_size, random_state=42, stratify=labels
        )
        
        LOGGER.info("Training Random Forest model...")
        # 随机森林也是树模型，对量纲不敏感，直接训练即可
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # 调用所有CPU核心加速训练
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        LOGGER.info(f"\nRandom Forest Model Evaluation:")
        LOGGER.info(f"  - Accuracy: {accuracy:.4f}")
        LOGGER.info(f"  - Precision: {precision:.4f}")
        LOGGER.info(f"  - Recall: {recall:.4f}")
        LOGGER.info(f"  - F1 Score: {f1:.4f}")
        
        joblib.dump(model, save_path)
        LOGGER.info(f"\nRandom Forest Model saved to {save_path}")
        
        return model
            
        
        
        


@smart_inference_mode()
def run_training(
    weights=ROOT / "yolov5s.pt",
    data=ROOT / "data/coco.yaml",
    imgsz=640,
    batch_size=16,
    device="",
    workers=8,
    low_conf_thres=0.1,
    mid_conf_thres=0.2,
    iou_thres_nms=0.45,
    iou_thres_match=0.45,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    early_stopping_rounds=10,
    half=False,
    project=ROOT / "runs/xgboost_train",
    name="exp",
    exist_ok=False,
    model_type='xgboost',
):
    """
    运行XGBoost训练
    """
    # 检查依赖
    #check_requirements(['xgboost', 'scikit-learn'])
    
    # 创建保存目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化训练器
    trainer = XGBoostTrainer(
        weights=weights,
        data=data,
        device=device,
        half=half,
        low_conf_thres=low_conf_thres,
        mid_conf_thres=mid_conf_thres,
        iou_thres_match=iou_thres_match,
    )
    
    # 加载数据
    '''data = check_dataset(data)
    imgsz = check_img_size(imgsz, s=trainer.stride)
    
    # 创建数据加载器 (使用训练集)
    train_path = data['train']
    dataloader = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        trainer.stride,
        single_cls=False,
        pad=0.5,
        rect=True,
        workers=workers,
        prefix=colorstr("train: "),
    )[0]'''
    
    
    data_info = check_det_dataset(data)
    train_path = data_info['train']  # 如果是 val_hybrid.py，这里改成 data_info['val']
    
    dataset = YOLODataset(
        img_path=train_path,
        imgsz=imgsz,
        batch_size=batch_size,
        augment=False,  # 提取特征/验证时不需要数据增强
        data=data_info,
        rect=True
    )
    dataloader = build_dataloader(dataset, batch=batch_size, workers=workers, shuffle=False)
    
    
    
    # 收集训练数据
    LOGGER.info("Collecting training data from YOLOv5 predictions...")
    features, labels = trainer.collect_training_data(
        dataloader,
        imgsz=imgsz,
        iou_thres_nms=iou_thres_nms,
    )
    
    if len(features) == 0:
        LOGGER.error("No training data collected! Check your low_conf_thres and mid_conf_thres settings.")
        return None
    
    
    # 根据传入参数选择训练不同的机器学习模型
    #model_type = opt.model_type if hasattr(opt, 'model_type') else 'xgboost'
    
    if model_type == 'xgboost':
        model_path = save_dir / "xgb_model.json"
        model = trainer.train_xgboost(
            features, labels, save_path=str(model_path),
            n_estimators=n_estimators, max_depth=max_depth, 
            learning_rate=learning_rate, early_stopping_rounds=early_stopping_rounds
        )
    elif model_type == 'svm':
        model_path = save_dir / "svm_model.joblib"
        model = trainer.train_svm(features, labels, save_path=str(model_path))
    elif model_type == 'rf':
        model_path = save_dir / "rf_model.joblib"
        model = trainer.train_rf(
            features, labels, save_path=str(model_path),
            n_estimators=n_estimators, max_depth=max_depth
        )
    else:
        LOGGER.error(f"Unsupported model type: {model_type}")
        return None
    
    
    
    # 保存训练配置
    config = {
        'weights': str(weights),
        'data': str(data),
        'low_conf_thres': low_conf_thres,
        'mid_conf_thres': mid_conf_thres,
        'iou_thres_match': iou_thres_match,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'num_samples': len(features),
        'num_tp': int(labels.sum()),
        'num_fp': int((~labels).sum()),
    }
    
    import yaml
    with open(save_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    LOGGER.info(f"\nTraining complete! Results saved to {colorstr('bold', save_dir)}")
    
    return model


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov8m.pt", help="YOLOv5 model path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco.yaml", help="dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument("--low-conf-thres", type=float, default=0.1, help="low confidence threshold")
    parser.add_argument("--mid-conf-thres", type=float, default=0.2, help="mid confidence threshold")
    parser.add_argument("--iou-thres-nms", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--iou-thres-match", type=float, default=0.45, help="IoU threshold for TP/FP matching")
    parser.add_argument("--n-estimators", type=int, default=10, help="number of XGBoost trees")
    parser.add_argument("--max-depth", type=int, default=4, help="max depth of XGBoost trees")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="XGBoost learning rate")
    parser.add_argument("--early-stopping-rounds", type=int, default=10, help="early stopping rounds")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--project", default=ROOT / "runs/xgboost_train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--model-type", type=str, default='xgboost', choices=['xgboost', 'svm', 'rf'], help="choose ml model to train")
    opt = parser.parse_args()
    return opt


def main(opt):
    #check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run_training(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

