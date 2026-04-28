

# 安装XGBoost和scikit-learn
pip install xgboost scikit-learn
pip install -r requirements.txt



#### 1. 训练XGBoost模型

```bash
python train_xgboost.py \
    --weights yolov5s.pt \
    --data coco128.yaml \
    --low-conf-thres 0.1 \
    --mid-conf-thres 0.2 \
    --n-estimators 100 \
    --max-depth 6
```

#### 2. 验证混合检测器性能

```bash
python val_hybrid.py \
    --weights yolov5s.pt \
    --data coco128.yaml \
    --xgb-model runs/xgboost_train/exp/xgb_model.json \
    --low-conf-thres 0.1 \
    --mid-conf-thres 0.2 \
    --high-conf-thres 0.25
```


