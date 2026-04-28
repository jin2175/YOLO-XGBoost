#《面向置信度决策的检测融合设计》


## 声明与致谢

本项目的核心检测框架基于 Ultralytics 官方的 [YOLOv5](https://github.com/ultralytics/yolov5) 代码库进行二次开发。衷心感谢 Ultralytics 团队出色的开源工作。

主要修改与新增内容：
* 遵循开源协议要求，特此声明本项目在原版 YOLOv5 基础上进行了以下修改：
* 新增 hybrid_detector.py: 用于提取 YOLO 骨干/颈部网络特征并与传统分类器对接。
* 新增 train_xgboost.py: 实现了基于提取特征的 XGBoost 模型训练流程。
* 新增 val_hybrid.py: 用于混合模型的端到端精度验证与 FPS 评估。
* 对原版 utils/和 models/下的部分文件进行了必要修改，以支持特征图的输出。

开源许可证：
    本项目继承 YOLOv5 的开源协议（见根目录 LICENSE文件）。
