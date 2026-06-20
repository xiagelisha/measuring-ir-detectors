# shared/ — 共享基础设施

本目录为全部基线与创新算法的**公共基础设施**, 保证所有实验在同一套数据加载、增强、指标与可视化口径下进行, 结果直接可比。

当前 (Step 0.1) 仅创建占位骨架, 后续 **Step 0.2** 将在此填入:

- `dataset.py` — 红外检测数据集加载 (统一 train/val/test split)
- `transforms.py` — 训练/推理增强 (Albumentations 管线)
- `metrics.py` — mAP@0.5 / mAP@0.5:0.95 / P / R / F1 (基于 pycocotools)
- `visualizer.py` — 预测可视化、对比拼图、论文级配色
- `configs/` — 全局配置 (已含 `global.yaml`) + 各算法可继承的基础 YAML
