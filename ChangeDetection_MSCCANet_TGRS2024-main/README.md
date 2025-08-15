# 下面所有内容是YMH在论文作者原github项目的README.md文件基础上进行修改补充的
## 训练网络：python train.py
此程序将该程序内每一轮训练、验证、测试的模型超参数和指标保存在 tmp/train/、tmp/val/、tmp/test/ 目录下；
将每一轮训练使用的模型保存在 tmp/train/ 目录下；
将历史训练指标保存在 tmp/train/train_history.json；
将F1Score最佳的验证和测试模型保存在 tmp/ 目录下如best_val_checkpoint_epoch_num.pt、best_test_checkpoint_epoch_num.pt，并将该模型的超参数和指标保存在 tmp/ 目录下。

## 验证网络：python eval.py
此程序将验证记录按日期时间保存在 tmp/eval_record/ 目录下，其中 Best_Resout.json 记录了哪一轮的模型有最佳的Precision、Recall、F1Score，checkpoint_epoch_num.json 记录了num轮模型的超参数和指标。

## 使用训练好的网络模型进行预测且可视化：python visualization.py
该程序允许用户指定加载模型路径（path = './tmp/best_test_checkpoint_epoch_54.pt'）
可视化结果保存在 output_img/ 目录下如 00000.png

# Citation
"A Multiscale Cascaded Cross-Attention Hierarchical Network for Change Detection on Bitemporal Remote Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-16, 2024
(https://github.com/cslxju/ChangeDetection_MSCCANet_TGRS2024)

# Requirements

- Python 3.6

- Pytorch 1.4

- torchvision 0.5.0

# Dataset

- [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit) (Change Detection Dataset)
- [LEVIR](https://justchenhao.github.io/LEVIR/), 
- [BCDD](https://study.rsgis.whu.edu.cn/pages/download/)
- Crop LEVIR and BCDD datasets into 256x256 patches. The pre-processed BCDD dataset can be obtained from [BCDD_256x256](https://drive.google.com/file/d/1VrdQ-rxoGVM_8ecA-ObO0u-O8rSTpSHA/view?usp=sharing).
我在复现时使用的数据集是BCCD_256x256，数据集目录名称一定要严格按照下面的格式命名。
- Prepare datasets into the following structure and set their path in `metadata.json`
    ```
    ├─train
        ├─A        ...jpg/png
        ├─B        ...jpg/png
        ├─OUT    ...jpg/png
        └─list     ...txt
    ├─val
        ├─A
        ├─B
        ├─OUT
        └─list
    ├─test
        ├─A
        ├─B
        ├─OUT
        └─list
    ```

# Train from scratch

    python train_val.py
	
# Evaluate model performance

    python eval.py

# Visualization

    python visualization.py