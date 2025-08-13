此文件是YMH编写。
pretrained_models目录及其内容、model.py、utils.py为原github项目包含内容。
github上找到的论文源代码并不完全，model.py中缺少重要的VSS模块。
YMH添加的文件包括：
    SpectMamba_block.py
    data目录
SpectMamba_block.py完整实现了Conv-VSS双分支结构，重要点包括：
    SpectralGatingNetwork频域门控网络
    SS2D改进VSS
    PatchEmbed2D图像到Patch Embedding的转换