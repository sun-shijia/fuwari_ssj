---
title: DTA/I
published: 2025-03-19
description: 药物靶点相互作用，活性，成药性
tags: [DTA]
category: Summarize
draft: false
---

# DTA 论文整理

## [DeepDTA, 2018](https://doi.org/10.1093/bioinformatics/bty593)
:::note
**Dataset**  
Davis, KIBA

**Model**  
使用卷积神经网络（CNNs）编码小分子 SMILES 和蛋白 Sequence，然后 Maxpooling，FC

**Metrics**  
Concordance Index ($$CI$$)：衡量两个随机药物-靶点对的预测结合亲和力值是否按照与真实值相同的顺序进行预测  
Mean Squared Error ($$MSE$$): 衡量预测值与真实值之间的均方根误差  
Area Under Precision Recall ($$AUPR$$): 适用于二分类任务(将 Davis, KIBA 的 label 值分别确定一个阈值进而将数据集修改成二元数据集)  
$$r_{m}^{2}$$：评估QSAR模型的外部预测性能，>0.5 为可接受

**Baseline**  
[KronRLS](https://doi.org/10.1093/bib/bbu010)，  [SimBoost](https://doi.org/10.1186/s13321-017-0209-z)

**innovate**  
将二元分类任务转化为回归任务  
采用 CNN 架构编码 1D 序列信息
:::

## [GraphDTA, 2020](https://doi.org/10.1093/bioinformatics/btaa921)
:::note
**Dataset**  
Davis, KIBA

**Model**  
药物分子转化为图结构，每个节点是一个包含五个特征的二元特征向量，然后采用 GCN, GAT, GIN, GAT-GCN 四种网络输入  
蛋白分子采用处理 Sequence 的方式，先 padding 再 embedding 输入再卷积

**Metrics**  
$$CI$$  
$$MSE$$

**Baseline**  
[DeepDTA](https://doi.org/10.1093/bioinformatics/bty593)，  [WideDTA](https://doi.org/10.48550/arXiv.1902.04166)

**innovate**  
应用 GNN 用于回归任务
对 GNN 输出的 128 latent variable 进行了可解释性分析，发现脂肪族OH基团的数量影响最大，有两个潜在变量与其较相关, 但其他的还比较难解释
:::

## [FusionDTA, 2021](https://doi.org/10.1093/bib/bbab506)
:::note
**Dataset**  
Davis, KIBA

**Model**  
药物分子通过一个SMILES格式的词汇表，编码成 one-hot 再投影到一个低维的离散空间（维度由 embedding 层决定）  
蛋白表示采用了预训练 transformer (ESM-1b) 的 encoder，输入 embedding 是 token embedding 和 position embedding 的和，以 `[CLS]` 开始，`[SEP]` 结束  
采用预训练 + 微调结合模式，预训练任务是完形填空，预测蛋白序列的 masked token，微调任务是用预训练好的 encoder 编码然后进行下游 DTA 预测任务，编码结束后分别将药物分子和蛋白分子输入一个 Feedforward & Activation 然后再进入两层的 BiLSTM，目的是为了捕获 embedding 的长程依赖和短程依赖，>然后通过一个多头线性注意力机制，concact 然后作回归输出  
对 DTA 任务做知识蒸馏

**Metrics**  
$$CI$$  
$$MSE$$  
$$r_{m}^{2}$$

**Baseline**  
[KronRLS](https://doi.org/10.1093/bib/bbu010)，  [SimBoost](https://doi.org/10.1186/s13321-017-0209-z)，  [DeepDTA](https://doi.org/10.1093/bioinformatics/bty593)，  [WideDTA](https://doi.org/10.48550/arXiv.1902.04166)，  [MT-DTI](https://doi.org/10.48550/arXiv.1908.06760)，  [DeepCDA](https://doi.org/10.1093/bioinformatics/btaa544)，  [MATT_DTI](https://doi.org/10.1093/bib/bbab117)，  [GraphDTA](https://doi.org/10.1093/bioinformatics/btaa921)，  [GEFA](https://doi.org/10.1109/TCBB.2021.3094217)

**innovate**  
采用了预训练 + 微调形式  
采用 token 表示蛋白  
采用知识蒸馏方法
:::

## [MGraphDTA, 2022](https://doi.org/10.1039/D1SC05180F)
:::note
**Dataset**  
Davis，KIBA，Metz，Human，C.elegans，ToxCast

**Model**  
对于药物分子，构建了一个非常深的 MGNN 网络（包括 Multiscale block 和 Transition layer）提取特征，可以精确的提取子结构特征，将分子官能团等的信息提取到，不像之前的 GNN 可能深度较浅甚至包括不了完整的官能团，大环等，并且为了解决加深 GNN 网络带来的特征过平滑 (over-smoothing) 问题，引入了稠密连接 (dense connections), 整合给定顶点或原子的不同感受野的特征，保持分子不同尺度的子结构。  
对于蛋白分子，采用的 MCNN 是多尺度的卷积网络（单层双层三层 CNN 分别提取后拼接）提取特征，最后将药物分子和蛋白分子特征融合后过 MLP 输出  
还提出了一种简单但有效的可视化方法，称为Grad-AAM，用于研究GNN如何在DTA预测中做出决策。使用了 MGNN 最后一个图卷积层的梯度信息来了解每个神经元对亲和力决策的重要性

**Metrics**  
$$CI$$  
$$MSE$$  
$$r_{m}^{2}$$  
$$Precision	  Recall   AUC$$ (分类任务)  
$$Spearman$$

**Baseline**   
[DeepDTA](https://doi.org/10.1093/bioinformatics/bty593)，  [Wide-DTA](https://doi.org/10.48550/arXiv.1902.04166)，  [GraphDTA](https://doi.org/10.1093/bioinformatics/btaa921)，  [DeepAffinity](https://doi.org/10.1021/acs.jcim.0c00866)，  [TrimNet](https://doi.org/10.1093/bib/bbaa266)，  [TransformerCPI](https://doi.org/10.1093/bioinformatics/btaa524)，  [VQA-seq](https://doi.org/10.1038/s42256-020-0152-y)，  [GNN-CNN](https://doi.org/10.1093/bioinformatics/bty535)

**innovate**  
加深了 GNN 的网络并且从化学层面做出了合理化（引入 dense connections）  
提出了非注意力机制的可解释性方法称为 Grad-AAM，能够较为简便的应用到各种 GNN 模型
:::

## [GAL-DTA, 2024](https://doi.org/10.1016/j.ins.2024.121135)
:::note
**Dataset**  
3B0W，5DXU，1PKG，1RJB 四个蛋白，每个蛋白通过 generator 形成 4000 个 pair 用于训练，1000 + 2000 用于测试

**Model**  
首先通过 ChEMBL fine-tune 一个 Generator (DrugEx 库), 然后生成 4000 个分子，AutoDock-GPU 用来 label (Affinity), 得到 dataset  
然后用产生的 dataset 进行预测模型的训练，小分子用图表示，过 GAT 层，序列用 embedding 表示，过 BiLSTM 层，然后通过多头注意力和线性层得到最终预测结果

**Metrics**  
$$CI$$  
$$MSE$$  
Spearman's rank correlation($$Sp$$)  
$$Pearson$$

**Baseline**  
[DeepDTA](https://doi.org/10.1093/bioinformatics/bty593)，  [GraphDTA](https://doi.org/10.1093/bioinformatics/btaa921)，  [MGraphDTA](https://doi.org/10.1039/d1sc05180f)，  [FusionDTA](https://doi.org/10.1093/bib/bbab506)，  [CORESET](https://doi.org/10.48550/arXiv.1708.00489)，  [ACS-FW](https://proceedings.neurips.cc/paper_files/paper/2019/file/84c2d4860a0fc27bcf854c444fb8b400-Paper.pdf)，  [BAIT](https://proceedings.neurips.cc/paper_files/paper/2021/file/4afe044911ed2c247005912512ace23b-Paper.pdf)   

**innovate**  
预测模型改进不是很多，主要是用了预训练的生成数据和标注的方式得到了主动学习的数据集
:::

## [DMFF-DTA, 2025](https://doi.org/10.1038/s41746-025-01464-x)
:::note
**Dataset**  
Davis, KIBA

**Model**  
第一部分：序列特征提取模块，药物分子采用原子类型做 token，蛋白序列采用氨基酸种类做 token，然后将 token embedding 进入一个全连接层，再过一个 GEM, 然后进入 BiLSTM 分别提取大小分子序列信息，concact，过一个多头链接注意力模块，然后 FFN 得到序列特征输出  
第二部分：图结构特征提取模块，药物分子直接采用 Rdkit 得到图结构，蛋白使用结合 Uniprot 的结合位点信息的 contact map，然后根据结合位点的 index 相当于从 contact map 中取了一个子集（口袋）作为蛋白图，然后创建一个虚拟节点把两个图连起来，共同输入到之后的多层 GNN 得到图特征输出  
将两部分的特征输出融合过 FFN 得到 DTA 预测值

**Metrics**  
$$CI$$  
$$MSE$$  
$$r_{m}^{2}$$

**Baseline**   
[DeepDTA](https://doi.org/10.1093/bioinformatics/bty593)，  [GraphDTA](https://doi.org/10.1093/bioinformatics/btaa921)，  [MGraphDTA](https://doi.org/10.1039/d1sc05180f)，  [FusionDTA](https://doi.org/10.1093/bib/bbab506)，  [MSGNN-DTA](https://doi.org/10.3390/ijms24098326)

**innovate**  
采用了双向模型（序列和图同时提取信息，最后合并预测）  
结合了 AlphaFold 的 contact map 加到图结构的信息里
:::

