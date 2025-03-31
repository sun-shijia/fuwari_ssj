---
title: 如何区分 Conda 的 CUDA 和系统的 CUDA？
published: 2025-03-25
description: 服务器已配置系统 CUDA 的情况下如何在新建的虚拟环境中使用对应版本的 CUDA
tags: [others]
category: Summarize
draft: false
---

# 服务器情况

在 `user/local` 下已经有了 `cuda-11.1` 和 `cuda-11.4`，若直接 `nvcc --version` 则显示：

```python
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Aug_15_21:14:11_PDT_2021
Cuda compilation tools, release 11.4, V11.4.120
Build cuda_11.4.r11.4/compiler.30300941_0
```

说明 `nvcc` 来自 `/usr/local/cuda-11.4/`，而不是 Conda 虚拟环境。

# 如何让 Conda 只用自己的 CUDA，而不依赖系统？(可能存在的情况)

以新建虚拟环境 `MolTran_CUDA11` 为例，假如想要装 `CUDA 11.0` 对应 `pytorch 1.7.1`：

```bash
conda create --name MolTran_CUDA11 python=3.8.10
conda activate MolTran_CUDA11
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

此时使用：

```bash
python -c "import torch; print(torch.version.cuda)"
```

若显示 `11.0` 则没有问题，若报错或显示 `11.4` 则执行：

```bash
export LD_LIBRARY_PATH=/home/cuda/anaconda3/envs/MolTran_CUDA11/lib:$LD_LIBRARY_PATH
python -c "import torch; print(torch.version.cuda)"
```

若显示 `11.0` 则没有问题



