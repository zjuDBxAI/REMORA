> 本文保留原有构图、实验一至五与迁移说明。独立方法入口与最新运行步骤见[根目录 README](../README.md)。下列命令仍从仓库根目录执行。

# REMORA

Query-visible residual refinement for compressed graph search.

REMORA 为已有的向量量化器学习紧凑 residual codes，修正目标查询下的评分误差。它不替代 primary quantizer，而是复用 primary 编码和 RoarGraph，通过统一的训练原则提高压缩评分的排序准确性。

本仓库包含 **FP32 与五种 primary 的编码/建图源码、REMORA 训练与全库编码、实验一至五和报告生成代码**。数据集、模型、图索引、结果和编译环境需要另外准备。本文以当前正式的 **REMORA traversal** 协议为准；子目录中早期 terminal-only、低码率和 DP/grouping 说明属于历史版本。

## Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Dataset and workload](#dataset-and-workload)
- [Unified interface](#unified-interface)
- [Primary encoding and graph construction](#primary-encoding-and-graph-construction)
- [Training REMORA](#training-remora)
- [Evaluation](#evaluation)
- [Results and reproducibility](#results-and-reproducibility)
- [Repository layout](#repository-layout)
- [Limitations and troubleshooting](#limitations-and-troubleshooting)
- [Acknowledgements](#acknowledgements)

## Overview

REMORA 的离线训练与在线部署分为三个阶段：

1. **完整评分残差与 query-visible 表示。** 从实际 primary score 与 exact score 的差出发，验证原始空间中的线性重构关系；结合查询二阶矩和 residual 几何，学习 Balanced Product 的初始表示与码本。降维 primary 的投影误差也必须计入。
2. **候选条件编码适配。** 在固定 primary 图上，用 Primary traversal 收集训练和验证查询的候选池。中心化目标关注候选之间的相对评分误差，support-aware 正则化约束低支持度节点，随后更新节点的固定离线编码。checkpoint 只由验证查询选择。
3. **全库编码与在线评分。** 为所有节点生成紧凑 side codes，导出共享在线模型。查询时构造一次 residual LUT，与 primary score 相加，用于图搜索中的节点评分与队列排序。

统一的在线评分接口为：

$$
\widetilde s_i(q)=s_i^P(q)+h_\theta(q,a_i),
$$

其中 $a_i$ 是节点固定的 residual code，$h_\theta$ 是查表式 correction。**中心化是离线训练目标，不要求在线对候选池重新计算均值。查询向量保持浮点表示，不做数据库式量化。**

```text
Base embeddings + 独立的构图 / 训练 / 验证 / 测试查询
  → Primary 训练编码 → 同一 primary geometry 下的构图 Top-100
  → Primary RoarGraph
  → 完整评分残差 → Query-visible / Balanced Product 表示
  → Primary traversal 采集离线候选 → Centered + Support-aware 重编码
  → 全库 residual codes + 共享在线模型
  → REMORA traversal
      ├─ no-rerank：直接返回 Top-K
      └─ rerank：本分支队列 Top-L → 磁盘 FP32 有限精排 → Top-K
  → Performance / Ablation / Inversion / Traversal / Rate
```

Primary 与 REMORA 对照共享 **相同 primary model/codes、图、entry point 和 stopping policy**。离线用 Primary traversal 采集监督，不意味着正式 Ours 使用 Primary traversal；正式 Ours 从搜索开始就使用 REMORA score。

## Prerequisites

### 1. System and Python environment

主要依赖为 Linux、CMake ≥ 3.17、支持 C++20 的编译器、Boost program_options、OpenMP、glog、gflags、fmt 和 BLAS。CPU 快速内核与隔离 Faiss 配方以 **AVX-512** 机器为目标；换机器必须重新编译。

Ubuntu 的系统开发依赖可按需安装：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libboost-program-options-dev \
  libgoogle-glog-dev libgflags-dev libfmt-dev libopenblas-dev libeigen3-dev
```

建立独立 Python 环境，例如：

```bash
conda create -y -n remora --override-channels -c conda-forge \
  python=3.11 numpy scipy matplotlib pip
conda activate remora
cd /path/to/REMORA-code
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
```

先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与驱动兼容的 PyTorch，再安装其余依赖。不要不检查驱动就安装最新 CUDA wheel。

```bash
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -c 'import numpy, torch; print("numpy:", numpy.__version__); print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
lscpu
```

GPU 可用于初始 Faiss K-means、几何编码和 PyTorch 训练/重编码；正式 QPS 测试仍在 CPU 上执行。仅查看配置和生成运行计划不需要 GPU。

### 2. Build the native programs

以下命令从仓库根目录执行。`-j 8` 是编译并行度，与搜索线程数不是同一参数。

```bash
cmake -S ours -B ours/build-workflow-v1 -DCMAKE_BUILD_TYPE=Release \
  -DOURS_JI_KERNEL_ROOT="$PWD/ours/experimental/traversal-fast-v1" \
  -DRABITQ_ROOT="$PWD/thirdparty/RaBitQ-Library" \
  -DSAQ_ROOT="$PWD/thirdparty/SAQ"
cmake --build ours/build-workflow-v1 -j 8 --target \
  roar_primary_ji_search roar_exp234_audit roar_primary_ji_build roar_fp_build \
  roar_native_primary_reader roar_rabitq_quantize roar_export_saq_primary

export REMORA_NATIVE_PRIMARY_READER="$PWD/ours/build-workflow-v1/roar_native_primary_reader"
```

使用 conda 开发库时，可在 CMake 配置命令中追加 `-DCMAKE_PREFIX_PATH="$CONDA_PREFIX"`；系统编译器过旧时，需要另行配置支持 C++20 的 `CXX`。

统一 workflow 当前使用固定的 reader runtime 路径。首次部署到新目录时建立本仓库内的相对链接：

```bash
mkdir -p ours/runtime/native100-reader-v1
ln -s ../../build-workflow-v1/roar_native_primary_reader \
  ours/runtime/native100-reader-v1/roar_native_primary_reader
```

该步骤只在链接不存在时执行；已有链接先检查目标，不覆盖指向其他版本的文件。

### 3. Build GPU Faiss

仓库提供隔离的 Faiss v1.9.0 构建脚本，不附带预编译 Faiss。下面的 CUDA 路径与架构 **只是示例**，须按本机 toolkit 和 GPU 修改：

```bash
FAISS_CUDA_TOOLKIT=/usr/local/cuda-12.1 \
FAISS_CUDA_ARCH=89 \
FAISS_BUILD_JOBS=8 \
PYTHON_BIN="$PYTHON_BIN" \
  bash ours/test/t2i-10M/build_faiss_runtime.sh

export FAISS_PYTHON_PATH="$PWD/.runtime/faiss-v1.9.0-build/faiss/python/build/lib"
export PYTHONPATH="$FAISS_PYTHON_PATH${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -c 'import faiss; print("Faiss:", faiss.__version__); print("GPUs:", faiss.get_num_gpus())'
```

当前脚本构建 AVX-512 版本；不支持该指令集的 CPU 不能直接使用此配方。Python 主次版本变化后需重建 Faiss，不能复用另一 Python ABI 的扩展。

## Dataset and workload

### 1. Prepare embeddings and query splits

建图和训练使用 **embeddings**，不要求同时下载原始图片或视频。每个数据集至少需要：

| 输入 | 用途 | 要求 |
|---|---|---|
| Base vectors | Primary 编码、离线 residual、exact GT、可选 FP32 精排 | 所有文件使用相同节点 ID 顺序 |
| Build queries | Query–base Top-100 和 RoarGraph 构图 | 不使用测试查询 |
| Refiner training queries | 查询几何和训练候选池 | 与验证、测试查询分离 |
| Validation queries | 模型/编码 checkpoint 选择 | 不参与测试，也不泄漏到对应图训练 |
| Test queries | 正式检索与倒挂评估 | 不参与训练、配置选择或构图 |
| Exact ground truth | Recall 和 exact ranking pairs | 对整个原始数据库计算，而非某个 shard 或量化库 |

构图查询与 refiner 训练查询可按预先声明的协议共享；验证/测试隔离必须由 split provenance 记录，不能在看完测试结果后重新划分。

数据来源与当前适配配置：

| Dataset / workflow key | 当前 embedding 维度 | Primary 目标载荷 | 默认 residual | 来源或说明 |
|---|---:|---:|---:|---|
| T2I-10M / `t2i-10M` | 200 | 约 800 bits | 64 bits | [RoarGraph 数据说明](https://github.com/matchyc/RoarGraph#readme)；使用配套 embedding 与 split |
| T2I-1M / `t2i-1M` | 200 | 约 800 bits | 64 bits | T2I 的子集；记录抽样和 ID 映射 |
| WebVid-2.5M / `webvid` | 512 | 约 2048 bits | 160 bits | 当前为 2,505,000 节点的 512D 包；[原始 WebVid](https://github.com/m-bain/webvid)不等同于该处理后 embedding 包 |
| RS5M / `rs5m` | 512 | 约 2048 bits | 160 bits | [RS5M / GeoRSCLIP](https://github.com/om-ai-lab/RS5M)；以处理后文件 header 为准 |
| WordVector-ID / `wordvector-id` | 300 | 约 1200 bits | 96 bits | 当前是自划分 899K base，不应标为官方 WordVector-ID-1M |

预算原则是 $B_P\approx4D$、$B_R/B_P\approx8\%$。各方法使用最接近目标的合法 native configuration，并报告实际码长、padding 和辅助元数据；**不截断 OPQ/QAQ 的原始维度来凑预算**。数据源有自己的许可和下载条件，本仓库不打包或承诺提供这些处理后数据。

### 2. File formats and ground truth

以下格式均为 little-endian：

| 文件 | 布局 |
|---|---|
| `.fbin` | `uint32 rows, uint32 dim`，随后为 `rows × dim` 个 FP32 |
| 普通 ID `.ibin` | `uint32 rows, uint32 K`，随后为 `rows × K` 个 uint32 ID |
| `combined_ground_truth` | 相同 8-byte header，随后依次为完整 ID 矩阵和 FP32 score/distance 矩阵 |

combined GT 的大小为 `8 + 8 × rows × K` bytes；普通 ID 文件只有 `8 + 4 × rows × K` bytes。**不能把仅含 ID 的文件改名后当 combined GT 使用。**分数或距离的解释必须与 metric 相符；单位向量下的 L2/IP 换算也要显式校验。

完整五项实验建议准备 **全库 exact Top-200**。Top-100 可支持 Recall@10 和部分倒挂指标，但不能支持需要 Top-200 的审计。构图文件通常是训练 queries 的 **primary-scored Top-100 ID 文件**，不是测试 GT。是否归一化必须遵循数据集原始评分定义，不对 raw-IP 数据静默归一化。

### 3. Import an existing primary graph package

已有图时，必须同时获取构图服务器使用的 **primary model、primary codes、graph、base ID 顺序、配置与校验元数据**。只传图、重新训练一个“参数相同”的 primary，不能保证评分一致性。

已有配套 WebVid 包时：

```bash
"$PYTHON_BIN" ours/test/webvid/prepare_inputs.py \
  --package "$PWD/datasets/WebVid-2.5M-native4-M40-L500" \
  --tag remora-traversal-exp1-nvme-b160-v1 --side-bits 160
```

RS5M 与 WordVector 使用对应适配器：

```bash
"$PYTHON_BIN" ours/test/rs5m/prepare_inputs.py \
  --package "$PWD/datasets/RS5M-512-native4-M40-L500" \
  --tag remora-traversal-exp1-nvme-b160-v1 --side-bits 160 --gpu 0

"$PYTHON_BIN" ours/test/wordvector-id/prepare_inputs.py \
  --package "$PWD/datasets/WordVector-ID-native4-M40-L500" \
  --tag remora-traversal-exp1-nvme-b96-v1 --side-bits 96 --gpu 0
```

这些命令验证对应包的格式、传输元数据、节点顺序、评分和 split，生成 `inputs/imported-primary.json` 与训练/性能配置；**不建图、不重训 primary，也还没有训练 REMORA**。任意目录不能仅改成上述名字就通过导入。T2I-10M 使用已有 verified import report；历史文件 `imported-primary.jso` 是 JSON 文本，不是新的二进制格式。

## Unified interface

仓库根目录提供两个统一入口：`build_graph.py` 负责新 primary 图，`remora.py` 负责建图、REMORA 训练和实验调度。**SIFT/GIST 不纳入本项目的实验范围；当前支持原生 IP 数据，不隐式转换 L2。**

```bash
python build_graph.py --help
python remora.py list
```

### 新建 primary / FP32 图

先完成上面的 CMake 和 Faiss 安装。数据默认从 `--data-root` 中读取 `base.fbin`、`query.build.fbin`；QAQ/LeanVec 还读取 `query.train.fbin` 作为 primary 的训练查询。文件名不同时用 `--base`、`--build-queries`、`--fit-queries` 显式指定。已有模型/图不要用这个命令重新量化，继续走经过校验的导入路径。

```bash
python build_graph.py --dataset t2i-10M --method opq-pq \
  --base /path/to/base.10M.fbin --build-queries /path/to/query.build.fbin \
  --build-query-limit 100000 --degree 40 --build-beam 500 --knn-k 100 \
  --build-threads 64 --gpu 0 --output outputs/t2i-10M/graphs/opq-pq
```

默认仅打印计划。确认输入、量化参数和路径后，追加 `--execute` 启动：

```bash
python build_graph.py --dataset webvid --method saq \
  --data-root /path/to/webvid/data \
  --degree 40 --build-beam 500 --knn-k 100 --build-threads 64 --gpu 0 \
  --output outputs/webvid/graphs/saq --execute

python build_graph.py --dataset wordvector-id --method fp32 \
  --data-root /path/to/wordvector-id/data \
  --degree 40 --build-beam 500 --knn-k 100 --build-threads 64 --gpu 0 \
  --output outputs/wordvector-id/graphs/fp32 --execute
```

支持 `fp32/opq-pq/qaq/leanvec-ood/extended-rabitq/saq`，数据集标签为 `t2i-1M/t2i-10M/t2i-1B/webvid/rs5m/wordvector-id/laion-400M`。也可用 [dataset.example.json](../ours/workflow/configs/dataset.example.json) 的 `--dataset-config` 提供 IP、维度和文件路径；其中相对路径相对于仓库根目录，不相对于 JSON 所在目录。

一次执行包含：**primary 训练编码 → 原生 scorer/decode 一致性校验 → Faiss 全库分块 Top-K → native RoarGraph**。GPU KNN 不要求全库驻留显存；`--base-block` 和 `--query-batch` 控制块大小。默认使用指定的一张 GPU，不可用则报错，绝不自动回退 CPU；确实需要 CPU 时显式设置 `--knn-backend cpu`。此入口的 generic primary 训练仍使用现有 CPU Faiss 实现，不宣称它是全 GPU 训练器；REMORA 的 K-means/编码/优化 GPU 设置在训练配置内。

默认 primary 目标为 `4D` bits：OPQ/QAQ 保留原始维度，选择最近合法的 8-bit PQ 组数；200D 即 `M=100`、800 bits。RaBitQ 按 padding 后维度选择最近的原生位数，200D 即 `256×3=768` bits。SAQ 按 padded dimension 设置 total quota（200D→256D 时默认 `avg_bits=800/256=3.125`），使用原生 segmentation，最终码率读取导出的 quantization plan；不能把 padded 256D 的 `avg_bits=4` 误报成800 bit。LeanVec 使用本地 8-bit affine 复现，可用验证集选择 `--leanvec-dim`。可显式设置 `--primary-bits/--pq-groups/--rabitq-bits/--saq-avg-bits/--leanvec-dim`；不允许 OPQ/QAQ 为凑码率截维。下文旧低层配方的显式 `avg_bits=4` 是另一 native operating point，不是本入口默认预算。

结果目录含 `request.json`、primary model/codes、`query-base.knn.ibin`、KNN checkpoint、`graph.roar.index`、`construction.json`（native primary）、`summary.json` 和阶段日志。相同命令复用校验通过的编码、完整 KNN 和图；KNN 在数据库块之间提交两代 checkpoint，中断后重做未提交的块。**编码内部和构图内部没有新增断点；完整阶段可复用。**未完成的图拒绝覆盖，需要新输出目录。

输入或配置变化时拒绝复用，不用 `FORCE` 跳过身份验证。大型文件的恢复校验使用 stat 加分布式样本 SHA256，元数据明确标记其范围，不声称这是全文件 checksum。初始编码仍会生成全库重构等中间文件；大库需估算磁盘和构图 RAM。这个入口并没有变成磁盘图构建算法，也没有替代 LAION 专用多 GPU 流式流程。

### 训练 REMORA / 运行实验

已有经过导入校验的图，继续使用准备配置和 catalog；统一入口不会改变已有 artifact 或重新建图。新建图接入正式实验前仍需配套 graph import report、独立 train/validation/test 查询及测试 GT，不能把构图 KNN 当成测试 GT。

```bash
python remora.py train --dataset t2i-10M --config /path/to/prepare.config.json
python remora.py train --dataset t2i-10M --config /path/to/prepare.config.json --execute

python remora.py evaluate --dataset t2i-10M --experiments 1,3,4 --gpu 0
python remora.py evaluate --dataset t2i-10M --experiments 1,3,4 --gpu 0 --execute
python remora.py evaluate --dataset webvid --experiments 1,2,3,4,5 \
  --catalog /path/to/local-catalog.json --gpu 0 --execute
```

省略训练 `--config` 时使用 catalog 中的 prepare 配置。`--config` 入口显式拒绝 terminal-only 配置；正式训练/部署须声明 `scoring_placement=traversal`、`budget_allocator=balanced-product`。REMORA 训练的 `cuda:N`、Faiss K-means 和全库编码设备来自准备配置，不被评估的 `--gpu` 偷偷覆盖。评估 `--gpu` 用于实验二/五的训练，QPS 仍按配置使用 CPU 8 线程。请求多项实验时按依赖顺序 **一→三/四→二→五** 调度；没有请求的实验不自动添加。实验二/五依赖固定 teacher pairs，单独运行时须先准备它们。新服务器须更新 catalog/配置中的数据路径、GPU、绑核和 cold-store；数据下载、query 划分和 GT 计算不是此入口自动猜测的步骤。

## Primary encoding and graph construction

**已导入匹配图和 primary 的用户跳过本节，直接训练 REMORA。**本节用于从原始 embeddings 建新的 primary 图。示例的 `datasets/t2i-10M/data/*.fbin` 是建议路径，需替换为实际输入。

### 1. Train and encode the primary

| Method key | 当前编码入口 | 配置原则 |
|---|---|---|
| `opq-pq` | `ours/python/train_generic_primary.py` | 保留原始维度；合法 PQ groups 与 8-bit codeword |
| `qaq` | 同一入口，`--method qaq` | 尽量与 OPQ-PQ 使用相同 backbone capacity |
| `leanvec-ood` | 同一入口，`--method leanvec-ood` | 在验证集选降维配置；保留完整评分残差 |
| `extended-rabitq` | `roar_rabitq_quantize` | 原生 bits/padded dimension；校准开销另外计算 |
| `saq` | `roar_export_saq_primary` | 原生平均码率，由 SAQ 分配 |
| `fp32` | 不量化 | 原始向量评分 |

200D T2I 的 OPQ-PQ 与 QAQ 示例，显式覆盖旧脚本的 192D/768-bit 默认值：

```bash
"$PYTHON_BIN" ours/python/train_generic_primary.py \
  --method opq-pq --base-fbin datasets/t2i-10M/data/base.fbin \
  --query-fbin datasets/t2i-10M/data/query.train.fbin \
  --out-dir datasets/t2i-10M/primary/opq-pq \
  --opq-dim 200 --pq-groups 100 --primary-budget-bits 800 --threads 64

"$PYTHON_BIN" ours/python/train_generic_primary.py \
  --method qaq --base-fbin datasets/t2i-10M/data/base.fbin \
  --query-fbin datasets/t2i-10M/data/query.train.fbin \
  --out-dir datasets/t2i-10M/primary/qaq \
  --opq-dim 200 --pq-groups 100 --primary-budget-bits 800 --threads 64
```

LeanVec-OOD 也使用该入口。以下示例为 200D → 100D、8-bit reduced-coordinate 编码，载荷 800 bits；**它是合法配置示例，不代表已验证的最优降维维度**，实际配置由验证集选择：

```bash
"$PYTHON_BIN" ours/python/train_generic_primary.py \
  --method leanvec-ood --base-fbin datasets/t2i-10M/data/base.fbin \
  --query-fbin datasets/t2i-10M/data/query.train.fbin \
  --out-dir datasets/t2i-10M/primary/leanvec-ood \
  --leanvec-dim 100 --leanvec-bits-per-dim 8 --primary-budget-bits 800 --threads 64
```

当前 QAQ/LeanVec 是本地实现/复现，不能不加说明地标为论文官方实现。generic 导出会生成全库 reconstruction 等中间文件，不是 1B 级流式编码器，大库需先估算磁盘空间。

官方库适配的 Extended-RaBitQ 与 SAQ 示例：

```bash
mkdir -p datasets/t2i-10M/primary/extended-rabitq
OMP_NUM_THREADS=64 ours/build-workflow-v1/roar_rabitq_quantize \
  datasets/t2i-10M/data/base.fbin \
  datasets/t2i-10M/primary/extended-rabitq/base.roarq 3 ip false

ours/build-workflow-v1/roar_export_saq_primary \
  --base_fbin=datasets/t2i-10M/data/base.fbin \
  --query_fbin=datasets/t2i-10M/data/query.train.fbin \
  --out_dir=datasets/t2i-10M/primary/saq \
  --avg_bits=4 --threads=64 --query_limit=10000
```

RaBitQ 的 `3` 是 padded coordinate 的位数，不是整条向量码长；200D padding 到 256D 后为 `256 × 3 = 768` payload bits，4-bit 则是 1024 bits。`false` 选择该适配器的 exact quantizer 配置。SAQ 的码长以实际导出为准，不能由 `avg_bits` 推断全部物理存储。

### 2. Compute the build-query Top-100

对每个 primary，用构图训练 queries 在 **相同实际 primary scoring geometry** 下搜索全数据库。FP32 图使用 FP32 geometry。保存 query 范围、模型身份、ID 映射和 KNN backend；不能用测试 GT，也不能只扫描部分数据库。

GPU 放不下全库时，使用数据库分块、查询分批、局部与全局 Top-100 合并。统一 `build_graph.py` 已集成五种原生格式与 FP32 的分块 KNN；单独的底层入口为 `python ours/workflow/knn.py --help`。低层 `workflow/build.py` 仍要求提前准备 KNN；LAION 专用多 GPU 流程见下一小节。不要拿旧 primary+residual KNN 脚本生成 primary-only 图的训练邻居。

### 3. Build a primary graph or an FP32 graph

复制模板，填写路径和超参数：

```bash
mkdir -p configs
cp -n ours/workflow/configs/build-native.example.json configs/build-t2i10m-opq.json
cp -n ours/workflow/configs/build-fp32.example.json configs/build-t2i10m-fp32.json
```

Native primary 配置示例：

```json
{
  "kind": "native-primary",
  "primary_kind": "opq-pq",
  "primary_model": "datasets/t2i-10M/primary/opq-pq/primary_model.bin",
  "primary_codes": "datasets/t2i-10M/primary/opq-pq/primary_codes.bin",
  "queries": "datasets/t2i-10M/data/query.build.fbin",
  "knn": "datasets/t2i-10M/knn/opq-pq.query-base.top100.ibin",
  "knn_geometry": "primary",
  "graph": "datasets/t2i-10M/graphs/opq-pq/graph.roar.index",
  "summary": "datasets/t2i-10M/graphs/opq-pq/construction.json",
  "query_limit": 100000,
  "topk": 100,
  "degree": 40,
  "build_beam": 500,
  "threads": 64
}
```

| 配置字段 | 原生参数 / 含义 |
|---|---|
| `degree` | `M_pjbp`，投影图度数参数 |
| `build_beam` | `L_pjpq`，构图搜索宽度，不是测试 beam |
| `topk` | 构图 query–base KNN 的 K，通常 100 |
| `query_limit` | 构图查询数，必须与 KNN 行数对应 |
| `threads` | 构图 CPU 线程数，不是正式 QPS 线程数 |

先检查再构建：

```bash
python3 ours/workflow/build.py plan --config configs/build-t2i10m-opq.json
python3 ours/workflow/build.py check --config configs/build-t2i10m-opq.json
python3 ours/workflow/build.py run --config configs/build-t2i10m-opq.json --execute
```

FP32 使用 `kind=fp32`、`base`、`knn_geometry=fp32` 的配置执行相同入口。FP32 builder 使用整个 query 文件；限制查询数时先提供对应 query slice。`check` 校验 header/shape；native scorer parity 在 `run --execute` 时执行。已有 graph 或 summary 会拒绝覆盖。

```bash
python3 ours/workflow/build.py plan --config configs/build-t2i10m-fp32.json
python3 ours/workflow/build.py check --config configs/build-t2i10m-fp32.json
python3 ours/workflow/build.py run --config configs/build-t2i10m-fp32.json --execute
```

builder 接受 `GPRIM01/GPCOD01`、`RBQROAR1`、`SAQCODE1`；RaBitQ 的 model/codes 指向同一个 `.roarq` 文件。搜索 reader 能读某种导入格式，不代表 builder 也能读；外部 `OPQPQ002/ERBQRG/SAQRG/LVOOD` 等格式应复用匹配图，或先转换并验证评分一致性。

新建图完成后，需要形成 verified import report，再进入 REMORA 训练入口。已有数据集适配器负责对应契约；任意新格式/数据集还需相应适配，不能手填 `verified` 标志代替校验。

### 4. Large-scale LAION OPQ pipeline

较新的 LAION 构图实现单独放在 `build-source/`，不与正式实验内核混编：

```bash
cmake -S build-source/ours -B build-source/ours/build -DCMAKE_BUILD_TYPE=Release \
  -DRABITQ_ROOT="$PWD/thirdparty/RaBitQ-Library" \
  -DSAQ_ROOT="$PWD/thirdparty/SAQ"
cmake --build build-source/ours/build -j 8 \
  --target roar_primary_ji_build roar_native_primary_reader

"$PYTHON_BIN" build-source/ours/scripts/laion_opq_full.py \
  --source "$PWD/datasets/laion-400M" \
  --out "$PWD/outputs/build/laion-400M/opq-4b-v1" \
  --builder "$PWD/build-source/ours/build/roar_primary_ji_build" \
  --base-rows 0 --build-queries 10000000 --topk 100 \
  --pq-bits 8 --gpus 0,1,2 --threads 64 \
  --degree 40 --build-beam 500 --stage all
```

该入口要求配套 LAION shards/splits，`--base-rows 0` 使用可用全库。`--pq-bits 8` 是每个 2D PQ group 的码长，即平均 4 bits/dim；GPU 列表和线程数需适应本机。先用小规模新输出目录校验，再执行全库任务。

流程包括数据准备、OPQ 训练、流式编码、Faiss GPU 分块 Top-100 与构图；KNN 扫描有 checkpoint。构图 OPQ 节点对评分使用共享码字内积表，避免逐对重建 LUT。**GPU 分块 KNN 不意味着最终图采用磁盘构图**，构图 RAM 必须另行估算；某阶段有 checkpoint 也不意味着所有阶段均可逐步恢复。

## Training REMORA

### 1. Configure this machine and register the dataset

新机器上先复制 T2I native100 模板，或用前面的数据集导入器生成配置：

```bash
mkdir -p configs
cp -n ours/test/t2i-10M/config.native100-gpu.json configs/t2i10m-prepare.json
cp -n ours/test/t2i-10M/config.native100-exp1-nvme.json configs/t2i10m-performance.json
```

**执行任何 stage 前**编辑两个配置，覆盖模板及 `config.example.json` 中继承的原服务器路径：

| 字段 | 必须确认的设置 |
|---|---|
| `dataset`, `base_rows`, `base_dim` | 数据集身份与实际 fbin header |
| `base`, `train_queries`, `test_queries` | 本机输入路径 |
| `combined_ground_truth`, `graph_package` | 完整 GT 与 verified import report |
| `fp_graph` | 启用 FP32 reference 时需要的匹配图 |
| `primary_budget_bits`, `primary_dim`, `pq_groups` | T2I 主配置为 800、200、100，不落回旧默认值 |
| `side_bits`, `budget_allocator` | T2I 为 64、`balanced-product` |
| `device`, `kmeans_device`, `geometric_encoder_device` | 本机可用 GPU，例如 `cuda:0` |
| `geometric_encoder` | 当前 GPU Faiss 编码使用 `faiss` |
| `metric_query_*`, `select_query_*`, `build_query_*` | 配套 split，不把 T2I offset 生搬到 WebVid |
| `output_root` | 训练、性能分别使用新目录 |
| `build_dir` | 本机正式内核目录，例如 `ours/build-workflow-v1` |
| `scoring_placement` | **`traversal`** |
| `search_threads`, `cpus` | 正式测试为 8 线程、8 个本进程允许使用的 CPU ID |
| `rerank_storage`, `fp32_storage` | 正式磁盘测试为 **`direct`**；离线 mmap 不计为磁盘性能结果 |
| `memory_budget_bytes` | 当前搜索预算 256,000,000,000 bytes，不是整机独占保证 |

`cuda:N` 使用当前进程可见 GPU 的编号；设置 `CUDA_VISIBLE_DEVICES=2` 时通常应使用 `cuda:0`。模板中的 GPU 2、CPU 16–23 只是原服务器设置。

更新 [catalog.json](../ours/workflow/catalog.json) 对应 dataset 的 `prepare_config`、`performance_config`、`prepared`、`performance`、`cold_store` 等字段。cold store 放在实际 NVMe 路径；`/tmp` 不必然是 NVMe。

新跑实验三四时将 catalog 的 `teacher` 留空，使后续消融/码率读取本轮 `outputs/<dataset>/workflow-v1/exp34`。旧 `diagnostic_source`、`ablation_blocked` 等条目不能未经校验直接搬到新包。调整 methods 时，训练/性能配置与 catalog 的方法集合需要一致。

### 2. Preflight, smoke, then training

```bash
python3 ours/workflow/run.py list
python3 ours/workflow/run.py check --dataset t2i-10M
python3 ours/workflow/run.py plan --dataset t2i-10M --stages preflight,smoke,prepare
python3 ours/workflow/run.py run --dataset t2i-10M --stages preflight,smoke,prepare --execute
```

`plan` 只列命令与缺失输入；`check` 不是完整 scorer 校验。真正执行需要 `run ... --execute`。

`prepare` 复用匹配的 primary 图，不重训 primary、不重新构图。它准备 data view，收集训练/验证候选，训练 refiner 并编码全库。原始训练器是 [train_remora.py](../ours/test/t2i-10M/train_remora.py)，由 pipeline 提供配套 pool/model/code 参数。

当前以 Balanced Product 为基础，结合候选条件中心化、支持度正则化和节点重编码；每个节点只保存一条固定 code。码本数、表示维度和 padding 以训练报告为准，不把历史 8×25D 结构当作所有数据集的固定结构。

### 3. Check the generated artifacts

```text
<prepared>/
  configuration.json                   冻结配置与身份
  data/                                查询、GT 与离线 data view
  ji/<method>/b<B>/remora/
    model.npz                          离线模型
    online.bin                         共享在线模型
    side.bin                           全库紧凑 residual codes
    training.csv / training.json       训练历史与选择结果
    encoding-parity.json               编码一致性审计
  reused-graphs.csv                     复用图与 primary 记录
```

T2I-10M 的 64-bit side payload 为 `10,000,000 × 8 = 80 MB`（十进制）；文件 header 和共享模型另计。完整且身份匹配的模型/codes 可复用；配置或输入变化时使用新输出目录，不强行续跑旧冻结目录。

## Evaluation

推荐顺序为 **实验一 → 实验三四 → 实验二 → 实验五**：三四先产生共同 teacher pairs，供消融/码率审计复用。以下为已完成导入与 REMORA 编码的 T2I-10M；其他数据集替换 `--dataset` 并先更新 catalog。

### Experiment 1: Performance study

```bash
python3 ours/workflow/run.py plan --dataset t2i-10M --stages exp1
python3 ours/workflow/run.py run --dataset t2i-10M --stages exp1 --execute
```

正式协议：CPU 8 线程绑核、默认 10,000 测试查询、基础 3 次性能重复、自适应 beam/有限 rerank depth 采样、`L ≤ 200`。噪声点可能追加测量；性能重复次数不等于准确性诊断的重复次数。

| 分支 | Traversal scorer | 最终输出 / 精排选择 |
|---|---|---|
| Primary / no-rerank | Primary | 本分支队列 Top-K |
| REMORA / no-rerank | Primary + residual correction | 本分支队列 Top-K |
| Primary / rerank | Primary | Primary 队列 Top-L → 磁盘 FP32 |
| REMORA / rerank | Primary + residual correction | REMORA 队列 Top-L → 磁盘 FP32 |

rerank 均从 **自己的 traversal 队列**选 Top-L，不先用 REMORA 重排 Primary 候选再称为 baseline。输出 QPS–Recall、Mean/P50/P95/P99 service latency–Recall、Equal-Recall Efficiency 与 memory breakdown。

FP32 reference 默认可关闭；启用时设置 `include_fp32_reference=true`、提供正确 `fp_graph` 并明确 raw-vector 存储方式。**FP32 scorer 在量化图上运行不等于独立 FP32-built graph baseline。**多线程 QPS 不能用 `1000 / mean latency_ms` 简单替代。

正式 timing 不与训练、下载或其他性能扫描并行。绑核只约束本进程，不代表独占 CPU、内存带宽或 NVMe；共享负载必须在记录中说明。

### Experiments 3 and 4: Inversion and traversal analysis

```bash
python3 ours/workflow/run.py plan --dataset t2i-10M --stages exp3,exp4
python3 ours/workflow/run.py run --dataset t2i-10M --stages exp3,exp4 --execute
```

此入口复用模型和图，进行一次准确性诊断，**不新增三遍 QPS 测试**。指标包括：

- **AdjacentInv**：exact Top-R 的相邻排名 pair，默认 R=50/75/100/200，受 GT 深度约束。
- **BoundaryInv@10**：exact Top-10 边界两侧 h=5/10 窗口的跨边界 pair。`h=5` 不是 Recall@5，应同时写 K 与 h。
- **ExpandInv / QueueInv / StopInv**：所有方法对同一组 FP32 teacher-forced routing pairs 重新评分，按 query-balanced 方式统计。
- **Traversal**：free-running 的 visited/expanded、visited coverage、retained coverage 和输出 Recall；90%/95%/99% 目标下选择实测点。

GT 平局/无 pair 的处理以审计规则为准，并记录有效分母。没有真实 score-based stop 事件时，StopInv 为 N/A；停止边界 proxy 不冒充 StopInv。不同方法自采样 pair 不能证明同一 routing decision 更准确。

Equal-Recall 表复用实验一冻结测量；固定 L 的 traversal 表与可变 L 的性能 frontier 分开。未达到的目标标 N/A，不外推。

### Experiment 2: Ablation study

```bash
python3 ours/workflow/run.py plan --dataset t2i-10M --stages exp2
python3 ours/workflow/run.py run --dataset t2i-10M --stages exp2 --execute
```

固定 primary、图、refiner bits、beam、candidate policy 和 rerank policy，比较 Primary、Euclidean Residual、Alignment、Balanced Product、候选条件 MSE/固定正则、Centered/固定正则和 Support-aware Full。默认 beam=100/200、L=0/100；timing 与训练串行，倒挂审计复用固定 teacher pairs。

统一 planner 现已支持 `--gpu`，默认 0，不再固定 GPU 2。它只控制实验二/五的训练，不改变正式 CPU timing 设置。也可直接调用真实脚本，例如 T2I-10M 消融：

```bash
"$PYTHON_BIN" ours/scripts/run_frozen_ablation.py \
  --repo "$PWD" --dataset t2i-10M --side-bits 64 \
  --prepared outputs/t2i-10M/remora-native100-b64-v1 \
  --performance outputs/t2i-10M/remora-native100-traversal-exp1-nvme-b64-v2 \
  --teacher outputs/t2i-10M/workflow-v1/exp34 \
  --output outputs/t2i-10M/workflow-v1/exp2 \
  --beams 100,200 --depths 0,100 --repeats 3 --queries 10000 --gpu 0
```

路径须与本轮 catalog 一致。不要同时向同一目录启动直接脚本和 wrapper。

### Experiment 5: Bit budget and rate–distortion

```bash
python3 ours/workflow/run.py plan --dataset t2i-10M --stages exp5
python3 ours/workflow/run.py run --dataset t2i-10M --stages exp5 --execute
```

| Dataset | Side budgets |
|---|---|
| T2I-1M / T2I-10M | 32、64、96、128 bits |
| WebVid / RS5M | 80、160、240、320 bits |
| WordVector-ID | 48、96、144、192 bits；当前先补 Top-200 |

预算是默认 residual 的 0.5/1/1.5/2 倍。每个预算重新训练/编码，报告 DW、Score RMSE、RoutingInv、BoundaryInv@10、no-FP-rerank Recall 和存储；每个 accuracy point 一次运行，**不将 timing/QPS 当作性能结果**。大于 256-bit 的训练使用配套 `relative-rate-v1`。

本实验采用 **Balanced Product**，不是早期 `D_G → DP → grouping`；旧 allocator 应另设消融。GPU 用 `--gpu` 选择，真实 accuracy wrapper 为 [run_accuracy_rate.py](../ours/workflow/adapters/run_accuracy_rate.py)。

## Results and reproducibility

定位报告与运行入口测试：

```bash
python3 ours/workflow/run.py reports --dataset t2i-10M
python3 -m unittest discover -s ours/workflow/tests -v
```

`<performance>` 是 catalog 注册的性能输出；`<workflow>` 默认 `outputs/<dataset>/workflow-v1`。

| 实验 | 主要输出 |
|---|---|
| Performance | `<performance>/experiment1/aggregate.csv`、`summary.json`、`figures/*-recall-*.svg` |
| Ablation | `<workflow>/exp2/report/ablation-table.csv`、`inversion-table.csv`、`summary.json` |
| Inversion | `<workflow>/exp34/report/inversion-table.csv` 和 `experiment3/` 原始 score/pairs |
| Traversal | `<workflow>/exp34/report/traversal-table-no-rerank.csv`、`traversal-table-l100.csv`、coverage 分解 |
| Bit budget | `<workflow>/exp5/experiment5/report/requested-tables.csv`、分 primary/beam 表与曲线 |
| 运行记录 | `<workflow>/logs/`、各 stage 配置、身份、状态与退出码 |

曲线从完成重复测量的 **实测非支配点**绘制，Primary 虚线、REMORA 实线。平滑只能作为展示插值，不生成新的测量点；matched-Recall 表仍由原始测量计算。`reports` 只定位报告，不重新测性能。

存储统计区分 **图、primary codes、residual codes、共享 codebooks/transform、runtime buffer、FP32 hot/cold vectors**。Index Size 不等于 Peak RSS；组件 payload 估计也不等于包含 allocator 的 RSS。无 rerank 时在线压缩分支无需读取原始向量；离线训练和 GT 仍需要原始数据。

续跑前核对配置、输入 hash、二进制和 scoring contract。同身份的完成点可复用；迁移服务器或改变源码、primary、policy 时新建 output root，不将旧结果混入新曲线。`SOURCE_MANIFEST.csv` / `SOURCE_SUMMARY.json` 记录本源码包来源和 SHA-256，但不替代数据评分/传输校验。

## Repository layout

```text
REMORA-code/
  README.md / requirements.txt / SOURCE_MANIFEST.csv
  include/ src/ tests/              RoarGraph 构图库
  thirdparty/                       RaBitQ、SAQ、Eigen 等源码及许可证
  ours/
    src/ include/                   Primary、磁盘精排、SIMD、residual 在线评分
    python/                         编码、IO、模型导出与正式流程依赖
    experimental/traversal-fast-v1/ 正式搜索快速内核
    experimental/relative-rate-v1/  大预算训练入口
    test/t2i-10M/                   跨数据集正式协议与 REMORA 训练器
    test/webvid/ rs5m/ wordvector-id/数据集适配
    test/t2i-1M*/ t2i-1B/            正式流程仍引用的历史依赖
    scripts/                        冻结实验与报告调度
    workflow/                       统一入口、分块 KNN、plan/check/run 与构图模板
  build-source/                     较新的 LAION 构图实现
    thirdparty -> ../thirdparty
  datasets/                         用户准备的数据与匹配 primary 图
  outputs/                          运行后生成的模型、配置和报告
```

最后两类不在纯源码包中。详细索引见 [FILE_MAP.md](../ours/workflow/FILE_MAP.md)，统一入口说明见 [workflow README](../ours/workflow/README.md)。`traversal-fast-v1` 和被正式流程引用的旧命名 helper 不是可随意删除的废弃文件。

## Limitations and troubleshooting

- **首次迁移出现 `/home/xzl/...` 缺失。** 模板/catalog 有旧绝对路径；先替换数据、graph report、输出、reader、GPU、CPU affinity 与 cold-store，再执行。代码包不包含它们引用的数据/结果。
- **相同后缀却无法读取 primary。** 检查 magic/header 与身份，不仅按 `.bin` 或维度判断兼容性；先 smoke，再完整训练。
- **WordVector / RS5M 消融缺 GT。** WordVector 当前仅 Top-100，实验二/五需要 Top-200；旧 RS5M prepared view 也需匹配 Top-200，不能拼造或借用别的 split。
- **GPU / Faiss ABI 错误。** 核对驱动、toolkit、PyTorch wheel、Python ABI；K-means、编码 backend、训练 device 分别配置。
- **磁盘 rerank 异常慢。** 检查实际设备、O_DIRECT 支持、I/O 竞争和读深度；page-cache mmap 或内存 FP32 不能标成 NVMe 冷存储性能。
- **实验范围与超大图。** SIFT/GIST 明确不做；当前正式原始空间协议为 IP。构图 RAM 单独估算，uint32 ID builder 不能直接扩到 5B。
- **统一接口的边界。** 新图用 `build_graph.py`，已有图的 REMORA 训练/实验用 `remora.py`；首次导入、query split 和测试 GT 仍须明确准备。并非任意旧格式、任意指标或未知数据能无校验地一键运行五项实验。

## Acknowledgements

图索引基于 [RoarGraph](https://github.com/matchyc/RoarGraph)。项目保留第三方源码及各自许可证，包括 Faiss、RaBitQ、SAQ 等；本地复现与官方实现的区别须在报告中说明。安装、数据准备、构建和评估章节组织参考 [SQUID README](https://github.com/zjuDBxAI/SQUID#readme)。
