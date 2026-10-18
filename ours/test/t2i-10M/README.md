# T2I-10M：复用现成图的最终 REMORA 实验

## 2026-10-05：正式预算已更新，当前入口仍对应旧图

已确定T2I-10M/T2I-1B的主目标为 **200D、800-bit primary + 64-bit refiner**，
名义总载荷864 bit / 108 B。其他维度采用primary约4D、refiner约8%的预算，
各方法使用邻近原生配置并报告确切载荷、辅助数据与真实存储。
正式预算协议保存在本机 `/home/xzl/RoarGraph/ours/实验方案.md` 第3.3节。

**下面的现有运行命令仍复用79/80 B的迁移图，不是新800-bit预算实验。**
`qvrr.select_runs`仍按`code_bytes_per_vector <= 80`选择，配置里的
`primary_budget_bits=640`不代表所有图恰好有640-bit载荷。
新预算需要相应primary模型/codes、图与独立输出目录，并适配图选择规则；
不能只把配置或报告字段改成800来复用旧结果。现有命令保留用于旧预算对照。

本入口不训练 primary、不计算 query-base KNN、不调用建图程序，也不修改
`datasets/QVRR-quantized-graphs` 中任何文件。默认使用五种 primary 的
79/80 B/node 匹配预算图，排除 SAQ 的旧 264 B/node 图。

## 训练与在线流程

1. 严格读取 QVRR 的 OPQ-PQ/QAQ、Extended-RaBitQ、SAQ、LeanVec-OOD 文件。
   这些是迁移包的实现，不应冒称与官方库/论文全部配置等价。
2. 从实际 primary scorer 的线性映射构造原始空间完整评分残差：
   `r = x - A * xhat_s`。LeanVec 的降维误差也包含其中；不使用伪逆。
3. 用独立训练查询二阶矩和残差协方差构造 query-visible/JI 表示，
   Balanced Product 初始化；每组 8 bit，每个节点只有一条共享 code。
4. Primary-only traversal 收集固定候选池，监督标签为实际
   `exact score - native primary score`。候选池中心化去除共同偏移。
   正则权重为 `pool_mass * D_eff / n_eff`；重复 beam 的同一查询先合并。
5. 使用共享码本/正交旋转的真实目标下降检查和逐组离散重编码，
   独立 validation 候选池 Recall 选择 checkpoint。未观察节点使用全局几何编码。
6. 默认 Primary traversal → REMORA terminal → 可选 top-L FP32 磁盘精排。
   查询 primary 准备只做一次，terminal 复用候选 primary 分数并增加 residual LUT。
   原始 FP32 向量用于离线训练；无 FP32 精排的量化查询无需读取它们。

## 五项实验编号

1. Performance：8线程，QPS、mean/P50/P95/P99 **service** latency，有限磁盘精排；
   FP32 traversal 也使用同一种磁盘策略。注意多线程的 `1/QPS` 不是单查询延迟。
2. Ablation：Euclidean / query alignment / Balanced / centered / REMORA，
   固定64bit、图、beam、候选数、精排策略。
3. Inversion：R=50/100/200，Boundary h=5/10，同一 FP teacher 事件；
   查询平衡，排除 exact ties，approximate ties按 sign mismatch。
   原搜索没有基于分数的停止决策，所以真实 StopInv 是 N/A；
   StopBoundaryProxyInv 另列，不能替代真实 StopInv。
4. Traversal：只使用真实达到90/95/99%的点；未达到写明，不外推。
   terminal 模式不会改变同 beam 下 Primary 的搜索轨迹；改善来自最终排序。
5. Bit budget：16/32/48/64/96/128bit分别重新训练。
   **这是最终 Balanced Product 的码率实验，不是旧 D_G(b)→DP→grouping。**
   旧 DP 对照仍在 `ours/test/t2i-1B`，不可把两种分配器混为一谈。

## 数据与配置

默认原始向量、查询、GT、FP图指向本机 `/home/xzl/RoarGraph/data/t2i-10M`，
图包相对于当前 REMORA 仓库。换机器时复制一份 config，修改路径。
不使用已迁移 CMakeCache，独立构建到 `ours/build-t2i10m-remora-v1`。
BigANN GT 的 IDs+scores 格式会转换为标准矩阵，并与实际原始分数抽查。
不重新归一化200D数据。训练不读取 test queries/GT。

预检查、分数一致性检查（不启动完整训练）：

```bash
cd /home/xzl/REMORA
PYTHON_BIN="$CONDA_PREFIX/bin/python" COMMAND=preflight bash ours/test/t2i-10M/run_all.sh
PYTHON_BIN="$CONDA_PREFIX/bin/python" COMMAND=smoke bash ours/test/t2i-10M/run_all.sh
```

完整准备+实验（默认8线程，三次重复）：

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" bash ours/test/t2i-10M/run_all.sh
```

只做实验一、三、四：

```bash
EXPERIMENTS=1,3,4 PYTHON_BIN="$CONDA_PREFIX/bin/python" bash ours/test/t2i-10M/run_all.sh
```

默认训练 device=cpu；需要GPU时复制config并设置 `device`，例如 `cuda:2`。
本轮只安装代码并完成小规模验证，没有替您启动完整10M训练/性能扫描。
六档 residual payload 分别为2/4/6/8/12/16 B/node；side文件使用紧凑v3格式。
新读取器仍兼容旧v1/v2的64-bit word对齐文件。
所有阶段按源代码/配置/输入指纹续跑；源代码或配置变化请用新 output_root。
正式计时不要和训练/其他IO任务并发。SVG只连真实Pareto点，不生成虚构数据；
阈值以上无点会显示未达到，不会把低召回结果强行平滑到高召回。
论文结果需另报告训练分布、query划分、实现差异和存储预算。

## Faiss GPU 初始化、全库编码与 AVX-512（2026-10-04）

新增配置：`kmeans_device`、`geometric_encoder`、`geometric_encoder_device`、
`encoding_audit_size`。两项 device 默认 `auto`，跟随训练 `device`；
几何编码默认 `faiss`，可设为 `torch` 做冻结模型对照。

- 初始 K-means 使用 `faiss.Clustering` 和指定设备的 FP32 FlatL2。
  `cuda:2` 是进程可见 GPU 2，不是使用两张 GPU；不使用含义不同的
  `faiss.Kmeans(gpu=2)`。若设置 `CUDA_VISIBLE_DEVICES=2`，应改为 `cuda:0`。
- 全库几何编码缓存每组 Faiss FlatL2 索引，按块旋转、查找码字；不生成全库
  FP32 重构文件。近似平局按直接 FP64 距离复核，严格平局取最小 code ID。
  冻结模型后记录 `encoding-parity.json`，同时比较原 Torch 和 FP64 参考。
  **只替换普通几何编码；中心化目标、support 正则与节点重编码仍用原算法，
  不把它们改成普通欧氏最近邻。**
- OPQ-PQ/QAQ 使用 16 候选并行 PQ LUT gather；RaBitQ/SAQ 使用 packed-code
  批量解包和 FMA；LeanVec 使用 FP16 bounds 转换、8-bit codes 和批量 FMA。
  图搜索已调用这些 `ScoreBatch`，不是只在测试工具中使用。
  很小批次保留标量路径；超过 2 GiB 的 code payload 使用 64-bit gather offsets。
  primary、图、residual 码长及线上评分定义不变。

本机旧 `faiss-gpu 1.7.2` 在 Ada 上实际 K-means 出现 cuBLAS 错误。
已隔离编译官方 Faiss v1.9.0（CUDA 12.1、sm89、AVX512），不修改原 conda
环境。`run_all.sh` 自动加载 `.runtime/faiss-v1.9.0-build/faiss/python/build/lib`；
可用 `FAISS_PYTHON_PATH` 指定其他兼容安装。实际版本、模块路径、编译选项
记录在 `environment.csv`，不以旧 pip distribution metadata 代替实际加载版本。
`faiss_backend.py` 先加载 Torch，避免混用 CUDA 12 runtime 的加载顺序问题。

换机器或 Python ABI 时重新构建，勿直接复制本机二进制（本机 Python 3.9，
GPU sm89）。重建命令；需安装 CUDA toolkit、CMake、编译器和 BLAS/LAPACK：

```bash
cd /home/xzl/REMORA
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
  FAISS_CUDA_TOOLKIT=/usr/local/cuda-12.1 \
  FAISS_CUDA_ARCH=89 \
  bash ours/test/t2i-10M/build_faiss_runtime.sh
```

指定 GPU 2 的完整实验（仍为 CPU 8 线程在线性能测试）：

```bash
cd /home/xzl/REMORA
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
  CONFIG=ours/test/t2i-10M/config.gpu-faiss.json \
  COMMAND=preflight bash ours/test/t2i-10M/run_all.sh

PYTHON_BIN="$CONDA_PREFIX/bin/python" \
  CONFIG=ours/test/t2i-10M/config.gpu-faiss.json \
  bash ours/test/t2i-10M/run_all.sh
```

新输出根为 `outputs/t2i-10M/remora-gpu-faiss-simd-v2`，原结果不覆盖。
不设置 `GPU=2` 代替配置中的 `cuda:2`，本入口以 config 为准。

一致性测试：

```bash
PYTHONPATH=".runtime/faiss-v1.9.0-build/faiss/python/build/lib:ours/test/t2i-10M:ours/python:ours/test/t2i-1M/python" \
  FAISS_TEST_GPU=2 QVRR_BUILD_DIR="$PWD/ours/build-t2i10m-remora-v1" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$CONDA_PREFIX/bin/python" -B -m unittest -v \
  test_faiss_backend test_qvrr_simd test_integration
```

已通过 14 项测试：六档码率 CPU/GPU 编码、GPU 2 初始化和完整小规模训练、
短批次/尾部/跨字 packed codes、超过 2 GiB 偏移、旧 side 格式兼容等。
实际五种 primary 的 32 查询评分审计与 100 查询 free-running 搜索对照也通过。
LeanVec 标量/SIMD 分数有约 `6e-8` 的最大舍入差，所测 Top-10 集合和搜索结果
相同；其余四种分数逐位一致。不能据此保证所有未测试 near-ties 逐位相同。
审计报告在 `outputs/t2i-10M/faiss-simd-validation-v2/`。
实际 residual 编码审计使用各 primary 的 1536 个 fit 节点和 512 个独立 holdout
节点，冻结旋转/码本后 Faiss CPU、GPU 2 和 FP64 assignments 一致。旧 Torch
在 SAQ 上有 1/4096 个 assignment 受 FP32 临近平局舍入影响；报告保留该实例
和 FP64 距离差，其余四种与 Torch 一致。不将这种数值差异隐藏为100%一致。
该报告的 `speedup` 是 warm-code kernel 微基准，不是端到端 QPS；正式性能
收益必须通过新的独立扫描测量。GPU/CPU K-means 浮点累加不同，重新训练的
码本不保证逐位一致；编码对照使用冻结的同一份码本与旋转。
