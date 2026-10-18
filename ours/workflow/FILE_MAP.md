# REMORA 正式流程的代码文件索引

纯源码包的根入口为 `build_graph.py` 和 `remora.py`，由 `ours/workflow/interface.py` 实现。底层保留 `ours/workflow/run.py` 和 `ours/workflow/build.py`。实现仍在原路径，下面区分正式依赖、构图实现和历史研究入口，避免因为文件名字带有旧实验编号就误删共享代码。SIFT/GIST 不作为实验数据集。

## 数据与导入

| 文件 | 职责 |
|---|---|
| `ours/workflow/interface.py` | dataset/method/图超参数统一 CLI；新图编码、KNN、构图与已导入图的训练/实验调度 |
| `ours/workflow/knn.py` | 原生五-primary/FP32 分块 Faiss GPU/CPU exact KNN、Top-K 合并、双代块 checkpoint 与 parity |
| `ours/workflow/artifacts.py` | 严格矩阵格式、输入恢复身份、原子元数据与查询前缀切片 |
| `ours/test/webvid/prepare_inputs.py` | WebVid 文件与原生 primary 契约、query 划分、ground truth score convention |
| `ours/test/rs5m/prepare_inputs.py` | RS5M 输入、模型和图 provenance |
| `ours/test/wordvector-id/prepare_inputs.py` | 原生 raw-IP WordVector ID 数据自划分、实际 899K 节点及 Top-100 真值 |
| `ours/test/t2i-1M-native100/run_experiment1.py` | T2I-1M 的 native100 primary、既有图导入及实验配置 |
| `ours/test/t2i-10M/native100_primary.py`、`qvrr.py` | 多种原生模型格式、评分和完整原空间 reconstruction |
| `ours/src/native_primary_reader_main.cpp` | 原生模型解码 helper；纯源码包已从配套 RoarGraph checkout 补齐源码和编译 target，原 REMORA 目录此前只有 reader 运行产物 |
| `ours/python/drb_io.py`、`ours/test/t2i-1M/python/matrix_io.py` | 向量、ID、模型及矩阵 IO |

## Primary 和图

| 文件 | 职责 |
|---|---|
| `ours/python/train_generic_primary.py` | generic OPQ、QAQ、LeanVec 训练与导出；显式覆盖旧预算和维度默认值 |
| `thirdparty/RaBitQ-Library/sample/roar_rabitq_quantize.cpp`、`ours/src/export_rabitq_primary_main.cpp` | 官方 Extended-RaBitQ 编码和导出 |
| `thirdparty/SAQ/src/export_roar_primary.cpp` | SAQ 模型与 code 导出 |
| `ours/src/primary_ji_build_main.cpp` | native scorer 的 Top-K、parity 与 primary RoarGraph 构造 |
| `ours/src/fp_build_main.cpp` | FP32 RoarGraph 构造 |
| 根目录 `src/index_bipartite.cpp`、`include/index_bipartite.h` | RoarGraph 图投影、剪枝与构造库，不能只打包 `ours/` |
| `ours/CMakeLists.txt`、`thirdparty/` | 编译入口和 primary 上游依赖 |

LAION 的最新大库构图源自 `/home/xzl/RoarGraph` checkout；纯源码包单独保存在 `build-source/`，其完整链路为：

```text
download_laion_embeddings.py
  → laion_opq_full.py
      ↳ laion_opq_build.py
  → 对应新版本 roar_primary_ji_build
```

同目录的 `run_laion_opq_full.sh` 与 launch 脚本是调度入口。LAION 当前配置为 512D 原空间、OPQ M=256、8 bit/codeword、10M 构图查询、Top-100；subset runner 的历史默认不是正式全库预算。搬迁时必须一同传这两份 Python 实现、它们所编译的构图源码与实际配置，不要用 REMORA 中更旧的 builder 悄悄替换。

## REMORA 离线训练与在线评分

| 文件或目录 | 职责 |
|---|---|
| `method.py`、`ours/method/cli.py` | 独立方法入口：候选池、Alignment 变体、初始化和完整 training |
| `ours/method/alignment.py` | 完整评分残差、query-visible JI、Balanced Product 和初始码本 |
| `ours/method/training.py` | 候选中心化、支持度正则化、重编码与验证选择 |
| `ours/method/encoding.py` | 分块全库编码与原生在线模型导出 |
| `ours/test/t2i-10M/train_remora.py` | 保留旧参数和 Python API，转调 `ours/method/` 的兼容入口 |
| `ours/test/t2i-10M/faiss_backend.py` | 指定 GPU 的初始化聚类、Faiss 几何编码和 tie/parity 校验 |
| `ours/test/t2i-10M/pipeline.py`、`protocol.py` | 数据与配置校验、候选收集、模型与点身份、预算及 CSV 记录 |
| `ours/test/t2i-10M/scoring_contract.py` | traversal、own-queue rerank 与原生 summary 契约 |
| `ours/method/encoding.py` 内的 `export_model` | 当前 LUT 型 residual 在线模型与 byte-aligned side codes 导出；旧 JI-Pure 的单独导出器不是正式训练入口 |
| `ours/experimental/traversal-fast-v1/src/primary_ji_search_main.cpp` | 正式搜索与诊断的快速原生主程序 overlay |
| `ours/experimental/traversal-fast-v1/src/ji_online.cpp` | 复用 residual LUT 和 residual scoring kernel |
| `ours/src/generic_primary.cpp`、`qvrr_primary.cpp` | primary 批量评分、原生格式适配与 SIMD |
| `ours/src/exact_vector_store.cpp`、`fp32_inner_product.cpp` | 磁盘 O_DIRECT / mmap / 内存 FP32 与 exact dot kernel |
| `ours/experimental/relative-rate-v1/` | 320 bit 等大预算训练与对应原生模型能力 |

实际 transitive imports 还包括 `ours/test/t2i-1M/python/` 的旧命名 helper 和 `ours/test/t2i-1B/pipeline.py` 基类。当前不能通过简单删除历史目录得到最小依赖闭包。

## 五项实验

| 实验 | 当前统一入口调用的实现 | 默认是否新增 QPS |
|---|---|---|
| 一 Performance | `ours/scripts/run_native100_capped_experiment1.py`、`ours/test/t2i-10M/adaptive_performance.py`、`run.py` | 是，原配置三次并复用完成点 |
| 二 Ablation | `ours/scripts/run_frozen_ablation.py`、`train_remora.py` | 是，训练和 timing 串行 |
| 三 Inversion | `ours/workflow/adapters/run_frozen_experiments34.py` | 否 |
| 四 Traversal | 同一冻结 audit，结合已完成实验一观测 | 否 |
| 五 Rate | `ours/workflow/adapters/run_accuracy_rate.py` 复用 frozen rate trainer 与 RS5M 的 Recall/report helpers | 否，每个 accuracy point 一次，原生计时字段丢弃 |

冻结 audit 的副本来自 `RoarGraph/ours/scripts/run_frozen_experiments34.py`，增加了适应 WordVector Top-100 的 ranks 选择与固定 L 的 min-visited 统计。它在新输出目录使用，不覆盖旧 runner，也不改变旧测量的源码指纹。

## 不作为正式起点的研究脚本

`ours/test/t2i-1M/scripts/` 下的 A1d/A1f/A1g/A1h、terminal-only、早期图修复、Additive、hull、fused LUT 和 rate 原型保留作研究记录，不作为新数据集的默认流程。旧 `run_complete_suite.sh` 和 `ours/test/t2i-1B/run.py all` 也不等于当前最终 REMORA 五项实验。它们仍可能引用共享 helper，因此本次没有删除、移动或改写它们。

`run.py inventory` 生成保守的源码索引及 SHA-256；未列文件不是删除白名单。`package_sources.py` 按正式入口的依赖收集纯源码包；保留来源与 SHA-256，排除数据、结果和编译产物。迁移后仍应在有数据的环境做评分一致性 smoke，源码编译通过不等于数据契约验证通过。
