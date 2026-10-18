# REMORA 从建图到实验的运行入口

本目录管理数据和 primary 契约、建图以及实验一至五。单独运行 Alignment 或完整方法训练请使用根目录的 `method.py`，步骤见[最新 README](../../README.md)。方法主体已整理到 `ours/method/`，原训练脚本保留兼容入口；变更后使用新的实验输出目录以符合源码身份校验。所有正式 REMORA 分支均使用 **REMORA traversal**。

## 运行顺序

纯源码仓库根目录新增 `build_graph.py` 与 `remora.py`：

```bash
python build_graph.py --dataset t2i-10M --method opq-pq \
  --base /path/to/base.fbin --build-queries /path/to/query.build.fbin \
  --degree 40 --build-beam 500 --knn-k 100 --build-threads 64 --gpu 0
python remora.py train --dataset t2i-10M --config /path/to/prepare.config.json --execute
python remora.py evaluate --dataset t2i-10M --experiments 1,3,4 --gpu 0 --execute
```

默认只打印计划，`--execute` 才运行。新图命令串起原生 primary 编码、分块 Faiss GPU 全库 KNN 与构图；图内的投影/剪枝仍是 CPU native builder。正式 Ours 始终 REMORA traversal。SIFT/GIST 不在支持列表中，不隐式将 L2 转成 IP。完整参数与边界见根 README 的 Unified interface。

同一请求复用完整编码/KNN/图；KNN 是两代数据库块 checkpoint，已验证中断恢复。输入/配置变化拒绝复用；不覆盖已有未验证图。新建图用于 REMORA 前需正式导入清单和独立查询/GT。现有图直接导入，不重新量化。

```text
原始 embeddings 与独立 query 划分
  → primary 训练、编码、评分一致性验证
  → 用构图训练 queries 计算同一 primary 下的 Top-100
  → 构建 primary RoarGraph
  → 导入配套 graph、primary model、primary codes
  → REMORA 全评分残差与 Balanced 表示
  → Primary traversal 收集训练与验证候选
  → 候选条件中心化与支持度正则化重编码
  → 全库 residual codes 与在线模型
  → REMORA traversal 搜索
  → 不精排或本分支队列 Top-L 的 FP32 磁盘有限精排
  → 实验一至五及 CSV、图表
```

训练候选由 Primary traversal 产生，是离线监督采样；正式性能、倒挂和轨迹实验中的 Ours 是 REMORA traversal。Primary 与 REMORA 对照复用同一 primary、图、entry point 和 stopping policy。构图 Top-100 与测试 ground truth 是不同文件：前者使用训练构图查询和对应 primary 的评分，后者用于检索评估。

## 目录职责

| 位置 | 用途 |
|---|---|
| `ours/workflow/` | 本次整理的统一入口、目录索引、构图配置模板和入口测试 |
| `datasets/` | 原始或处理后的向量、primary model/codes、已有图、transfer manifest |
| `ours/src/`、`ours/include/` | primary、图搜索、FP32 存储、SIMD 和 residual 在线评分 |
| `ours/python/` | primary 导出、残差与度量、模型导出、数据读写 |
| `ours/test/t2i-10M/` | 当前跨数据集正式协议、训练器、性能采样、原生格式适配与报告 |
| `ours/scripts/`、各数据集 `ours/test/` | 冻结实验调度和数据集输入适配 |
| `ours/experimental/traversal-fast-v1/` | 正式搜索采用的原生快速评分 overlay，并非可随意删除的旧实验 |
| `ours/experimental/relative-rate-v1/` | 大于 256 bit 的 relative-rate 实验依赖 |
| `ours/test/t2i-1M/python/`、`ours/test/t2i-1B/` | 当前流程仍引用的共享历史依赖；不能整目录删除 |
| `outputs/` | 冻结配置、模型、测量记录和报告；不是代码 |
| `ours/build-*/`、`.runtime/` | 编译产物和隔离 Faiss 运行时；换机器可重建，但运行时不能删除 |

详细文件对应关系见 [FILE_MAP.md](FILE_MAP.md)。本目录不提供自动清理或删除命令，索引没有列到的文件也不因此成为可删除文件。

## 已有图的正式实验入口

在 `~/REMORA` 中运行。`PYTHON_BIN` 应指向含 NumPy、PyTorch 和配套 Faiss 的环境；系统 Python 可以列目录与生成计划，但不一定能训练。

```bash
cd ~/REMORA
export PYTHON_BIN="$CONDA_PREFIX/bin/python"

python3 ours/workflow/run.py list
python3 ours/workflow/run.py check --dataset t2i-10M
python3 ours/workflow/run.py reports --dataset t2i-10M
```

`catalog.json` 是本机已验证数据和冻结输出路径的索引，不是通用数据下载配置。换服务器时保留目录结构，并更新其绝对数据路径、cold-store 路径以及原实验配置中的路径。T2I 数据与导入清单仍有部分放在 `/home/xzl/RoarGraph`，不要只传 REMORA 而遗漏这些输入。

### REMORA 训练和全库编码

```bash
python3 ours/workflow/run.py plan --dataset t2i-10M --stages preflight,smoke,prepare
python3 ours/workflow/run.py run --dataset t2i-10M --stages prepare --execute
```

已有完整 refiner 的 `prepare` 会验证在线模型与 side-code header、节点数、码长、模型身份和文件大小后复用，不再训练或编码。没有完整输出时才调用原 `prepare` 流程；原生流程仍会校验输入、配置和评分契约。冻结目录中若配置发生变化，必须选择新的输出配置，不能强行覆盖。首次导入新数据集应使用相应 `prepare_inputs.py` 生成实验配置，再注册到 `catalog.json`，不能仅改一个 dataset 名字复用别的数据。

### 实验一 Performance Study

```bash
python3 ours/workflow/run.py plan --dataset t2i-10M --stages exp1
python3 ours/workflow/run.py run --dataset t2i-10M --stages exp1 --execute
```

调用原 adaptive capped scheduler，保留已完成点与原配置，默认 CPU 8 线程绑核、磁盘 `O_DIRECT` 有限 rerank、`L≤200`，同时保留 no-rerank 分支。每个分支从自己 traversal scorer 的队列选 Top-L，不额外先用 terminal scorer 改候选顺序。是否绘制 FP32 reference 由该数据集原配置决定；在量化图上跑 FP32 scorer 不可标成独立 FP32-built 图。

### 实验三和四 倒挂与轨迹

```bash
python3 ours/workflow/run.py plan --dataset t2i-10M --stages exp3,exp4
python3 ours/workflow/run.py run --dataset t2i-10M --stages exp3,exp4 --execute
```

两个阶段合并调用一个冻结诊断入口，不重训、不建图、不新增 QPS 扫描，也不做三遍 timing。邻近倒挂、边界倒挂使用 full-database exact truth；所有方法重评同一组 FP32 teacher routing pairs。轨迹同时记录 visited、retained coverage 与输出 Recall。Equal-Recall 表复用已完成实验一的观测，在固定 L 下选满足目标的最少 visited 实测点，不外推未达目标。

`StopInv` 只有真实 score-based stop decision 才有意义；队列耗尽时报告 N/A。停止边界 proxy 必须单独命名，不能冒充真正 StopInv。

### 实验二 消融

```bash
python3 ours/workflow/run.py plan --dataset t2i-10M --stages exp2
python3 ours/workflow/run.py run --dataset t2i-10M --stages exp2 --execute
```

使用冻结 Full checkpoint 和同一 FP32 teacher pairs。固定 side budget、图、primary、beam、候选池与 rerank policy；包含 Euclidean、Alignment、Balanced、候选 MSE 固定正则、Centered 固定正则、Support-aware Full。默认测 beam 100/200、L 0/100；性能测量仍三次，训练与 timing 串行。无 teacher 输入时先跑实验三。

### 实验五 码率

```bash
python3 ours/workflow/run.py plan --dataset t2i-10M --stages exp5
python3 ours/workflow/run.py run --dataset t2i-10M --stages exp5 --execute
```

调用原 rate trainer 和 quality audit，然后为每个码率、beam 做一次 Recall 正确性运行；复用已有匹配观测，不输出其 QPS 或 timing 作为性能结果。T2I 默认 32/64/96/128 bits；WebVid 与 RS5M 默认 80/160/240/320 bits，保持 0.5/1/1.5/2 倍相对预算。当前正式算法是 Balanced Product，不是早期的 `D_G → DP → grouping`；若要对照 DP，应另立消融，不能把两种训练器混称同一实验。

| 注册数据集 | 原始维度 | Primary 目标码长 | 默认 side bits | 实验五预算 |
|---|---:|---:|---:|---|
| T2I-1M 与 T2I-10M | 200 | 约 800 | 64 | 32、64、96、128 |
| WebVid-2.5M 与 RS5M | 512 | 约 2048 | 160 | 80、160、240、320 |
| WordVector-ID 自划分 899K | 300 | 约 1200 | 96 | 48、96、144、192 |

Primary 是 native configuration 下的目标区间，不保证各方法恰好相同位数：RaBitQ padding、SAQ 分配与辅助元数据须单独报告。WordVector 当前只有 Top-100 truth，三四可运行；旧消融和 rate quality 入口依赖 Top-200，补齐前会阻止启动，不能凭 Top-100 拼造 Top-200。RS5M 三四五复用已验证的只读 Top-200 reference view，不改原 Top-100 性能输入；其旧消融入口还要求 prepared 数据也有相同 Top-200，未补齐时明确阻止启动。

新诊断、消融、码率结果默认写到 `outputs/<dataset>/workflow-v1/{exp34,exp2,exp5}`，旧报告保持不动。`reports` 只定位已有报告，不重新计算或修改它们。`run` 不加 `--execute` 仍只是列计划。

## 从原始向量编码和建图

没有现成图时按下面顺序操作；不要把完整旧 `t2i-1B prepare` 当成最新方法一键入口，它还包含早期 JI-Pure DP/grouping 训练。

### 编译原生构图和搜索目标

```bash
cmake -S ours -B ours/build-workflow-v1 -DCMAKE_BUILD_TYPE=Release \
  -DOURS_JI_KERNEL_ROOT="$PWD/ours/experimental/traversal-fast-v1" \
  -DRABITQ_ROOT="$PWD/thirdparty/RaBitQ-Library" \
  -DSAQ_ROOT="$PWD/thirdparty/SAQ"
cmake --build ours/build-workflow-v1 -j 16 \
  --target roar_primary_ji_build roar_fp_build roar_primary_ji_search roar_exp234_audit
```

新 build 目录不覆盖正式测量使用的旧二进制。`-march=native` 只使用本 CPU 支持的指令，换服务器要重新编译，不能仅凭该选项声称 AVX-512 已生效。

### Primary 编码

已有图必须复用与构图完全一致的 primary model/codes，不重新训练替代。原生编码来源包括 `train_generic_primary.py`、`roar_rabitq_quantize` 与 `roar_export_saq_primary`；传入数据集维度和合法 native 预算，不使用历史默认 200→192 来凑码率。

例如 **新的 200D OPQ 数据集**，以下是 CPU generic 训练导出的配方，不是已有图的修复命令，也不是 1B 级流式 GPU 编码器：

```bash
"$PYTHON_BIN" ours/python/train_generic_primary.py \
  --method opq-pq --base-fbin datasets/NEW_DATASET/data/base.fbin \
  --query-fbin datasets/NEW_DATASET/data/query.train.fbin \
  --out-dir datasets/NEW_DATASET/primary/opq-pq \
  --opq-dim 200 --pq-groups 100 --primary-budget-bits 800 --threads 64
```

该旧 generic 导出会生成全库 reconstruction 中间文件，大库需预先算清磁盘与内存开销。官方 RaBitQ/SAQ 参数必须按 native 配置与实际 header 校验，不能套 OPQ 参数。LeanVec 降维本来就是其方法的一部分，需要独立验证集选配置并报告完整评分残差。当前 QAQ/LeanVec 接口包含本地实现，不能不加说明就当成论文官方实现。

### 构图查询 Top-100

对每个 primary，构图训练 queries 在**同一 primary reconstruction/scoring geometry**下搜索全数据库 Top-100。保留 query 范围、原始 ID 映射、模型身份、实际码率及 KNN backend 的 provenance。GPU 放不下全库时使用分块 Faiss GPU scan 与 Top-100 合并，不能只扫描一个 shard 冒充全库，也不能把测试 GT 拿来构图。

LAION-400M 的流式 OPQ、GPU 分块 Top-100 和码字两两内积预计算构图实现目前在配套 RoarGraph checkout：`ours/scripts/laion_opq_full.py` + `laion_opq_build.py`，入口 `run_laion_opq_full.sh`。它使用自己的 build 版本、checkpoint 和 frozen config；见 [FILE_MAP.md](FILE_MAP.md)。本次整理没有切换或重启正在运行的 LAION 作业。

### Native primary 或 FP32 构图

复制并填好本目录 `configs/build-native.example.json` 或 `build-fp32.example.json` 的路径与参数，再执行：

```bash
python3 ours/workflow/build.py plan --config ours/workflow/configs/build-native.example.json
python3 ours/workflow/build.py check --config path/to/my-build.json
python3 ours/workflow/build.py run --config path/to/my-build.json --execute
```

入口要求已存在的构图 KNN，阻止构图器因 KNN 缺失而偷偷改用 CPU 全库扫描；启动前检查维度、KNN shape、primary 格式和 model/code identity，再执行 native score/decode parity。已有 graph 或 summary 一律拒绝覆盖，直接导入复用或选择新的输出。

当前 native builder 支持 `GPRIM01/GPCOD01` generic、`RBQROAR1` 官方 Extended-RaBitQ、`SAQCODE1` 官方 SAQ。导入包的 `OPQPQ002/ERBQRG/SAQRG/LVOOD` 文件由搜索 reader 支持，不代表旧 builder 能读：它们必须复用配套图，或经过格式转换及评分一致性验证后才建新图。后缀 `.bin` 不能替代格式检查。

SIFT/GIST 明确排除，不做这些实验。1B/400M 构图必须另做可用 RAM 和图内存估算，本目录不会把小规模配置自动扩成磁盘图算法。

## 代码索引与运行日志

```bash
python3 ours/workflow/run.py inventory > remora-source-inventory.csv
python3 -m unittest discover -s ours/workflow/tests -v
```

CSV 索引包含源码角色、相对路径、大小和 SHA-256，不包含 datasets、outputs、build 缓存或第三方依赖全集。它是维护清单，不是独立可运行的 transfer bundle；传输时仍需保留 root graph library、第三方库、原生 reader 和隔离 Faiss 的构建资料。运行日志在 `outputs/<dataset>/workflow-v1/logs/`，记录真实命令、阶段、时间和退出码；没有可信运行速度时 total time 标为 unknown，不伪造进度估计。
