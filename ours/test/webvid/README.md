# WebVid 实验一

现成输入：`datasets/WebVid-2.5M-native4-M40-L500`，2,505,000 个 512D 节点。

本轮对照为五种 primary 各自的 Primary traversal 与 REMORA traversal。
两者共享原 primary codes、码本、图和 entry point；不构图、不重新量化。
Primary 为 2048 nominal bits/node；REMORA 默认增加 160 bits/node
（20 个 8-bit learned groups，约 7.81% 相对预算，载荷 50.1 MB/primary）。

准备流程：原生评分和完整 raw residual 校验 → 独立训练/验证候选收集 →
query-visible/Balanced Product 初始化 → 候选中心化、支持度正则和节点改码 →
全库编码及 Faiss 编码一致性审计。K-means、几何编码、中心化训练使用 GPU 2。
训练候选由 Primary traversal 收集，只是离线监督；**正式 REMORA 分支始终使用
REMORA traversal 和 REMORA 队列评分，不是 terminal-only**。

性能流程：磁盘/内存输出 IDs 一致性检查 → 8 个 CPU 线程绑核 16–23 →
10,000 条测试查询，3 次重复，测 QPS、服务延迟 P50/P95/P99、Recall@10、
Visited 和内存组成。无精排和有限精排分开画图，primary 虚线、REMORA 实线。
FP32 精排使用 `/tmp/xzl-webvid-cold/base.2.5M.cold.bin` 的 NVMe O_DIRECT，
每条分支从**自身搜索队列前 L 个**候选精排，L≤min(beam,200)。
全流程串行，离线训练不会与正式计时同时运行；绑核不等于独占服务器。

原 ground truth 为使用归一化查询的 squared-L2 标签；导入时转换为 raw-query
内积分数并与实际 FP32 内积校验，原数据和查询不做重新归一化。
使用源服务器的独立 1024 条验证查询，验证和测试均不参与训练。
本包没有 FP32 图，所以本轮不提供 FP32 traversal 搜索曲线；不能以 exact
ground truth 的计算速度冒充 FP32 图搜索。

运行（已有完成阶段会复用；不要改变冻结配置/代码后直接复用同一输出目录）：

```bash
cd /home/xzl/REMORA
bash ours/test/webvid/run_experiment1.sh
```

分阶段：`WEBVID_PHASE=preflight|smoke|prepare|performance`。默认 `all`，准备成功后
自动启动性能扫描。输出根目录：
`outputs/webvid/remora-traversal-exp1-nvme-b160-v1`。
训练细节日志在 `prepared/logs`，性能在 `performance/experiment1`。
性能 CSV 保留原始测量；未达到的目标 Recall 不外推，不用平滑生成新测量值。

```bash
tmux new-session -d -s remora-webvid-exp1 -c /home/xzl/REMORA \
  'bash -o pipefail -c "bash ours/test/webvid/run_experiment1.sh 2>&1 | tee outputs/webvid/remora-traversal-exp1-nvme-b160-v1/run.log"'
tmux attach -t remora-webvid-exp1
```

已有同名 tmux 时不要再启动第二份。按 Ctrl-B 再 D 可离开会话，不中止实验。
