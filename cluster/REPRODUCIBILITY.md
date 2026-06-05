# Cluster vs WSL — 可行性 & 可复现性 (2026-06-05)

## 推理引擎
用 **vLLM**(快), 不是 transformers。pipeline = run_vllm.py → vLLM HTTP server (Qwen2.5-32B-Instruct-AWQ)。
(cluster 上 `transformers==4.47.1` 只是 vLLM 内部依赖, 非 transformers 推理路径)

## WSL (当前主用)
- vLLM 常驻 :8000, 独占 GPU, 即时随时跑。适合**快速迭代**。
- 缺点: "在我机器上能跑", 论文复现性弱(手动起服务/有状态)。

## CMU Mind cluster — mode A (self-contained sbatch) 已验证可行
- 登录: ssh -l yizhouc3 mind.cs.cmu.edu (免密)。anaconda/cuda 在**计算节点**, 登录节点没有 → 环境/作业都在计算节点。
- 关键环境坑(已解): vllm 0.6.6 与 transformers 不配会 `ProcessorMixin` 崩 → pin `transformers==4.47.1`。
  脚本: cluster/setup_env.sh (经 srun 在 CPU 节点建 conda env `medllm`)。
- 运行: cluster/run_pipeline.slurm —— 一个 sbatch 内: 起 vLLM 后台→健康检查→跑 run_vllm.py→teardown。
  `sbatch cluster/run_pipeline.slurm exp/xxx.yaml`。
- GPU 适配: L40S 48GB 稳装 32B-AWQ+16k 长 note; 24GB 卡(a5000/titanrtx/3090)偏紧(OOM/截断风险); 12GB(titanxp/2080Ti)装不下。

## 可复现性 vs 响应性 (核心结论)
| | WSL | cluster mode A |
|---|---|---|
| 可复现性 | 弱(手动/有状态) | **强**(一条 sbatch 固化全流程, 锁版本) |
| 响应性 | **即时**(常驻独占) | 排队(共享调度, walltime 有限, 会被抢占) |
| "随时有机器吗" | 是(独占) | 不保证特定卡。L40S 当时被多用户占满→估等1-2天(保守, 实际常更快); 24GB 卡通常更空但对本任务偏紧 |

**Slurm 无"永久常驻服务器"是设计如此**(批处理+walltime+抢占)。"随时可用"的体验来自 WSL。
cluster 的"随时"= 提交作业排队; 小作业(~25min)投到不挤的卡通常分钟~小时级起跑。

## 用法建议
- 快速迭代 → WSL。
- 最终可复现归档跑 → cluster mode A。想缩短等待: 把 --gres 从 L40S 放宽(分别投 L40S/a5000/titanrtx), 落到先空的卡。
- 两者共用同一 run_vllm.py, 互不冲突。

## 文件
- cluster/setup_env.sh (一次性环境, 经 srun 跑)
- cluster/run_pipeline.slurm (mode A 作业)
