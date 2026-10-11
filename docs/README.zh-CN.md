# MemPursuit

论文 **Recall What Is Missing: Composing Evidence from Long-Term Agent Memory** 的代码实现。

MemPursuit 使用 simplex 约束的全校正 Sparse OMP，从候选记忆中逐步选择互补信息。算法复用查询和记忆的嵌入向量，根据重构残差选择下一条记忆，并从同一条选择序列生成不同长度的上下文。

项目提供独立的 NumPy/Numba 实现、基于 Mem0 的记忆系统，以及 LoCoMo、LongMemEval-S 和 BEAM-500K 的完整评估流程。

[English](../README.md) · [评估协议](experiments.md) · [许可证](../NOTICE)

## 安装

需要 Python 3.10 或更新版本、Git，以及带 Compose v2 插件的 Docker。默认配置在本地使用 CPU，通过外部 API 提取记忆、生成嵌入、回答和评分，需要联网并准备有足够额度的 API 凭据。

建议使用独立 Python 环境；本项目提供 `mem0` 命名空间，请避免在同一环境中同时安装 `mem0ai`。

克隆仓库并安装：

```bash
git clone https://github.com/FengQi-HITSZ/MemPursuit.git
cd MemPursuit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[pipeline,fast]'
python -m spacy download en_core_web_sm
cp .env.example .env
```

在 `.env` 中填写 `OPENAI_API_KEY`。默认使用 GPT-4o-mini 提取记忆、生成回答和评分，使用 `text-embedding-3-small` 生成 1536 维嵌入。通过 `OPENAI_BASE_URL` 配置兼容 OpenAI 的服务，也可用 `ANSWER_*`、`JUDGE_*` 变量分别配置回答和裁判模型。

## 下载数据集

```bash
python -m scripts.download_datasets locomo longmemeval beam
```

| Benchmark | 数据划分 | 评估规模 | 每次记忆提取的输入 |
|---|---|---|---|
| LoCoMo | LoCoMo-10，类别 1–4 | 10 个对话，1540 道题 | 1 条消息 |
| LongMemEval-S | Cleaned S | 500 道题 | 同一 session 中相邻的 2 条消息 |
| BEAM | 500K | 35 个对话，700 道题 | 同一 session 中相邻的 2 条消息 |

数据保存在 `datasets/`。两条消息组成一次提取输入，LLM 可从中生成多条记忆；最后不足两条时单独处理，不跨 session 配对。

论文主表使用 **K=50**。下列命令默认评估 K=10/20/50/200，对照[论文结果](experiments.md#datasets-and-results)时读取 K=50 的输出。重新构建记忆和调用模型可能产生分数波动，因为记忆提取和生成具有非确定性，托管模型版本也可能变化。

## 运行 LoCoMo

```bash
docker compose up -d --build
curl --noproxy localhost,127.0.0.1 http://localhost:8889/health
```

首次构建会下载 spaCy 和 BM25 模型。服务就绪后执行：

```bash
python -m benchmarks.locomo.run --project-name locomo-mempursuit --resume
```

默认召回最多 200 条候选记忆，使用固定预算 Sparse OMP 重排，评估 Top-10/20/50/200。检索阈值为 0.1，对话并发数为 2，请求超时为 900 秒。

小规模运行示例：

```bash
python -m benchmarks.locomo.run --project-name locomo-smoke \
  --conversations 0 --max-questions 1 --max-workers 1 --resume
```

该命令完整写入第 0 个对话，并评估其中 1 道题。逐题输出保存在 `results/locomo/predicted_<project>/`，汇总结果位于 `results/locomo/locomo_results_*.json`。

## 运行 LongMemEval-S 和 BEAM

启动四个记忆服务以并行构建和检索：

```bash
docker compose -f compose.native.yaml up -d --build
```

等待 8891–8894 端口的 `/health` 就绪后执行：

```bash
python -m benchmarks.longmemeval.run --project-name longmemeval-mempursuit \
  --mem0-hosts http://localhost:8891,http://localhost:8892,http://localhost:8893,http://localhost:8894 \
  --resume

python -m benchmarks.beam.run --project-name beam-mempursuit \
  --mem0-hosts http://localhost:8891,http://localhost:8892,http://localhost:8893,http://localhost:8894 \
  --resume
```

共享服务时，依次运行两个 benchmark。LongMemEval 默认写入/检索/评估并发为 12/12/16，BEAM 为 4/4/16，可通过 `--ingest-workers`、`--retrieval-workers` 和 `--max-workers` 调整。

也可通过 `--mem0-host http://localhost:8889` 使用单个服务。`--mode smoke` 选择较少的记忆作用域；`--max-questions 1` 限制评估题数，但仍完整写入所选题目需要的记忆。

输出位于 `results/<benchmark>/<project>/`，包含运行配置、写入检查点、检索结果、提示词快照、逐题评分及 `metrics.json`。

## 配置与续跑

- `configs/mempursuit.yaml`：固定预算，NumPy 后端。
- `configs/adaptive.yaml`：根据残差改善程度提前停止，NumPy 后端。
- `configs/numba.yaml`：固定预算，Numba 后端。

在 `.env` 中设置 `MEM0_CONFIG_FILE` 后重新创建记忆服务即可切换配置。LoCoMo 自适应评估还需添加 `--adaptive-context`，并使用新的项目名。

添加 `--resume` 可继续中断的任务。续跑记忆构建时，需要同时保留 Qdrant 数据、SQLite 历史数据库和结果目录。分开执行检索与评估、复用检索结果及切换回答模型的方法见 [英文指南](../README.md#resume-and-reuse-retrieval)。

评估协议说明了各数据集的检索阈值、上下文排序和评分方式。算法延迟测量命令见 [英文指南](../README.md#adaptive-selection-and-acceleration)。

## 许可证

项目采用 [Apache 2.0](../LICENSE) 许可证，第三方组件声明见 [NOTICE](../NOTICE)。数据集和模型遵循各自的许可证。
