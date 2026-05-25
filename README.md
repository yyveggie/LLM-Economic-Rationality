# LLM Economic Rationality — Replication Toolkit

复现并扩展 Chen, Liu, Shan & Zhong (2023) *PNAS*
"The emergence of economic rationality of GPT" 的实验流水线。

支持的能力：

- 完整复现 4 偏好领域 × 3 条件 × 温度变体 × 人口学变体的 baseline 实验。
- 单模型设计：在 `configs/models.yaml` 里配 1 个模型，跑完换下一个。
- 全套指标：CCEI / HMI / MPI / MCI / Spearman + CES/DA 偏好参数估计。

## 快速开始

```bash
pip install -r requirements.txt

# 复制模板，填入你的 api_key
cp configs/models.example.yaml configs/models.yaml

# 编辑 configs/models.yaml 里的 provider/model/base_url/api_key
# 编辑 configs/experiments.yaml 把 active 改成你想跑的实验，或用 --experiment 覆盖

python run.py --experiment smoke_test           # 冒烟测试
python run.py --experiment baseline_quick --plots
```

> ⚠️ `configs/models.yaml` 已加入 `.gitignore`，不会被 commit。
> 仓库里只追踪 `configs/models.example.yaml` 模板。

## 如何启动一个实验

两种方式，二选一。

**方式 A：改配置 + 直接跑**

```yaml
# configs/experiments.yaml 顶部
active: smoke_test       # 改成你想要的实验名
```

```bash
python run.py
```

**方式 B：命令行临时指定（推荐）**

```bash
python run.py --experiment smoke_test
```

### 常用命令组合

```bash
# 跑 LLM + 算指标 + 出图（一键到底）
python run.py --experiment baseline_quick --plots

# 已经跑过 LLM 了，只重算指标和重画图
python run.py --experiment baseline_quick --skip-llm --plots

# 只跑 LLM，先攒数据，回头再分析
python run.py --experiment baseline_quick --skip-analysis
```

### 跑前检查清单

1. `pip install -r requirements.txt` 装好依赖
2. `configs/models.yaml` 里的 `api_key` 已填
3. 第一次先跑 `smoke_test` 确认能通，再上大实验

### 结果落在哪

```
results/<experiment>/<model_key>/raw_decisions.csv  # 每轮决策原始数据
results/<experiment>/data.csv                       # 汇总指标 (CCEI 等)
results/<experiment>/figures/Figure1_CCEI.pdf       # --plots 才生成
results/<experiment>/figures/Figure2_Spearman.pdf
results/<experiment>/figures/Figure3_Preferences.pdf
results/<experiment>/figures/Figure4_Variations_<model>.pdf
```

## 跨模型对比

`configs/models.yaml` 一次只配一个模型。要比较多个模型：

```bash
# 1) 在 models.yaml 把 model 改成 gpt-3.5-turbo，跑实验
python run.py --experiment paper_baseline

# 2) 改 models.yaml 换成 claude-sonnet-4-5，再跑一次
python run.py --experiment paper_baseline

# 3) 每个模型的结果会落在 results/paper_baseline/<model_key>/
#    画图脚本自动把多个模型同图对比
python run.py --experiment paper_baseline --skip-llm --plots
```

`results/<experiment>/data.csv` 是聚合后的总表，行带 `model` 字段，
绘图脚本会按 `model` 分组画线/散点。

## `configs/models.yaml` 字段

| 字段 | 含义 |
|------|------|
| `provider` | `openai` / `anthropic` / `openai_compatible` |
| `model` | 服务方使用的模型 ID |
| `base_url` | OpenAI 兼容服务的入口 URL（官方 OpenAI/Anthropic 留空）|
| `api_key` | 直接填字符串（本地无 key 服务填 `""`） |
| `default_temperature` / `max_tokens` / `request_timeout` | 推理参数 |
| `concurrency` | 并发请求数（按服务方限速调） |

文件里以注释形式列了 OpenAI / Anthropic / DeepSeek / Qwen / Moonshot /
GLM / 本地 vLLM 的速查模板，复制覆盖即可换模型。

## `configs/experiments.yaml` 内置实验

| 名称 | 用途 |
|------|------|
| `smoke_test` | 1 领域 × 2 subject 的最小冒烟测试 |
| `baseline_quick` | 4 领域 baseline，每个 20 subject |
| `paper_baseline` | 4 领域 baseline，每个 100 subject (Fig.1–3) |
| `framing_sensitivity` | baseline + price framing + discrete choice |
| `temperature_sweep` | 温度 0 / 0.5 / 1.0 |
| `demographic_variants` | 8 种人口学描述符 |
| `full_replication` | 上述全部条件 × 全部变体 |

## 实现细节与论文一致性

- 任务生成：M, N ∈ [0.1, 1] 且 max(M, N) ≥ 0.5，保留两位小数（SI §1）。
- Prompt：逐字对照 SI §1，包括 system / assistant / user 三段消息以及
  baseline / price-framing / discrete-choice 的差异。
- CCEI：对效率参数 e 二分，每个 e 上构造 R⁰ / P⁰ 矩阵做传递闭包检验 GARP。
- HMI：≤5 删除穷举；MCI：贪心上界；MPI：2-cycle 平均近似。
- 偏好参数估计：SI 中的对数需求方程
  `ln(x1/x2) = (1/(ρ-1)) ln(p1/p2) + ln((1-α)/α)`；
  Risk 域按论文用 `x1 = max(xA, xB)` 进行 better-outcome 重排。

## 注意事项

- price framing / discrete choice 不再做温度或人口学变体（与论文一致）。
- 不同模型回答鲁棒性差异大；解析失败的轮次写为 NaN，按 subject 自动剔除。
- 跑全量 `paper_baseline` 大约 10 000 次 API 调用，按你的并发预计 1–2 小时。
