# Week7 语言模型训练与对比分析平台

一个基于 Streamlit 的课程实验项目，用统一界面展示统计语言模型、从零训练 RNN、预训练模型推理与 PPL 评价。

## 项目目标

- 理解 n-gram 统计建模与平滑思想。
- 观察字符级 RNN/LSTM 的训练过程与生成行为。
- 对比 Masked LM（BERT）与 Causal LM（GPT-2）的任务差异。
- 使用困惑度（Perplexity）评价不同句子的可预测性。

## 功能总览

- 模块1：n-gram + Add-one（Laplace）平滑概率计算与分解。
- 模块2：字符级 RNN/LSTM 从零训练，实时 Loss 曲线与文本生成。
- 模块3：BERT `[MASK]` Top-5 预测 + GPT-2 续写对比。
- 模块4：GPT-2 逐句计算 Loss 与 PPL，并给出反差参考。

## 环境要求

- Python 3.10 或 3.11（推荐）。
- 可联网环境（首次运行时下载 NLTK Reuters 与 Hugging Face 模型）。
- 可选 GPU（未检测到 GPU 时自动走 CPU）。

## 安装与启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

启动后浏览器访问终端输出的本地地址（通常是 `http://localhost:8501`）。

## 依赖说明

`requirements.txt` 中核心依赖如下：

- `streamlit`：Web 可视化交互界面。
- `nltk`：Reuters 语料与基础 NLP 工具。
- `torch`：模块2训练与模块4计算的底层框架。
- `transformers`：BERT/GPT-2 推理与加载。

## 各模块说明

### 模块1：n 元语言模型与平滑

- 支持语料来源切换：`NLTK Reuters（优先）`（可一键下载并使用前300篇构建统计）或 `手动输入文本`（网络不可用时兜底）。
- 支持 `n=2` 或 `n=3`。
- 支持未平滑概率与 Add-one 平滑概率对比。
- 提供案例句（含零概率示例）与逐步分解表（context、count、prob）。

### 模块2：从零训练字符级 RNN/LSTM

- 可选择模型：`RNN` 或 `LSTM`。
- 可调超参数：`Hidden Size`、`Epochs`、`Sequence Length`、`Learning Rate`。
- 内置多组训练语料案例（规律/半规律）。
- 训练时实时显示进度与 Loss 曲线。
- 训练完成后可输入 Seed 生成指定长度文本。

### 模块3：预训练架构对比

- 左侧 BERT：对包含 `[MASK]` 的句子做 Top-5 预测。
- 右侧 GPT-2：基于 Prompt 续写，并展示完整生成与截断后续片段。
- 用于展示 Masked LM 与 Causal LM 的任务机制差异。

### 模块4：Perplexity 评价

- 使用 GPT-2 对每行句子计算 `Token 数`、`Cross-Entropy Loss` 与 `PPL = exp(Loss)`。
- 内置多组案例（新闻、技术、模板化句、乱码句）便于反差实验。
- 多句有效结果时会给出最高/最低 PPL 比值参考。

## 目录结构

```text
week7/
├── app.py
├── theme.py
├── requirements.txt
├── README.md
└── modules/
    ├── module1_ngram.py
    ├── module2_placeholder.py
    ├── module3_placeholder.py
    └── module4_placeholder.py
```

## 使用建议

- 模块1先用案例句观察未见 n-gram 的零概率问题，再勾选平滑比较变化。
- 模块2建议先用“规律语料”验证 loss 下降，再换自然文本观察生成质量差异。
- 模块3建议固定同一语义主题，分别写 `[MASK]` 句和 Prompt，便于横向比较。
- 模块4建议输入“通顺句 + 乱码句”混合列表，观察 PPL 对可预测性的响应。

## 常见问题

- `未检测到 PyTorch/transformers`：重新执行 `pip install -r requirements.txt`，并确认当前终端已激活 `.venv`。
- Reuters 下载失败：检查网络后重试，或在模块1切到“手动输入文本”模式。
- 首次加载 BERT/GPT-2 很慢：属于正常现象，模型会被缓存，后续运行会更快。

## 提交与去敏说明

- 本项目已通过 `.gitignore` 排除了 `pdf`、`实验报告.md`、缓存文件等内容。
- 建议提交前执行 `git status`，确认仅包含代码与必要文档。
