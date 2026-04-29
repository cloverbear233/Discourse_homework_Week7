# NLP Week9：机器翻译机制与质量评测系统

## 运行环境

- 建议使用 `base` 环境。
- Python 3.9+。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 模型下载源（hf-mirror）

项目默认在代码里强制设置 `HF_ENDPOINT=https://hf-mirror.com`（在导入 `transformers` 前），会优先通过 hf-mirror 下载模型，不走直连 Hugging Face。

如需手动指定，也可在启动前执行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 依赖冲突说明（Keras 3）

如果你遇到类似报错：
`Keras 3 is not yet supported in Transformers`

本项目已通过代码禁用 TF 自动导入（优先走 PyTorch），并在 `requirements.txt` 中加入 `tf-keras` 作为兼容包。你可以：

```bash
pip install -r requirements.txt
```

## 启动应用

在 `NLP/week9` 目录执行：

```bash
streamlit run app.py
```

## 模块功能介绍

- 模块1：神经机器翻译引擎（NMT）
  - 输入英文句子，加载 `Helsinki-NLP/opus-mt-en-zh` 并输出中文译文（含加载 Spinner）。
- 模块2：基于规则的直译 vs. 神经网络意译
  - 使用简化英汉词典进行“逐词直译”（空格分词，词典没命中就保留英文）。
  - 将逐词直译结果与模块1的 NMT 译文并排展示，便于观察语序/消歧差异。
- 模块3：机器翻译质量自动评测（BLEU）
  - 输入 `Reference` 与 `Candidate`，计算 BLEU 分数并给出分数含义说明。
  - Candidate 可一键由模块1生成；同时提供多参考/多场景对比示例以观察 BLEU 对 n-gram 匹配与分词方式的敏感性。

## 使用说明（模块1）

1. 选择示例句（包含 `It rains cats and dogs.`）或输入英文文本。
2. 点击“开始翻译”。
3. 等待 Spinner 完成后查看中文译文。

> 提示：首次运行会通过 hf-mirror 下载 `Helsinki-NLP/opus-mt-en-zh`，耗时可能较长。
