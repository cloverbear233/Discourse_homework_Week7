# Week7 语言模型训练与对比分析平台

## 当前版本

- 模块1：n-gram + Add-one 平滑（可用）
- 模块2：RNN 自定义训练（占位）
- 模块3：预训练模型对比（占位）
- 模块4：综合分析看板（占位）

## 安装与启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 模块1说明

- 默认优先使用 NLTK Reuters 语料。
- 若 Reuters 未安装，可在页面点击按钮下载。
- 也可切换到手动输入文本模式作为兜底。
- 支持 n=2/3、未平滑与 Add-one 平滑概率对比、步骤级分解展示。
