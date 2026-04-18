# -*- coding: utf-8 -*-
"""Module 4: GPT-2 perplexity evaluation."""

from __future__ import annotations

import math
import os

import streamlit as st

# Force Transformers to avoid TensorFlow/Keras code path in this project.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - runtime dependency fallback
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

DEFAULT_PPL_TEXT = (
    "The stock market closed higher today after strong earnings.\n"
    "Language models can generate coherent text with enough context.\n"
    "The quantum pineapple negotiated with martian bankers yesterday."
)
PPL_CONTRAST_TEXT = (
    "The committee approved the new policy after a long discussion.\n"
    "Banana quantum swiftly under cloud pencil whispers twelve."
)
PPL_NEWS_TEXT = (
    "The central bank said inflation is easing this quarter.\n"
    "Investors expect interest rates to remain stable.\n"
    "The company reported stronger revenue than analysts predicted."
)
PPL_TECH_TEXT = (
    "The model uses attention to capture long-range dependencies.\n"
    "Gradient clipping can improve training stability in recurrent networks.\n"
    "Tokenization quality directly affects downstream language modeling."
)
PPL_PATTERN_TEXT = (
    "hello world hello world hello world hello world.\n"
    "abcde abcde abcde abcde abcde.\n"
    "today is monday and tomorrow is tuesday."
)
PPL_GIBBERISH_TEXT = (
    "Cloud banana orbit seven pencil thunder quickly.\n"
    "Laptop river tomato sings under triangle engines.\n"
    "Glass quantum chair whisper zero cactus loudly."
)
PPL_CASES: list[dict[str, str]] = [
    {"label": "常规样例（混合）", "text": DEFAULT_PPL_TEXT},
    {"label": "PPL反差（通顺句 vs 乱码句）", "text": PPL_CONTRAST_TEXT},
    {"label": "新闻语体样例", "text": PPL_NEWS_TEXT},
    {"label": "技术语体样例", "text": PPL_TECH_TEXT},
    {"label": "高规律模板样例", "text": PPL_PATTERN_TEXT},
    {"label": "随机拼凑乱码样例", "text": PPL_GIBBERISH_TEXT},
]


@st.cache_resource(show_spinner=False)
def load_gpt2_for_ppl() -> tuple[object, object, str]:
    """Load GPT-2 tokenizer/model for perplexity evaluation."""
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("缺少 torch/transformers 依赖，无法加载 GPT-2。")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def compute_sentence_ppl(tokenizer: object, model: object, device: str, sentence: str) -> tuple[int, float, float]:
    """Compute token count, cross-entropy loss, and perplexity for one sentence."""
    if torch is None:
        raise RuntimeError("PyTorch 不可用。")

    encoded = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoded["input_ids"].to(device)
    token_count = int(input_ids.shape[1])

    if token_count < 2:
        raise ValueError("句子过短，至少需要 2 个 token 才能计算困惑度。")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = float(outputs.loss.item())

    try:
        ppl = float(math.exp(loss))
    except OverflowError:
        ppl = float("inf")

    return token_count, loss, ppl


def render_module4_placeholder() -> None:
    st.subheader("模块4：语言模型评价（Perplexity 困惑度）")
    st.markdown("使用 **GPT-2** 计算每条测试句的交叉熵损失，并按公式 `PPL = exp(Loss)` 输出困惑度。")

    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        st.error("未检测到 torch/transformers。请先安装依赖后再使用本模块。")
        return

    with st.container(border=True):
        if "week7_ppl_input" not in st.session_state:
            st.session_state["week7_ppl_input"] = DEFAULT_PPL_TEXT
        case_labels = [item["label"] for item in PPL_CASES]
        selected_case_label = st.selectbox("案例库（可一键载入）", options=case_labels, index=0)
        if st.button("载入所选案例", key="week7_ppl_load_selected"):
            selected = PPL_CASES[case_labels.index(selected_case_label)]
            st.session_state["week7_ppl_input"] = selected["text"]

        text_block = st.text_area(
            "输入多段测试句子（每行一条）",
            key="week7_ppl_input",
            height=180,
            help="建议每行输入一条英文句子，系统会逐条计算 Loss 与 PPL。",
        )
        st.caption("公式：Perplexity = exp(Cross-Entropy Loss)")

        if st.button("计算 PPL", type="primary"):
            sentences = [line.strip() for line in text_block.splitlines() if line.strip()]
            if not sentences:
                st.warning("请先输入至少一条测试句子。")
                return

            try:
                with st.spinner("正在加载 GPT-2 并计算，请稍候..."):
                    tokenizer, model, device = load_gpt2_for_ppl()
                    rows: list[dict[str, str]] = []
                    for sent in sentences:
                        try:
                            token_count, loss, ppl = compute_sentence_ppl(tokenizer, model, device, sent)
                        except Exception as exc:
                            rows.append(
                                {
                                    "Sentence": sent,
                                    "Tokens": "-",
                                    "Loss": "-",
                                    "PPL": "-",
                                    "Status": f"失败：{exc}",
                                }
                            )
                        else:
                            rows.append(
                                {
                                    "Sentence": sent,
                                    "Tokens": str(token_count),
                                    "Loss": f"{loss:.4f}",
                                    "PPL": f"{ppl:.4f}" if math.isfinite(ppl) else "inf",
                                    "Status": "成功",
                                }
                            )
            except Exception as exc:
                st.error(f"加载或计算失败：{exc}")
            else:
                st.dataframe(rows, use_container_width=True, hide_index=True)
                successful = [r for r in rows if r["Status"] == "成功"]
                if len(successful) >= 2:
                    try:
                        ppls = [float(r["PPL"]) for r in successful if r["PPL"] != "inf"]
                        if len(ppls) >= 2:
                            min_ppl = min(ppls)
                            max_ppl = max(ppls)
                            ratio = max_ppl / min_ppl if min_ppl > 0 else float("inf")
                            st.info(f"PPL 反差参考：当前有效句子中最高/最低 PPL 比值约为 `{ratio:.2f}` 倍。")
                    except Exception:
                        pass
