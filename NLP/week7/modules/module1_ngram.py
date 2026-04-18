# -*- coding: utf-8 -*-
"""Module 1: n-gram language model and Laplace smoothing demo."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import nltk
import streamlit as st
from nltk.corpus import reuters

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")
FALLBACK_TEXT = (
    "Language models learn token patterns from text. "
    "A trigram model estimates a word using two previous words. "
    "Smoothing helps assign probability to unseen events."
)
DEFAULT_QUERY = "language models can explain unseen events"
CASE_SENTENCES: list[dict[str, str]] = [
    {
        "label": "高频短句（应为已见 trigram）",
        "text": "the company said",
    },
    {
        "label": "高频短句（应为已见 trigram）",
        "text": "it also said",
    },
    {
        "label": "高频短句（应为已见 trigram）",
        "text": "in the market",
    },
    {
        "label": "正常新闻句（较常见表达）",
        "text": "the company said it expects higher sales",
    },
    {
        "label": "零概率示例（罕见 Trigram）",
        "text": "the quantum pineapple negotiates with martian bankers",
    },
    {
        "label": "零概率示例（科幻组合）",
        "text": "ancient robots brew coffee on the moon",
    },
]


@st.cache_data(show_spinner=False)
def build_ngram_counts(tokens: list[str], n: int) -> tuple[Counter[tuple[str, ...]], Counter[tuple[str, ...]], set[str]]:
    """Build n-gram and context counters."""
    grams: Counter[tuple[str, ...]] = Counter()
    contexts: Counter[tuple[str, ...]] = Counter()

    if len(tokens) < n:
        return grams, contexts, set(tokens)

    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        context = gram[:-1]
        grams[gram] += 1
        contexts[context] += 1

    return grams, contexts, set(tokens)


def normalize_text(text: str) -> list[str]:
    """Simple English tokenization and cleanup."""
    lowered = text.lower()
    return TOKEN_PATTERN.findall(lowered)


def is_reuters_available() -> bool:
    """Check whether Reuters corpus is available locally."""
    try:
        nltk.data.find("corpora/reuters.zip")
        return True
    except LookupError:
        try:
            nltk.data.find("corpora/reuters")
            return True
        except LookupError:
            return False


def try_download_reuters() -> bool:
    """Download Reuters corpus via NLTK downloader."""
    try:
        return bool(nltk.download("reuters", quiet=True))
    except Exception:
        return False


def get_reuters_tokens(limit_docs: int = 300) -> list[str]:
    """Load Reuters tokens with a doc limit for responsive demo."""
    file_ids = reuters.fileids()[:limit_docs]
    words: list[str] = []
    for file_id in file_ids:
        words.extend(reuters.words(file_id))
    return [w.lower() for w in words if TOKEN_PATTERN.fullmatch(w.lower())]


def calc_joint_probability(
    query_tokens: list[str],
    n: int,
    ngram_counts: Counter[tuple[str, ...]],
    context_counts: Counter[tuple[str, ...]],
    vocab_size: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Return unsmoothed/smoothed probability and step metrics."""
    if len(query_tokens) < n:
        return 0.0, 0.0, []

    prob_raw = 1.0
    prob_smooth = 1.0
    steps: list[dict[str, Any]] = []

    for i in range(len(query_tokens) - n + 1):
        gram = tuple(query_tokens[i : i + n])
        context = gram[:-1]
        target = gram[-1]

        count_ngram = ngram_counts.get(gram, 0)
        count_context = context_counts.get(context, 0)

        raw = 0.0
        if count_context > 0:
            raw = count_ngram / count_context

        smooth = (count_ngram + 1) / (count_context + vocab_size)

        prob_raw *= raw
        prob_smooth *= smooth

        steps.append(
            {
                "context": " ".join(context),
                "target": target,
                "ngram": " ".join(gram),
                "count_ngram": count_ngram,
                "count_context": count_context,
                "prob_raw": raw,
                "prob_smooth": smooth,
                "is_unseen": count_ngram == 0,
            }
        )

    return prob_raw, prob_smooth, steps


def scientific(prob: float) -> str:
    """Format probabilities in scientific notation."""
    if prob <= 0.0:
        return "0"
    return f"{prob:.6e}"


def render_module1_ngram() -> None:
    """Render module 1 UI."""
    st.subheader("模块1：n 元语言模型与数据平滑")
    st.markdown(
        "默认优先使用 **NLTK Reuters** 语料；若本机未安装，可一键下载，或切到手动文本模式继续实验。"
    )

    left, right = st.columns([1, 1])

    with left:
        corpus_mode = st.radio(
            "语料来源",
            options=["NLTK Reuters（优先）", "手动输入文本"],
            index=0,
            horizontal=True,
        )

    with right:
        n = st.selectbox("n 值", options=[2, 3], index=1)

    reuters_ok = is_reuters_available()

    if corpus_mode == "NLTK Reuters（优先）":
        if not reuters_ok:
            st.warning("当前未检测到 Reuters 语料。你可以点击下方按钮下载，或切换到手动输入文本模式。")
            if st.button("下载 Reuters 语料", type="primary"):
                with st.spinner("正在下载 Reuters，请稍候..."):
                    ok = try_download_reuters()
                if ok and is_reuters_available():
                    st.success("Reuters 下载完成。")
                    reuters_ok = True
                else:
                    st.error("下载失败，请检查网络后重试；你仍可使用手动文本模式。")

        if reuters_ok:
            with st.spinner("正在加载 Reuters 语料并构建词表..."):
                corpus_tokens = get_reuters_tokens(limit_docs=300)
            st.info(f"已加载 Reuters 子集（前 300 篇），可用 token 数：{len(corpus_tokens)}")
        else:
            corpus_tokens = normalize_text(FALLBACK_TEXT)
            st.caption("当前使用内置兜底文本，仅用于不中断课堂演示。")
    else:
        manual_text = st.text_area(
            "手动输入英文语料",
            value=FALLBACK_TEXT,
            height=140,
            help="建议输入 3-8 句英文文本，以便观察 n-gram 统计特征。",
        )
        corpus_tokens = normalize_text(manual_text)
        st.info(f"手动语料 token 数：{len(corpus_tokens)}")

    if len(corpus_tokens) < n:
        st.error(f"当前语料 token 数不足以构建 {n}-gram（至少需要 {n} 个 token）。")
        return

    ngram_counts, context_counts, vocab = build_ngram_counts(corpus_tokens, n)
    vocab_size = len(vocab)

    st.markdown("---")
    st.markdown("### 案例句子（含零概率示例）")

    case_labels = [f"{item['label']}：{item['text']}" for item in CASE_SENTENCES]
    selected_case = st.selectbox("选择案例句子", options=case_labels, index=2)
    selected_text = CASE_SENTENCES[case_labels.index(selected_case)]["text"]
    if st.button("将案例句子填入输入框"):
        st.session_state["week7_query_input"] = selected_text

    case_rows: list[dict[str, str]] = []
    for item in CASE_SENTENCES:
        tokens = normalize_text(item["text"])
        if len(tokens) < n:
            status = "过短"
        else:
            _, _, case_steps = calc_joint_probability(
                query_tokens=tokens,
                n=n,
                ngram_counts=ngram_counts,
                context_counts=context_counts,
                vocab_size=vocab_size,
            )
            has_unseen = any(step["is_unseen"] for step in case_steps)
            status = "是（含未见 n-gram）" if has_unseen else "否"
        case_rows.append({"案例": item["text"], "未见 n-gram": status})
    st.dataframe(case_rows, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 句子概率计算")

    if "week7_query_input" not in st.session_state:
        st.session_state["week7_query_input"] = DEFAULT_QUERY

    query = st.text_input(
        "输入英文句子（用于计算联合概率）",
        key="week7_query_input",
    )
    enable_smoothing = st.checkbox("启用 Add-one (Laplace) 平滑", value=False)

    query_tokens = normalize_text(query)
    if not query_tokens:
        st.warning("请输入至少一个英文词。")
        return

    if len(query_tokens) < n:
        st.warning(f"输入句子过短：{n}-gram 需要至少 {n} 个 token。")
        return

    prob_raw, prob_smooth, steps = calc_joint_probability(
        query_tokens=query_tokens,
        n=n,
        ngram_counts=ngram_counts,
        context_counts=context_counts,
        vocab_size=vocab_size,
    )

    unseen_steps = [s for s in steps if s["is_unseen"]]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("词表大小 |V|", vocab_size)
    with c2:
        st.metric("未平滑联合概率", scientific(prob_raw))
    with c3:
        display_prob = prob_smooth if enable_smoothing else prob_raw
        label = "当前生效概率（平滑）" if enable_smoothing else "当前生效概率（未平滑）"
        st.metric(label, scientific(display_prob))

    st.markdown(
        rf"""
- 未平滑条件概率：$P(w_i|h)=\frac{{C(h,w_i)}}{{C(h)}}$
- Add-one 条件概率：$P_{{add1}}(w_i|h)=\frac{{C(h,w_i)+1}}{{C(h)+|V|}}$
- 联合概率通过所有步骤条件概率连乘得到。
"""
    )

    if unseen_steps:
        st.warning("检测到未见 n-gram，未平滑概率会受到零概率事件影响。")
        st.write("导致零概率/稀疏问题的片段：")
        for s in unseen_steps:
            st.code(f"{s['ngram']}  -> C(ngram)=0, C(context)={s['count_context']}")
    else:
        st.success("当前输入句子的每个 n-gram 在语料中都出现过。")

    st.markdown("### 步骤级概率分解")
    step_rows: list[dict[str, Any]] = []
    for idx, s in enumerate(steps, start=1):
        row = {
            "step": idx,
            "context": s["context"],
            "target": s["target"],
            "ngram": s["ngram"],
            "C(h,w)": s["count_ngram"],
            "C(h)": s["count_context"],
            "P_raw": f"{s['prob_raw']:.6e}",
            "P_add1": f"{s['prob_smooth']:.6e}",
            "未见事件": "是" if s["is_unseen"] else "否",
        }
        step_rows.append(row)

    st.dataframe(step_rows, use_container_width=True)

    st.caption(
        "说明：即使未勾选平滑，表格仍展示 P_add1 供对比观察；勾选后“当前生效概率”切换为平滑结果。"
    )
