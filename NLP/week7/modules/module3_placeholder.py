# -*- coding: utf-8 -*-
"""Module 3: pretrained architecture comparison (BERT vs GPT-2)."""

from __future__ import annotations

import os

import streamlit as st

# Force Transformers to avoid TensorFlow/Keras code path in this project.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - runtime dependency fallback
    pipeline = None

DEFAULT_MASKED = "The man went to the [MASK] to buy some milk."
DEFAULT_PROMPT = "In the future, language models will"


@st.cache_resource(show_spinner=False)
def get_fill_mask_pipeline():
    if pipeline is None:
        raise RuntimeError("transformers 不可用，请先安装依赖。")
    return pipeline("fill-mask", model="bert-base-uncased", framework="pt")


@st.cache_resource(show_spinner=False)
def get_gpt2_pipeline():
    if pipeline is None:
        raise RuntimeError("transformers 不可用，请先安装依赖。")
    return pipeline("text-generation", model="gpt2", framework="pt")


def _truncate_to_words(text: str, n_words: int) -> str:
    words = text.split()
    return " ".join(words[:n_words])


def render_module3_placeholder() -> None:
    st.subheader("模块3：预训练架构对比（Masked LM vs. Causal LM）")
    st.markdown("左侧使用 **BERT** 做 `[MASK]` 预测，右侧使用 **GPT-2** 做自回归续写。")

    if pipeline is None:
        st.error("未检测到 transformers。请先安装依赖后再使用本模块。")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### BERT · Masked Language Modeling")
        masked_input = st.text_input(
            "输入带 [MASK] 的句子",
            value=DEFAULT_MASKED,
            key="week7_m3_mask_input",
        )

        if st.button("BERT Top-5 预测", key="week7_m3_run_bert", type="primary"):
            if "[MASK]" not in masked_input:
                st.warning("请在句子中包含 `[MASK]` 标记。")
            else:
                try:
                    fill_mask = get_fill_mask_pipeline()
                    preds = fill_mask(masked_input, top_k=5)
                except Exception as exc:
                    st.error(f"BERT 推理失败：{exc}")
                else:
                    rows = []
                    for item in preds:
                        rows.append(
                            {
                                "token": item.get("token_str", "").strip(),
                                "probability": f"{float(item.get('score', 0.0)):.4f}",
                                "sequence": item.get("sequence", ""),
                            }
                        )
                    st.dataframe(rows, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### GPT-2 · Causal Language Modeling")
        prompt = st.text_input(
            "输入前缀 Prompt",
            value=DEFAULT_PROMPT,
            key="week7_m3_prompt_input",
        )

        if st.button("GPT-2 生成后续（约20词）", key="week7_m3_run_gpt2", type="primary"):
            if not prompt.strip():
                st.warning("请输入非空 Prompt。")
            else:
                try:
                    gpt2_gen = get_gpt2_pipeline()
                    outputs = gpt2_gen(
                        prompt,
                        max_new_tokens=30,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1,
                    )
                    full_text = outputs[0]["generated_text"]
                except Exception as exc:
                    st.error(f"GPT-2 推理失败：{exc}")
                else:
                    continuation = full_text[len(prompt) :] if full_text.startswith(prompt) else full_text
                    continuation_20 = _truncate_to_words(continuation.strip(), 20)
                    st.write("**生成结果（完整）**")
                    st.code(full_text)
                    st.write("**后续约20词（便于对比）**")
                    st.code(continuation_20)
