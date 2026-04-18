# -*- coding: utf-8 -*-
"""Week7 language modeling training and comparison platform."""

from __future__ import annotations

import streamlit as st

from modules.module1_ngram import render_module1_ngram
from modules.module2_placeholder import render_module2_placeholder
from modules.module3_placeholder import render_module3_placeholder
from modules.module4_placeholder import render_module4_placeholder
from theme import inject_week7_theme, render_footer_attribution, render_week7_intro


def main() -> None:
    st.set_page_config(
        page_title="Week7 · 语言模型训练与对比分析平台",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_week7_theme()

    st.title("Week 7 · 语言模型训练与对比分析平台")
    st.caption("模块1-4可用（统计模型 / 自训练RNN / 预训练模型 / PPL评估）")
    render_week7_intro()

    st.sidebar.title("模块导航")
    nav = st.sidebar.radio(
        "选择模块",
        options=["m1", "m2", "m3", "m4"],
        format_func=lambda k: {
            "m1": "模块1：n-gram & Smoothing",
            "m2": "模块2：从零训练 RNN 语言模型",
            "m3": "模块3：预训练架构对比（BERT vs GPT-2）",
            "m4": "模块4：语言模型评价（Perplexity）",
        }[k],
        index=0,
        key="week7_sidebar_nav",
        label_visibility="collapsed",
    )

    if nav == "m1":
        render_module1_ngram()
    elif nav == "m2":
        render_module2_placeholder()
    elif nav == "m3":
        render_module3_placeholder()
    else:
        render_module4_placeholder()

    render_footer_attribution()


if __name__ == "__main__":
    main()
