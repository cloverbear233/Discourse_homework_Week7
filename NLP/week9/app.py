import streamlit as st

from modules.module1_nmt import render_module1_nmt
from modules.module2_placeholder import render_module2_placeholder
from modules.module3_placeholder import render_module3_placeholder
from theme import inject_week9_theme, render_footer_attribution


st.set_page_config(page_title="Week9 机器翻译对比与评测系统", layout="wide", initial_sidebar_state="expanded")


def main() -> None:
    inject_week9_theme()

    st.sidebar.title("模块导航")
    tab = st.sidebar.radio(
        "请选择模块",
        ["模块1：神经机器翻译", "模块2：BLEU 质量评测", "模块3：翻译机制对比"],
        label_visibility="collapsed",
    )

    if tab.startswith("模块1"):
        render_module1_nmt()
    elif tab.startswith("模块2"):
        render_module2_placeholder()
    else:
        render_module3_placeholder()

    render_footer_attribution()


if __name__ == "__main__":
    main()
