import streamlit as st

from services.translation_service import translate_en_to_zh
from theme import render_week9_intro

EXAMPLE_TEXTS = {
    "默认示例": "Machine translation helps people communicate across languages.",
    "俚语测试": "It rains cats and dogs.",
    "复杂长句": (
        "Although the project looked impossible at first, the team kept refining every module, "
        "and eventually delivered a reliable system that worked under real-world constraints."
    ),
}


def _apply_example_text() -> None:
    selected = st.session_state.get("m1_example_select", "默认示例")
    st.session_state["m1_input_text"] = EXAMPLE_TEXTS.get(selected, EXAMPLE_TEXTS["默认示例"])


def render_module1_nmt() -> None:
    st.title("Week9 机器翻译机制与质量评测系统")
    st.caption("模块1已实现：神经机器翻译（NMT）。")
    render_week9_intro()

    left, right = st.columns(2)
    with left:
        if "m1_input_text" not in st.session_state:
            st.session_state["m1_input_text"] = EXAMPLE_TEXTS["默认示例"]
        st.selectbox(
            "案例例句",
            list(EXAMPLE_TEXTS.keys()),
            key="m1_example_select",
            on_change=_apply_example_text,
            help="选择后会自动填充下方输入框。",
        )
        text = st.text_area("输入英文句子", key="m1_input_text", height=220)
        st.caption("提示：首次运行会下载模型，可能耗时较长。")
        import os as _os
        st.caption(
            "模型下载 endpoint："
            f"HF_ENDPOINT={_os.environ.get('HF_ENDPOINT','未设置')}；"
            f"HUGGINGFACE_HUB_BASE_URL={_os.environ.get('HUGGINGFACE_HUB_BASE_URL','未设置')}"
        )
        run = st.button("开始翻译", use_container_width=True)

    with right:
        st.markdown("**中文译文输出**")
        if run:
            with st.spinner("模型翻译中..."):
                result = translate_en_to_zh(text)
            if result.get("ok") == "true":
                st.markdown(
                    f'<div class="result-box">{result.get("translation", "")}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    (
                        f'<div class="small-note">模型：{result.get("model")} ｜ '
                        f'本次推理耗时：{result.get("elapsed_ms")} ms（若重复输入通常会更快）</div>'
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.error(result.get("error", "翻译失败。"))
                if result.get("detail"):
                    st.caption(f"错误详情：{result['detail']}")
        else:
            st.markdown(
                '<div class="result-box">请输入英文文本并点击“开始翻译”。</div>',
                unsafe_allow_html=True,
            )
