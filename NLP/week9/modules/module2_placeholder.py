import streamlit as st

import re
from typing import Dict, List, Tuple

from services.translation_service import translate_en_to_zh


# 一个极简“英汉词典”，用于模拟早期基于词典替换的逐词直译。
# 这里刻意覆盖课堂常见现象：俚语/固定搭配/少量功能词。
EN_ZH_DICT: Dict[str, str] = {
    "it": "它",
    "rains": "下雨",
    "rain": "下雨",
    "cats": "猫",
    "cat": "猫",
    "dogs": "狗",
    "dog": "狗",
    "and": "和",
    "helps": "帮助",
    "help": "帮助",
    "people": "人们",
    "communicate": "交流",
    "across": "跨越",
    "languages": "语言",
    "language": "语言",
    "machine": "机器",
    "translation": "翻译",
    "helps": "帮助",
    "across": "横跨",
    "the": "",
    "a": "",
    "an": "",
    "project": "项目",
    "looked": "看起来",
    "impossible": "不可能的",
    "at": "在",
    "first": "起初",
    "team": "团队",
    "kept": "继续",
    "refining": "改进",
    "every": "每一",
    "module": "模块",
    "eventually": "最终",
    "delivered": "交付",
    "reliable": "可靠的",
    "system": "系统",
    "worked": "运作",
    "real-world": "真实世界的",
    "constraints": "约束",

    # 相对从句/倒装相关（用于展示“语序鸿沟”）
    "book": "书",
    "that": "那个",
    "you": "你",
    "gave": "给了",
    "me": "我",
    "is": "是",
    "interesting": "有趣的",
    "only": "只有",
    "then": "那时",
    "did": "",
    "he": "他",
    "realize": "意识到",
    "realized": "意识到",
    "mistake": "错误",

    # 一词多义（bank：河岸 vs 金融银行）
    "i": "我",
    "sat": "坐",
    "by": "在…旁边",
    "river": "河",
    "went": "去",
    "to": "到",
    "withdraw": "取款",
    "money": "钱",
    "bank": "银行",
}


EXAMPLE_TEXTS: Dict[str, str] = {
    "默认示例": "Machine translation helps people communicate across languages.",
    "俚语测试（cats and dogs）": "It rains cats and dogs.",
    "复杂长句（词典局限）": (
        "Although the project looked impossible at first, the team kept refining every module, "
        "and eventually delivered a reliable system that worked under real-world constraints."
    ),
    "定语从句（语序鸿沟）": "The book that you gave me is interesting.",
    "倒装（Only then ...）": "Only then did he realize the mistake.",
    "一词多义 bank：河岸": "I sat by the bank of the river.",
    "一词多义 bank：金融银行": "I went to the bank to withdraw money.",
}


def _split_core_punct(token: str) -> Tuple[str, str]:
    """
    将 token 拆成“核心词 + 尾部标点”，用于逐词替换时保留如 dogs. / hello, 里的标点。
    """
    m = re.match(r"^(.*?)([.,!?;:]+)?$", token)
    if not m:
        return token, ""
    core = m.group(1) or ""
    punct = m.group(2) or ""
    return core, punct


def rule_based_word_by_word_translate(text: str, dictionary: Dict[str, str]) -> str:
    """
    模拟早期基于“词典直接替换”的机器翻译：
    - 用空格分词
    - 词典命中则替换，否则保留原英文词
    - 维持尾部标点
    """
    cleaned = text.strip()
    if not cleaned:
        return ""

    tokens: List[str] = cleaned.split()
    zh_tokens: List[str] = []
    for tok in tokens:
        core, punct = _split_core_punct(tok)
        key = core.lower()
        mapped = dictionary.get(key)
        if mapped is None:
            mapped = core
        zh_tokens.append(f"{mapped}{punct}".strip())

    out = " ".join(zh_tokens).strip()
    # 去掉“空格 + 标点”，例如 “狗 .” -> “狗.”
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return out


def render_module2_placeholder() -> None:
    st.header("模块2：基于规则的直译 vs. 神经网络意译")
    st.info("本模块模拟早期“基于词典逐词替换”的机器翻译，并将其与模块1的 NMT 译文并排对比。")

    # 输入区
    st.caption(f"词典大小：{len(EN_ZH_DICT)}（简化版，用于课堂演示）")

    left, _ = st.columns([1, 1])  # 只是为了对齐布局（右侧留空占位）
    with left:
        st.selectbox(
            "案例例句",
            list(EXAMPLE_TEXTS.keys()),
            key="m2_example_select",
            help="选择后将自动填充输入文本。",
        )

    if "m2_input_text" not in st.session_state:
        st.session_state["m2_input_text"] = EXAMPLE_TEXTS["默认示例"]

    # selectbox 变更时同步输入框内容
    selected = st.session_state.get("m2_example_select", "默认示例")
    st.session_state["m2_input_text"] = EXAMPLE_TEXTS.get(selected, EXAMPLE_TEXTS["默认示例"])

    text = st.text_area("输入英文句子（建议包含俚语/固定搭配）", value=st.session_state["m2_input_text"], height=180)
    st.session_state["m2_input_text"] = text

    run = st.button("开始对比", use_container_width=True)
    if not run:
        st.markdown('<div class="result-box">点击“开始对比”后，将在下方并排展示 NMT 与逐词直译结果。</div>', unsafe_allow_html=True)
        return

    # 输出对比区
    with st.spinner("NMT 翻译中..."):
        nmt_result = translate_en_to_zh(text)
        nmt_translation = nmt_result.get("translation", "")

    rule_translation = rule_based_word_by_word_translate(text, EN_ZH_DICT)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### NMT（模块1）译文")
        if nmt_result.get("ok") == "true" and nmt_translation:
            st.markdown(f'<div class="result-box">{nmt_translation}</div>', unsafe_allow_html=True)
            st.caption(
                f"模型：{nmt_result.get('model')} ｜耗时：{nmt_result.get('elapsed_ms')} ms"
            )
        else:
            st.error(nmt_result.get("error", "NMT 翻译失败。"))
            if nmt_result.get("detail"):
                st.caption(f"错误详情：{nmt_result['detail']}")

    with c2:
        st.markdown("### 逐词直译（规则引擎模拟）")
        if rule_translation:
            st.markdown(f'<div class="result-box">{rule_translation}</div>', unsafe_allow_html=True)
        else:
            st.warning("规则引擎未生成译文（请确认输入非空）。")

        with st.expander("为什么“逐词直译”会翻得不理想？"):
            st.markdown("- 词序与语法结构不转换")
            st.markdown("- 定语从句/倒装需要结构重组，逐词直译无法完成")
            st.markdown("- 一词多义依赖上下文消歧，词典替换通常会“同义错误”")
            st.markdown("- 俚语与固定搭配可能无法被词典覆盖")

