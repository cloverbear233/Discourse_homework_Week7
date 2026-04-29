import re
from typing import List

import streamlit as st

from services.translation_service import translate_en_to_zh
from collections import Counter
from math import exp, log


def _tokenize_zh(text: str, mode: str) -> List[str]:
    """
    BLEU 需要“离散 token 序列”。课堂演示建议：
    - 若文本包含空格，用空格分词，能更直观看到 n-gram 匹配效果；
    - 否则按中文字符分词。
    """
    t = text.strip()
    if not t:
        return []

    if mode == "space":
        if re.search(r"\s+", t):
            return [x for x in re.split(r"\s+", t) if x]
        # 没有空格就退回到字符分词
        return list(t.replace(" ", ""))

    # mode == "char"
    return list(t.replace(" ", ""))


def _ngrams(tokens: List[str], n: int) -> Counter:
    if n <= 0:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1)))


def _compute_bleu(candidate: str, reference: str, token_mode: str, max_order: int = 4) -> float:
    """
    手写 BLEU（等价于 n-gram precision + brevity penalty 的经典形式）。
    原本可直接用 nltk.translate.bleu_score.sentence_bleu，但你当前环境里
    nltk 与 fractions.Fraction 参数兼容性存在问题，因此这里改为稳定实现。
    """
    cand_tokens = _tokenize_zh(candidate, token_mode)
    ref_tokens = _tokenize_zh(reference, token_mode)
    if not cand_tokens or not ref_tokens:
        return 0.0

    cand_len = len(cand_tokens)
    ref_len = len(ref_tokens)
    effective_order = min(max_order, cand_len, ref_len)
    if effective_order <= 0:
        return 0.0

    # brevity penalty
    bp = 1.0 if cand_len > ref_len else exp(1.0 - (ref_len / max(1, cand_len)))

    precisions: List[float] = []
    for n in range(1, effective_order + 1):
        cand_ngrams = _ngrams(cand_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)

        total = sum(cand_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue

        # clipped matches
        matches = 0
        for ng, cnt in cand_ngrams.items():
            matches += min(cnt, ref_ngrams.get(ng, 0))

        # 简单平滑：避免 log(0) 或 precision=0 导致 BLEU=0
        # 当 matches==0 时，用 add-one 平滑近似 method1 的效果。
        if matches == 0:
            precisions.append((matches + 1) / (total + 1))
        else:
            precisions.append(matches / total)

    # weights: uniform 1/max_order
    weights = 1.0 / effective_order
    # 使用平滑后的 precisions，仍保持“基于 n-gram 匹配”的核心含义
    score_log_sum = 0.0
    for p in precisions:
        # p 可能极小，仍用 log；若 p==0 则直接返回 0（除非已平滑）
        if p <= 0:
            return 0.0
        score_log_sum += weights * log(p)

    bleu = bp * exp(score_log_sum)
    return float(bleu)


def render_module3_placeholder() -> None:
    st.header("模块3：机器翻译质量自动评测（BLEU Score）")
    st.info("输入英文原文、参考译文（Reference）与候选译文（Candidate），自动计算 BLEU 并解释其含义。")

    # 快速案例：自动填充评测输入
    cases = {
        "默认示例": {
            "source_en": "Machine translation helps people communicate across languages.",
            "reference_zh": "机器翻译 帮助 人们 跨越 语言 交流",
            "candidate_zh": "机器翻译 帮助 人们 跨越 语言 交流",
        },
        "参考A：同词不同序": {
            "source_en": "Machine translation helps people communicate across languages.",
            "reference_zh": "交流 语言 跨越 人们 帮助 机器翻译",
            "candidate_zh": "机器翻译 帮助 人们 跨越 语言 交流",
        },
        "参考B：同序不同义（换词）": {
            "source_en": "Machine translation helps people communicate across languages.",
            "reference_zh": "机器翻译 促进 人们 穿越 语言 交流",
            "candidate_zh": "机器翻译 帮助 人们 跨越 语言 交流",
        },
        "几乎不重叠（低分）": {
            "source_en": "Machine translation helps people communicate across languages.",
            "reference_zh": "天气 很 好 今天 看 日出",
            "candidate_zh": "机器翻译 帮助 人们 跨越 语言 交流",
        },
        "长度更长的参考": {
            "source_en": "Machine translation helps people communicate across languages.",
            "reference_zh": "机器翻译 帮助 人们 跨越 语言 交流 并 支持 多轮 对话",
            "candidate_zh": "机器翻译 帮助 人们 跨越 语言 交流",
        },
    }

    case_names = list(cases.keys())
    selected_case_name = st.selectbox(
        "快速案例（自动填充评测输入）",
        case_names,
        index=0,
        key="m3_case_select",
    )
    selected_case = cases.get(selected_case_name, cases["默认示例"])

    # 仅在案例发生切换时才写入输入框对应的 session_state，避免覆盖用户“从模块1生成 Candidate”的结果。
    if st.session_state.get("m3_case_applied") != selected_case_name:
        st.session_state["m3_source_en"] = selected_case["source_en"]
        st.session_state["m3_reference_zh"] = selected_case["reference_zh"]
        st.session_state["m3_candidate_zh"] = selected_case["candidate_zh"]
        st.session_state["m3_case_applied"] = selected_case_name

    # 基础输入
    st.subheader("评测输入")
    left, mid, right = st.columns(3)
    with left:
        source_en = st.text_area(
            "1) 待翻译英文原文",
            value=st.session_state.get("m3_source_en", cases["默认示例"]["source_en"]),
            height=130,
            key="m3_source_en",
        )
    with mid:
        reference_zh = st.text_area(
            "2) 标准中文参考译文（Reference）",
            value=st.session_state.get(
                "m3_reference_zh", cases["默认示例"]["reference_zh"]
            ),
            height=130,
            key="m3_reference_zh",
        )
        token_mode = st.selectbox("分词方式（影响BLEU）", ["space", "char"], index=0, help="space适合课堂示例（有空格）。char适合无空格中文。")

    # 如果用户首次进入且尚未触发案例写入，这里做一个兜底初始化。
    if "m3_candidate_zh" not in st.session_state:
        st.session_state["m3_candidate_zh"] = cases["默认示例"]["candidate_zh"]

    with right:
        gen = st.button("从模块1生成 Candidate（NMT）", use_container_width=True)

        # 注意：必须在下面 text_area 实例化之前更新 session_state，否则会触发 Streamlit API 异常。
        if gen:
            with st.spinner("模型翻译中..."):
                nmt_result = translate_en_to_zh(source_en)
            if nmt_result.get("ok") == "true":
                st.session_state["m3_candidate_zh"] = nmt_result.get("translation", "")
                st.success("已更新 Candidate。")
            else:
                st.error(nmt_result.get("error", "NMT 生成失败。"))
                if nmt_result.get("detail"):
                    st.caption(f"错误详情：{nmt_result['detail']}")

        candidate_zh = st.text_area(
            "3) 机器生成候选译文（Candidate）",
            height=130,
            key="m3_candidate_zh",
        )

    # 计算 BLEU
    calc = st.button("计算 BLEU", use_container_width=True)
    if calc:
        bleu = _compute_bleu(
            candidate=candidate_zh,
            reference=reference_zh,
            token_mode=token_mode,
        )
        st.metric("BLEU（0~1，越高越好）", f"{bleu:.4f}")

        with st.expander("这个分数代表什么？"):
            st.markdown(
                "BLEU 主要衡量 **候选译文与参考译文的 n-gram 重合程度**（并带有简短惩罚）。"
            )
            st.markdown("- 若候选译文在相同位置/局部搭配上与参考译文片段匹配，n-gram precision 会更高。")
            st.markdown("- 若出现同义替换但 n-gram 不同，BLEU 往往下降（即：语义相近但词不相同可能得分不高）。")
            st.markdown("- 若语序错乱，即使词汇集合相同，n-gram 顺序错位也会导致匹配显著减少。")

    # 课堂观察：多场景 BLEU 对比
    with st.expander("课堂观测：多参考多场景 BLEU（一键汇总）", expanded=True):
        st.markdown("同一句英文，我们人为构造多种“参考译文”，观察 BLEU 在不同失配类型下如何变化。")

        # 固定候选译文 token（便于演示 BLEU 的局限）
        candidate_demo = "机器翻译 帮助 人们 跨越 语言 交流"

        # 参考集合：覆盖“语序错乱 / 同义替换 / 部分重叠 / 几乎不重叠 / 长度不匹配 / 标记化敏感”
        refs = {
            "参考1 完全一致": candidate_demo,
            "参考2 同词不同序（语序错位）": "交流 语言 跨越 人们 帮助 机器翻译",
            "参考3 同序不同义（token替换）": "机器翻译 促进 人们 穿越 语言 交流",
            "参考4 部分重叠（短一些但有匹配片段）": "机器翻译 帮助 语言 交流",
            "参考5 几乎不重叠（完全换主题）": "天气 很 好 今天 看 日出",
            "参考6 参考更长（长度惩罚/精度下降）": "机器翻译 帮助 人们 跨越 语言 交流 并 支持 多轮 对话",
            "参考7 参考更短（brevity penalty 更明显）": "机器翻译 帮助 人们 交流",
            "参考8 标点敏感（在space模式下 token可能不匹配）": "机器翻译 帮助 人们 跨越 语言 交流。",
        }

        demo_mode = st.selectbox(
            "观测用分词方式（影响 token 序列）",
            ["space", "char"],
            index=0,
            key="m3_demo_token_mode",
        )

        if st.button("计算多种 BLEU 并展示结果表", use_container_width=True):
            rows = []
            for ref_name, ref_text in refs.items():
                bleu = _compute_bleu(candidate_demo, ref_text, demo_mode)
                rows.append(
                    {
                        "场景": ref_name,
                        "BLEU": f"{bleu:.4f}",
                        "Reference": ref_text,
                    }
                )

            st.table(rows)

            with st.expander("如何解读这些差异？（简明版）"):
                st.markdown("- 完全一致时 BLEU 应最高。")
                st.markdown("- 同词不同序会让 n-gram 顺序匹配显著减少，BLEU 往往下降。")
                st.markdown("- 同序不同义会因为 token 不同而导致重合度下降。")
                st.markdown("- 部分重叠通常能保留较低阶 n-gram 的匹配，但高阶会迅速变差。")
                st.markdown("- 长度不匹配触发 brevity penalty，同时也会影响 n-gram precision。")
                st.markdown("- 标点敏感场景提醒：tokenization/分词策略会直接影响 BLEU。")

