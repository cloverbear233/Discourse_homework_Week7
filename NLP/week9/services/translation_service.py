import os
import time
from typing import Dict

import streamlit as st

# 强制本实验走 PyTorch 路线，避免 transformers 在导入 TF 模块时触发
# “Keras 3 未被支持”的依赖冲突。
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

# 在导入 transformers/pipeline 之前就设置镜像，避免 hub 在导入时读取默认地址。
# 由于 huggingface_hub 在不同版本可能读取不同的环境变量名，这里做多字段兜底。
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
os.environ["HF_HUB_ENDPOINT"] = HF_MIRROR_ENDPOINT
os.environ["HUGGINGFACE_HUB_BASE_URL"] = HF_MIRROR_ENDPOINT

from transformers import pipeline  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"
MAX_INPUT_CHARS = 1000
LOCAL_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".hf_cache", "opus-mt-en-zh")
)


@st.cache_resource(show_spinner=False)
def get_translator():
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    # 先强制从 hf-mirror 拉取模型快照到本地目录，再从本地加载，
    # 从而避免直连 huggingface.co（例如 config.json 的 HEAD 超时）。
    try:
        # 如果之前已下载过，优先本地加载避免重复网络请求。
        local_snapshot_dir = snapshot_download(
            repo_id=MODEL_NAME,
            endpoint=HF_MIRROR_ENDPOINT,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            local_files_only=True,
        )
    except Exception:
        local_snapshot_dir = snapshot_download(
            repo_id=MODEL_NAME,
            endpoint=HF_MIRROR_ENDPOINT,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            etag_timeout=20,
        )
    # 显式指定框架为 PyTorch，进一步避免 TF Marian 组件被导入。
    return pipeline("translation", model=local_snapshot_dir, framework="pt")


@st.cache_data(show_spinner=False)
def cached_translate(text: str) -> Dict[str, str]:
    translator = get_translator()
    output = translator(text, max_length=256)
    translated_text = output[0]["translation_text"] if output else ""
    return {"translation": translated_text}


def translate_en_to_zh(text: str) -> Dict[str, str]:
    cleaned = text.strip()
    if not cleaned:
        return {"ok": "false", "error": "请输入英文句子后再进行翻译。"}
    if len(cleaned) > MAX_INPUT_CHARS:
        return {"ok": "false", "error": f"输入过长（>{MAX_INPUT_CHARS} 字符），请缩短后重试。"}

    t0 = time.perf_counter()
    try:
        result = cached_translate(cleaned)
    except Exception as exc:
        return {
            "ok": "false",
            "error": (
                "模型加载或翻译失败。请确认网络可用，并已在 base 环境安装 "
                "transformers、torch、sentencepiece 后重试。"
            ),
            "detail": str(exc),
        }

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    translation = result.get("translation", "").strip()
    if not translation:
        return {"ok": "false", "error": "模型未返回有效译文，请更换输入后重试。"}
    return {
        "ok": "true",
        "translation": translation,
        "model": MODEL_NAME,
        "elapsed_ms": str(elapsed_ms),
    }
