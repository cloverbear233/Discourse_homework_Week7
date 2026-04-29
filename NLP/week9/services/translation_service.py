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


_WEIGHT_FILES = {
    "pytorch_model.bin",
    "model.safetensors",
    "tf_model.h5",
    "tf_model.h5.index",
    "model.ckpt.index",
    "flax_model.msgpack",
}


def _local_model_has_weights(model_dir: str) -> bool:
    """
    transformers 从本地目录加载时，除了 config/tokenizer/vocab，
    还需要真正的权重文件（pytorch_model.bin / model.safetensors 等）。
    """
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        return False

    # 先快速检查一层目录
    for name in _WEIGHT_FILES:
        if os.path.exists(os.path.join(model_dir, name)):
            return True

    # 再递归查找（snapshot_download 可能把文件放在子目录）
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f in _WEIGHT_FILES:
                return True
    return False


@st.cache_resource(show_spinner=False)
def get_translator():
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    # 1) 如果本地目录已经有权重，直接加载，避免任何网络请求
    if _local_model_has_weights(LOCAL_MODEL_DIR):
        return pipeline("translation", model=LOCAL_MODEL_DIR, framework="pt")

    # 2) 本地目录缺权重：尝试从 hf-mirror 下载补齐
    try:
        snapshot_download(
            repo_id=MODEL_NAME,
            endpoint=HF_MIRROR_ENDPOINT,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            etag_timeout=20,
        )
    except Exception:
        # 3) 作为兜底：尝试使用 transformers 默认缓存（~/.cache/huggingface）
        #    只做本地加载（不再发起网络请求）。
        try:
            return pipeline(
                "translation",
                model=MODEL_NAME,
                framework="pt",
                model_kwargs={"local_files_only": True},
            )
        except Exception as exc:
            raise RuntimeError(
                "模型权重缺失，且无法从 hf-mirror 下载到本地。"
                "请确认 hf-mirror 网络可用，或将模型权重预先下载到本地缓存。"
            ) from exc

    # 下载后仍未发现权重文件，说明 mirror 仍未成功下载到完整权重
    if not _local_model_has_weights(LOCAL_MODEL_DIR):
        raise RuntimeError(
            "模型权重仍未在本地目录找到（仅检测到 config/tokenizer/vocab 等小文件）。"
            "请确认网络可用并重试；必要时删除本目录后重新下载。"
        )

    return pipeline("translation", model=LOCAL_MODEL_DIR, framework="pt")


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
