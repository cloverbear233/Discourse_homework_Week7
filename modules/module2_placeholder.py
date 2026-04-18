# -*- coding: utf-8 -*-
"""Module 2: train a character-level RNN language model."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - runtime dependency fallback
    torch = None
    nn = None

DEFAULT_CORPUS = (
    "To be, or not to be, that is the question.\n"
    "Whether tis nobler in the mind to suffer.\n"
    "The slings and arrows of outrageous fortune.\n"
)
CORPUS_CASES: list[dict[str, str]] = [
    {
        "label": "默认诗句（中等规律）",
        "text": DEFAULT_CORPUS,
    },
    {
        "label": "明显规律：hello world 循环",
        "text": ("hello world " * 30).strip() + "\n",
    },
    {
        "label": "明显规律：abcde 循环",
        "text": ("abcde " * 40).strip() + "\n",
    },
    {
        "label": "规律+变化：日期模板句",
        "text": (
            "today is monday. tomorrow is tuesday.\n"
            "today is monday. tomorrow is tuesday.\n"
            "today is monday. tomorrow is tuesday.\n"
        ),
    },
]


@dataclass
class RNNArtifacts:
    model: nn.Module
    stoi: dict[str, int]
    itos: dict[int, str]
    model_type: str
    hidden_size: int
    vocab_size: int


class CharRNNLM(nn.Module):
    """Simple char-level autoregressive LM with RNN/LSTM backend."""

    def __init__(self, vocab_size: int, hidden_size: int, model_type: str) -> None:
        super().__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if model_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for ch, idx in stoi.items()}
    return stoi, itos


def encode_text(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text if ch in stoi]


def create_batches(indices: list[int], seq_len: int) -> list[tuple[list[int], list[int]]]:
    pairs: list[tuple[list[int], list[int]]] = []
    for i in range(0, len(indices) - seq_len, seq_len):
        x = indices[i : i + seq_len]
        y = indices[i + 1 : i + seq_len + 1]
        if len(x) == seq_len and len(y) == seq_len:
            pairs.append((x, y))
    return pairs


def train_model(
    text: str,
    model_type: str,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    seq_len: int,
) -> tuple[RNNArtifacts, list[float]]:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch 不可用，请先安装 torch。")

    stoi, itos = build_vocab(text)
    indices = encode_text(text, stoi)
    if len(indices) < seq_len + 1:
        raise ValueError(f"语料过短，至少需要 {seq_len + 1} 个字符。")

    batches = create_batches(indices, seq_len=seq_len)
    if not batches:
        raise ValueError("无法构建训练批次，请增加语料长度。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CharRNNLM(vocab_size=len(stoi), hidden_size=hidden_size, model_type=model_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    progress = st.progress(0)
    status = st.empty()
    chart_holder = st.empty()

    loss_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in batches:
            x = torch.tensor(x_batch, dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(y_batch, dtype=torch.long, device=device).unsqueeze(0)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / len(batches)
        loss_history.append(avg_loss)
        progress.progress((epoch + 1) / epochs)
        status.markdown(f"**Epoch {epoch + 1}/{epochs}** | Loss: `{avg_loss:.6f}`")
        chart_holder.line_chart(loss_history)

    artifacts = RNNArtifacts(
        model=model,
        stoi=stoi,
        itos=itos,
        model_type=model_type,
        hidden_size=hidden_size,
        vocab_size=len(stoi),
    )
    return artifacts, loss_history


def generate_text(artifacts: RNNArtifacts, seed: str, gen_len: int) -> str:
    if torch is None:
        return ""
    if not seed:
        seed = " "

    model = artifacts.model
    stoi = artifacts.stoi
    itos = artifacts.itos
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 将 seed 中未知字符替换为空格或词表第一个字符
    fallback_char = " " if " " in stoi else itos[0]
    normalized_seed = "".join(ch if ch in stoi else fallback_char for ch in seed)

    model.eval()
    hidden = None
    generated = normalized_seed

    with torch.no_grad():
        for ch in normalized_seed[:-1]:
            x = torch.tensor([[stoi[ch]]], dtype=torch.long, device=device)
            _, hidden = model(x, hidden)

        last_char = normalized_seed[-1]
        for _ in range(gen_len):
            x = torch.tensor([[stoi[last_char]]], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_idx = int(torch.argmax(probs, dim=-1).item())
            next_char = itos[next_idx]
            generated += next_char
            last_char = next_char

    return generated


def render_module2_placeholder() -> None:
    st.subheader("模块2：从零训练 RNN 语言模型（字符级）")
    st.markdown("在本页输入短语料，训练一个 Char-level RNN/LSTM，并观察 Loss 曲线与生成结果。")

    if torch is None or nn is None:
        st.error("未检测到 PyTorch。请先安装 `torch` 后再使用本模块。")
        return

    with st.container(border=True):
        if "week7_rnn_corpus_input" not in st.session_state:
            st.session_state["week7_rnn_corpus_input"] = DEFAULT_CORPUS

        case_labels = [item["label"] for item in CORPUS_CASES]
        selected_label = st.selectbox("训练语料案例", options=case_labels, index=0)
        selected_case = CORPUS_CASES[case_labels.index(selected_label)]
        if st.button("使用该案例语料"):
            st.session_state["week7_rnn_corpus_input"] = selected_case["text"]

        corpus_text = st.text_area(
            "训练语料（建议英文短诗/名言）",
            key="week7_rnn_corpus_input",
            height=170,
            help="字符级训练，文本越有规律，loss 下降通常越明显。",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            model_type = st.selectbox("模型类型", options=["RNN", "LSTM"], index=0)
            hidden_size = st.slider("Hidden Size", min_value=16, max_value=128, value=64, step=16)
        with c2:
            epochs = st.slider("Epochs", min_value=10, max_value=200, value=60, step=10)
            seq_len = st.slider("Sequence Length", min_value=8, max_value=80, value=40, step=8)
        with c3:
            learning_rate = st.slider(
                "Learning Rate",
                min_value=0.001,
                max_value=0.050,
                value=0.010,
                step=0.001,
                format="%.3f",
            )

        train_clicked = st.button("开始训练", type="primary")
        if train_clicked:
            clean_text = corpus_text or ""
            if len(clean_text) < 20:
                st.warning("训练语料太短，建议至少 20 个字符。")
            else:
                try:
                    artifacts, loss_history = train_model(
                        text=clean_text,
                        model_type=model_type,
                        hidden_size=hidden_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        seq_len=seq_len,
                    )
                except Exception as exc:
                    st.error(f"训练失败：{exc}")
                else:
                    st.session_state["week7_rnn_artifacts"] = artifacts
                    st.session_state["week7_rnn_loss_history"] = loss_history
                    st.success("训练完成。你可以在下方输入 Seed 生成文本。")

    st.markdown("### 文本生成")
    artifacts: RNNArtifacts | None = st.session_state.get("week7_rnn_artifacts")
    if artifacts is None:
        st.info("请先点击“开始训练”，训练完成后会解锁生成。")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        seed = st.text_input("Seed（起始字符或字符串）", value="the ")
    with c2:
        gen_len = st.slider("生成长度", min_value=20, max_value=200, value=50, step=10)

    if st.button("生成文本"):
        try:
            generated = generate_text(artifacts=artifacts, seed=seed, gen_len=gen_len)
            st.code(generated)
        except Exception as exc:
            st.error(f"生成失败：{exc}")
