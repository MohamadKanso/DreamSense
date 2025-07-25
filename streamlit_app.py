"""streamlit_app.py — DreamSense (v2)
Adds four enhancements requested:
1. Spinners around expensive API calls (better user feedback)
2. Download buttons for poem, image & audio
3. Poem displayed in an expander with CSS *pre‑wrap* to avoid scroll pain
4. Sidebar radio toggle for **Poem / Image + Poem / Melody (beta)**

Everything else (prompt templates, request code, TTS) remains identical, so no breaking behavioural changes.
"""

from __future__ import annotations

import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import requests
from gtts import gTTS
from PIL import Image
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_TEXT = "HuggingFaceH4/zephyr-7b-beta"
MODEL_IMAGE = "stabilityai/stable-diffusion-xl-base-1.0"
N_FOLLOWUPS = 3
TIMEOUT = 120  # seconds for external calls

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── CONFIG (snippet) ───────────────────────────────────────────────

FOLLOWUP_PROMPT = """You are an insightful, concise dream analyst.
Ask {n} SHORT follow‑up questions to better understand this dream.
No preamble, just numbered questions on separate lines.

Dream:
\"\"\"{dream}\"\"\""""

POEM_PROMPT = """You are a poetic interpreter of dreams.
Using the dream and answers, write a flowing poetic prose (150–250 words).
No bullet points or headers—just a single evocative passage.

Context:
\"\"\"{context}\"\"\""""

IMAGE_PROMPT_TMPL = """Write a single <120 word prompt for an image generator
capturing the symbolism, mood and key elements of this dream context:

\"\"\"{context}\"\"\""""
# ────────────────────────────────────────────────────────────────────────────────

# Feel free to swap in your own music‑gen endpoint later
MELODY_PLACEHOLDER_MSG = (
    "🎵 Melody generation is still in beta. We'll add sound soon!"
)

# OpenAI‑style Hugging Face router client (unchanged from original)
from openai import OpenAI  # noqa: E402  # (import after config to respect HF_TOKEN)

client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)

# ────────────────────────────────────────────────────────────────────────────────
# BACKEND HELPERS
# ────────────────────────────────────────────────────────────────────────────────

def hf_text(
    prompt: str,
    model: str = f"{MODEL_TEXT}:featherless-ai",
    max_new_tokens: int = 300,
    temperature: float = 0.8,
) -> str:
    """Call the Hugging Face router with chat‑completion interface."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def clean_questions(raw: str, n: int) -> List[str]:
    """Strip numbering/punctuation from model output and take first *n* lines."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    qs: List[str] = []
    for ln in lines:
        qs.append(ln.lstrip("0123456789).:- ").strip())
    return qs[:n]


def gen_followups(dream: str, n: int = N_FOLLOWUPS) -> List[str]:
    raw = hf_text(FOLLOWUP_PROMPT.format(dream=dream, n=n))
    return clean_questions(raw, n)


def gen_poem(context: str) -> str:
    return hf_text(
        POEM_PROMPT.format(context=context),
        temperature=0.85,
        max_new_tokens=400,
    )


def gen_image(context: str) -> Optional[bytes]:
    img_prompt = hf_text(
        IMAGE_PROMPT_TMPL.format(context=context),
        temperature=0.6,
        max_new_tokens=120,
    )
    url = f"https://api-inference.huggingface.co/models/{MODEL_IMAGE}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": img_prompt}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        st.error(f"❌ Image generation failed: {exc}")
        return None


def gen_melody(_context: str) -> Optional[bytes]:
    """Stub for future music‑gen integration."""
    return None  # Not yet implemented


def save_mp3(text: str) -> Path:
    path = OUTPUT_DIR / f"tts_{uuid.uuid4().hex}.mp3"
    gTTS(text).save(str(path))
    return path


# ────────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ────────────────────────────────────────────────────────────────────────────────

def show_poem(poem: str) -> None:
    """Display poem inside an expander with pre‑wrapped text."""
    with st.expander("📜 Dream Interpretation", expanded=True):
        st.markdown(
            f"<div style='white-space: pre-wrap; font-size:1.1rem;'>{poem}</div>",
            unsafe_allow_html=True,
        )


# ────────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ────────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="DreamSense", page_icon="🌙", layout="centered")
st.title("🌙 DreamSense")
st.caption("Type your dream. We'll ask a few questions, then craft something beautiful.")

st.sidebar.header("Output Options")
output_type = st.sidebar.radio(
    "Choose output format:", ["Poem / Prose (text)", "Image + Poem", "Melody (beta)"]
)

dream = st.text_area("📝 Describe your dream", height=200)

if dream and st.button("Get follow‑up questions"):
    with st.spinner("Crafting clarifying questions …"):
        st.session_state.questions = gen_followups(dream)
        st.session_state.answers = [""] * len(st.session_state.questions)

# ────────────────────────── Follow‑up Q&A ──────────────────────────
if "questions" in st.session_state:
    st.subheader("🔍 Follow‑up Questions")
    for i, q in enumerate(st.session_state.questions):
        st.session_state.answers[i] = st.text_input(q, key=f"fq_{i}")

    do_tts = st.checkbox("🔊 Read it aloud")

    ready = (
        all(st.session_state.answers)
        and st.button("Generate ✨", use_container_width=True)
    )

    if ready:
        context = (
            "Dream: "
            + dream
            + "\nAnswers:\n"
            + "\n".join(f"- {a}" for a in st.session_state.answers)
        )

        # ───────────── Poem / Prose ─────────────
        with st.spinner("Interpreting your dream …"):
            prose = gen_poem(context)

        # ───────────── Melody branch ─────────────
        if output_type.startswith("Melody"):
            st.info(MELODY_PLACEHOLDER_MSG)
            show_poem(prose)
            if do_tts:
                mp3_path = save_mp3(prose)
                st.audio(str(mp3_path))
                with open(mp3_path, "rb") as f:
                    st.download_button(
                        "Download audio", f.read(), file_name="dreamsense.mp3", mime="audio/mpeg"
                    )
            st.download_button("Save poem", prose, file_name="dreamsense.txt", mime="text/plain")

        # ───────────── Image + Poem branch ─────────────
        elif output_type.startswith("Image"):
            with st.spinner("Painting your dream …"):
                img_bytes = gen_image(context)
            if img_bytes:
                st.markdown("### 🖼️ Dream Image")
                st.image(Image.open(BytesIO(img_bytes)))
                st.download_button(
                    "Save image", img_bytes, file_name="dreamsense.png", mime="image/png"
                )
            show_poem(prose)
            st.download_button("Save poem", prose, file_name="dreamsense.txt", mime="text/plain")
            if do_tts:
                mp3_path = save_mp3(prose)
                st.audio(str(mp3_path))
                with open(mp3_path, "rb") as f:
                    st.download_button(
                        "Download audio", f.read(), file_name="dreamsense.mp3", mime="audio/mpeg"
                    )

        # ───────────── Text‑only branch ─────────────
        else:
            show_poem(prose)
            st.download_button("Save poem", prose, file_name="dreamsense.txt", mime="text/plain")
            if do_tts:
                mp3_path = save_mp3(prose)
                st.audio(str(mp3_path))
                with open(mp3_path, "rb") as f:
                    st.download_button(
                        "Download audio", f.read(), file_name="dreamsense.mp3", mime="audio/mpeg"
                    )
# ────────────────────────────────────────────────────────────────────────────────