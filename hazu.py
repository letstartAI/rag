"""
Hazu Multimodal RAG Agent — 3D Gemini Edition
==============================================
Features:
1. Complete Document Visibility & Accurate Counts: Automatically extracts document overviews (total entries, rows, columns, pages) so questions like "how many entries are there?" are answered with 100% exact numbers!
2. Full-Context Smart Feeding: For small-to-medium files (up to 30 chunks / ~1,000 rows), feeds complete data into Gemini's 1-million token context window, eliminating missing excerpts.
3. Multi-File Upload: Upload multiple PDFs, Word documents (.docx/.doc), Excel sheets (.xlsx/.xls), CSVs, and TXT files simultaneously.
4. Resilient Embedding Engine: Automatic batching and polite rate-limiting with auto-retry and live countdowns for Google Gemini's 30k tokens/min free tier embedding limit.
5. Pure Gemini Intelligence: Fast, reliable Gemini models with automatic 429 Resource-Exhausted / quota-fallback protection.
6. 3D Glassmorphic UI: High-performance 3D visual effects with perspective tilts, glowing layered shadows, tactile 3D buttons, and glossy modern cards.
7. Per-User Memory: Auto-saves current chats and past sessions per user to disk.
"""

import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv, find_dotenv, unset_key

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Model & Provider imports (Pure Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(find_dotenv(), override=True)
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)

st.set_page_config(
    page_title="Hazu 3D RAG & Gemini Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Safe defaults
default_gemini_model = "gemini-flash-lite-latest"
chunk_size = 800
chunk_overlap = 100
top_k = 10
temperature = 0.2
web_search_enabled = False

# Fallback candidate models in case of 429 quota exhaustion or 404
GEMINI_FALLBACK_CANDIDATES = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

# ------------------------------------------------------------ 3D ANIMATED STYLING
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #090a16;
        --card-bg: rgba(22, 26, 48, 0.75);
        --card-border: rgba(255, 255, 255, 0.12);
        --accent-glow: rgba(124, 58, 237, 0.35);
        --primary-gradient: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #d946ef 100%);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.18) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(217, 70, 239, 0.15) 0%, transparent 45%),
                    radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.8) 0%, transparent 60%),
                    linear-gradient(135deg, #070814 0%, #0d1024 50%, #151433 100%);
        background-attachment: fixed;
        color: #f1f5f9;
    }

    /* 3D Floating Header */
    .rag-header {
        background: linear-gradient(135deg, rgba(49, 46, 129, 0.85) 0%, rgba(79, 70, 229, 0.85) 35%, rgba(124, 58, 237, 0.85) 70%, rgba(217, 70, 239, 0.85) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 24px;
        padding: 28px 34px;
        box-shadow: 0 20px 50px -10px rgba(124, 58, 237, 0.5),
                    0 0 35px rgba(79, 70, 229, 0.25),
                    inset 0 1px 2px rgba(255, 255, 255, 0.4);
        margin-bottom: 20px;
        transform: perspective(1000px) translateZ(0);
        transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.35s ease;
        position: relative;
        overflow: hidden;
    }
    .rag-header:hover {
        transform: perspective(1000px) translateY(-3px) translateZ(8px);
        box-shadow: 0 26px 60px -8px rgba(124, 58, 237, 0.65),
                    0 0 45px rgba(217, 70, 239, 0.35),
                    inset 0 1px 2px rgba(255, 255, 255, 0.5);
    }
    .rag-header h1 {
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        margin: 0;
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -0.5px;
        text-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }
    .rag-header p {
        color: #e2e8f0;
        margin: 8px 0 0 0;
        font-size: 15px;
        font-weight: 500;
    }

    /* 3D Glassmorphic Cards */
    .glass-card-3d {
        background: rgba(22, 26, 48, 0.72);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.5),
                    0 0 20px rgba(124, 58, 237, 0.12),
                    inset 0 1px 1px rgba(255, 255, 255, 0.18);
        transform: perspective(1000px) translateZ(0);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card-3d:hover {
        transform: perspective(1000px) translateY(-2px) translateZ(4px);
        box-shadow: 0 22px 50px -8px rgba(0, 0, 0, 0.65),
                    0 0 28px rgba(124, 58, 237, 0.22),
                    inset 0 1px 1px rgba(255, 255, 255, 0.25);
    }

    /* 3D Chat Bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 18px !important;
        margin-bottom: 14px !important;
        padding: 16px 20px !important;
        transform: perspective(1000px) translateZ(0);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    [data-testid="stChatMessage"]:hover {
        transform: perspective(1000px) translateY(-1px) translateZ(2px);
    }

    /* Assistant Bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(22, 26, 48, 0.8) !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        box-shadow: 0 12px 32px -6px rgba(0, 0, 0, 0.5),
                    inset 0 1px 1px rgba(255, 255, 255, 0.12) !important;
    }

    /* User Bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #4338ca 0%, #6366f1 50%, #7c3aed 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: 0 12px 28px rgba(79, 70, 229, 0.38),
                    inset 0 1px 1px rgba(255, 255, 255, 0.35) !important;
    }

    /* 3D Interactive Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 12px 18px !important;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4),
                    inset 0 1px 1px rgba(255, 255, 255, 0.35) !important;
        transform: perspective(600px) translateZ(0);
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:hover {
        transform: perspective(600px) translateY(-2px) translateZ(6px) !important;
        box-shadow: 0 14px 32px rgba(124, 58, 237, 0.55),
                    inset 0 1px 1px rgba(255, 255, 255, 0.5) !important;
    }
    .stButton > button:active {
        transform: perspective(600px) translateY(1px) translateZ(0) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
    }

    /* 3D Pulse Status Badges */
    .badge-3d {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.3px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35), inset 0 1px 1px rgba(255,255,255,0.3);
        transform: perspective(600px) translateZ(0);
    }
    .badge-green {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.25);
        animation: pulseGlowGreen 2.5s infinite;
    }
    .badge-blue {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.25);
        animation: pulseGlowBlue 2.5s infinite;
    }
    .badge-purple {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.25);
    }

    @keyframes pulseGlowGreen {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5), inset 0 1px 1px rgba(255,255,255,0.3); }
        70% { box-shadow: 0 0 0 12px rgba(16, 185, 129, 0), inset 0 1px 1px rgba(255,255,255,0.3); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0), inset 0 1px 1px rgba(255,255,255,0.3); }
    }
    @keyframes pulseGlowBlue {
        0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.5), inset 0 1px 1px rgba(255,255,255,0.3); }
        70% { box-shadow: 0 0 0 12px rgba(59, 130, 246, 0), inset 0 1px 1px rgba(255,255,255,0.3); }
        100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0), inset 0 1px 1px rgba(255,255,255,0.3); }
    }

    /* 3D Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(13, 15, 28, 0.94) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 8px 0 25px rgba(0, 0, 0, 0.45);
    }

    /* Inputs with 3D Depth */
    [data-testid="stChatInput"] {
        border-radius: 18px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), inset 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------- per-user storage + chat memory
USER_DATA_DIR = BASE_DIR / "user_data"
LAST_USER_FILE = USER_DATA_DIR / "last_user.txt"


def safe_username(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "").strip()).strip("_")
    return clean or "default_user"


def user_dir(name: str) -> Path:
    d = USER_DATA_DIR / safe_username(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def sessions_dir(name: str) -> Path:
    d = user_dir(name) / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_user_config(name: str) -> dict:
    try:
        p = user_dir(name) / "config.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_user_config(name: str, cfg: dict):
    if not name:
        return
    try:
        (user_dir(name) / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_last_user() -> str:
    try:
        if LAST_USER_FILE.exists():
            return LAST_USER_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def set_last_user(name: str):
    try:
        USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        LAST_USER_FILE.write_text(name or "", encoding="utf-8")
    except Exception:
        pass


def load_current_chat(name: str) -> list:
    """Returns list of (role, text) tuples from disk."""
    try:
        p = user_dir(name) / "current_chat.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return [(m[0], m[1]) for m in data if isinstance(m, list) and len(m) == 2]
    except Exception:
        pass
    return []


def save_current_chat(name: str, messages: list):
    if not name:
        return
    try:
        data = [[r, t] for r, t in messages]
        (user_dir(name) / "current_chat.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def save_session(name: str, messages: list, document_info: str = "") -> str:
    """Save the current chat as a named memory session. Returns filename."""
    if not name or not messages:
        return ""
    title = ""
    for role, text in messages:
        if role == "user":
            title = text.strip()[:60]
            break
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"{stamp}_{safe_username(title)[:40] or 'chat'}.json"
    try:
        (sessions_dir(name) / fname).write_text(
            json.dumps(
                {"created": stamp, "document": document_info, "messages": [[r, t] for r, t in messages]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
    return fname


def list_sessions(name: str) -> list:
    try:
        files = sorted(sessions_dir(name).glob("*.json"), reverse=True)
        out = []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                msgs = data.get("messages", [])
                first_q = next((t for r, t in msgs if r == "user"), f.name)
                out.append((f.name, data.get("created", ""), first_q[:60], len(msgs)))
            except Exception:
                out.append((f.name, "", f.name, 0))
        return out
    except Exception:
        return []


def load_session(name: str, fname: str) -> list:
    try:
        data = json.loads((sessions_dir(name) / fname).read_text(encoding="utf-8"))
        return [(m[0], m[1]) for m in data.get("messages", []) if len(m) == 2]
    except Exception:
        return []


# ------------------------------------------------------------- session state
if "username" not in st.session_state:
    st.session_state.username = get_last_user() or "hazu"
if "active_user" not in st.session_state:
    st.session_state.active_user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "doc_overviews" not in st.session_state:
    st.session_state.doc_overviews = []
if "files_sig" not in st.session_state:
    st.session_state.files_sig = None
if "indexed_files_summary" not in st.session_state:
    st.session_state.indexed_files_summary = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
if "active_model" not in st.session_state:
    st.session_state.active_model = default_gemini_model


def extract_plain_text(response):
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
        ]
        if parts:
            return "\n".join(parts)
    return str(content)


# ------------------------------------------------------------------ file loaders
def load_pdf(pdf_bytes: bytes, filename: str):
    from langchain_community.document_loaders import PyPDFLoader

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        loaded = PyPDFLoader(tmp_path).load()
        for doc in loaded:
            doc.metadata["source"] = filename

        # Add structured document overview chunk
        overview_text = (
            f"=== PDF DOCUMENT OVERVIEW ===\n"
            f"File: {filename}\n"
            f"Total Pages: {len(loaded)} pages\n"
        )
        overview_doc = Document(page_content=overview_text, metadata={"source": filename, "type": "overview"})
        return [overview_doc] + loaded
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def load_docx(file_bytes: bytes, filename: str):
    import io
    from docx import Document as DocxDocument

    doc = DocxDocument(io.BytesIO(file_bytes))
    texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if row_text:
                texts.append(row_text)

    overview_text = (
        f"=== WORD DOCUMENT OVERVIEW ===\n"
        f"File: {filename}\n"
        f"Total Paragraphs & Table Rows: {len(texts)} sections\n"
    )
    overview_doc = Document(page_content=overview_text, metadata={"source": filename, "type": "overview"})
    content_text = "\n\n".join(texts)
    content_doc = Document(page_content=content_text, metadata={"source": filename, "type": "docx"})
    return [overview_doc, content_doc]


def load_excel(file_bytes: bytes, filename: str):
    """
    Reads all sheets, computes exact row/entry counts, and creates
    an explicit Spreadsheet Overview Chunk so counting queries like
    'how many entries are there' are answered with 100% precision.
    """
    import io
    import pandas as pd

    sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, dtype=str)
    docs = []

    total_file_entries = 0
    sheet_summaries = []

    for sheet_name, df in sheets.items():
        df = df.dropna(how="all")
        df = df.fillna("")
        row_count = len(df)
        total_file_entries += row_count
        cols = list(df.columns.astype(str))
        header_str = " | ".join(cols)

        sheet_summaries.append(
            f"Sheet '{sheet_name}': {row_count} total rows/entries. Columns ({len(cols)}): {cols}"
        )

        if df.empty:
            continue

        # Group rows into batches of 30 so each chunk retains clear column headers
        row_batch_size = 30
        for start_idx in range(0, len(df), row_batch_size):
            sub_df = df.iloc[start_idx : start_idx + row_batch_size]
            rows = [" | ".join(row.astype(str)) for _, row in sub_df.iterrows()]
            text = (
                f"Sheet: {sheet_name} (Entries {start_idx+1} to {start_idx+len(sub_df)} of {row_count} total)\n"
                f"Columns: {header_str}\n" + "\n".join(rows)
            )
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "sheet": str(sheet_name),
                    "type": "excel",
                    "rows_range": f"{start_idx+1}-{start_idx+len(sub_df)}",
                }
            ))

    # Add master overview chunk at the top
    overview_text = (
        f"=== SPREADSHEET OVERVIEW ===\n"
        f"File: {filename}\n"
        f"Total Entries Across File: {total_file_entries} entries\n"
        + "\n".join(sheet_summaries)
        + f"\n============================="
    )
    overview_doc = Document(
        page_content=overview_text,
        metadata={"source": filename, "type": "overview", "total_entries": total_file_entries}
    )

    return [overview_doc] + docs


def load_text(file_bytes: bytes, filename: str):
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = [l for l in text.splitlines() if l.strip()]

    overview_text = (
        f"=== TEXT DOCUMENT OVERVIEW ===\n"
        f"File: {filename}\n"
        f"Total Non-Empty Lines: {len(lines)} lines\n"
    )
    overview_doc = Document(page_content=overview_text, metadata={"source": filename, "type": "overview"})
    cleaned_text = re.sub(r'\n{3,}', '\n\n', text)
    content_doc = Document(page_content=cleaned_text, metadata={"source": filename, "type": "text"})
    return [overview_doc, content_doc]


def load_single_file(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        return load_pdf(data, uploaded_file.name)
    if name.endswith(".docx"):
        return load_docx(data, uploaded_file.name)
    if name.endswith(".doc"):
        try:
            return load_docx(data, uploaded_file.name)
        except Exception:
            return load_text(data, uploaded_file.name)
    if name.endswith((".xlsx", ".xls")):
        return load_excel(data, uploaded_file.name)
    if name.endswith((".csv", ".txt")):
        if name.endswith(".csv"):
            # Also parse CSV via pandas for exact row counting
            try:
                import io
                import pandas as pd
                df = pd.read_csv(io.BytesIO(data), dtype=str)
                df = df.dropna(how="all").fillna("")
                row_count = len(df)
                header = " | ".join(df.columns.astype(str))
                overview = Document(
                    page_content=f"=== CSV OVERVIEW ===\nFile: {uploaded_file.name}\nTotal Rows/Entries: {row_count}\nColumns: {header}\n",
                    metadata={"source": uploaded_file.name, "type": "overview", "total_entries": row_count}
                )
                rows = [" | ".join(r.astype(str)) for _, r in df.iterrows()]
                content = Document(
                    page_content=f"CSV Content ({uploaded_file.name}):\nHeader: {header}\n" + "\n".join(rows),
                    metadata={"source": uploaded_file.name, "type": "csv"}
                )
                return [overview, content]
            except Exception:
                pass
        return load_text(data, uploaded_file.name)
    raise ValueError(f"Unsupported format: {uploaded_file.name}")


def build_multi_vectorstore(uploaded_files, size: int, overlap: int, api_key: str, progress_bar):
    """
    Builds the vectorstore using incremental batch embedding to prevent hitting
    Google's 30,000 tokens-per-minute Free Tier rate limit. Includes automatic
    retry countdowns on 429 RESOURCE_EXHAUSTED.
    """
    all_docs = []
    summary = []
    overviews = []
    n_files = len(uploaded_files)

    for idx, uf in enumerate(uploaded_files):
        pct = int(5 + (idx / n_files) * 30)
        progress_bar.progress(pct, f"Reading {uf.name} ({idx+1}/{n_files})…")
        try:
            docs = load_single_file(uf)
            # Separate overview chunks from content chunks
            for d in docs:
                if d.metadata.get("type") == "overview":
                    overviews.append(d)
                all_docs.append(d)
            summary.append({"name": uf.name, "docs": len(docs), "status": "ok"})
        except Exception as e:
            summary.append({"name": uf.name, "docs": 0, "status": f"Error: {str(e)[:100]}"})

    if not all_docs:
        raise RuntimeError("No text could be extracted from the uploaded files.")

    progress_bar.progress(40, "Splitting documents into optimized chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    progress_bar.progress(50, f"Preparing 3D embeddings for {len(chunks)} chunks…")
    embeddings = None
    _emb_errors = []
    for _model in ["models/gemini-embedding-001", "models/gemini-embedding-2"]:
        try:
            _cand = GoogleGenerativeAIEmbeddings(model=_model, google_api_key=api_key)
            _cand.embed_query("test")
            embeddings = _cand
            break
        except Exception as _e:
            _emb_errors.append(f"{_model}: {_e}")

    if embeddings is None:
        raise RuntimeError("Gemini embedding initialization error: " + " | ".join(_emb_errors)[:400])

    # Incremental batch indexing with 429 auto-retry and live countdown
    batch_size = 5
    total_chunks = len(chunks)
    vs = None

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        batch_pct = int(52 + (i / total_chunks) * 44)
        progress_bar.progress(
            batch_pct,
            f"Embedding chunks {i+1}–{min(i+batch_size, total_chunks)} of {total_chunks}…"
        )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                if vs is None:
                    vs = FAISS.from_documents(batch, embeddings)
                else:
                    vs.add_documents(batch)
                time.sleep(0.4)
                break
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                    m = re.search(r'retry in\s+([0-9.]+)\s*s', err_str, re.IGNORECASE)
                    if not m:
                        m = re.search(r'retryDelay.*?(\d+)\s*s', err_str, re.IGNORECASE)
                    wait_sec = int(float(m.group(1))) + 2 if m else 30

                    for remaining in range(wait_sec, 0, -1):
                        progress_bar.progress(
                            batch_pct,
                            f"⏳ Rate limit reached (free tier: 30k tokens/min). Resuming automatically in {remaining}s… ({min(i+batch_size, total_chunks)}/{total_chunks} chunks)"
                        )
                        time.sleep(1)
                else:
                    raise e

    progress_bar.progress(100, f"Indexed {len(chunks)} chunks across {len(uploaded_files)} files!")
    return vs, chunks, overviews, summary


# ------------------------------------------------------------- web search
def perform_web_search(query: str, max_results: int = 4):
    """Returns (results, error). Results: list of {title, href, body}."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results)), ""
    except Exception as e:
        return [], str(e)[:200]


# ------------------------------------------------------------- unified answering with 429 quota protection
def generate_ai_answer(
    question: str,
    context: str,
    web_text: str,
    gemini_key: str,
    preferred_model: str,
    temperature: float,
) -> str:
    """
    Generates an answer using Google Gemini with automatic 429 / Resource Exhausted
    model-fallback protection.
    """
    system_instruction = (
        "You are an expert AI document and data analysis assistant powered by Google Gemini.\n"
        "Ground your answers thoroughly in the provided document context below.\n"
        "IMPORTANT RULES:\n"
        "1. For questions about total entries, total rows, counts, list of columns, or document structure, "
        "refer to the SPREADSHEET/DOCUMENT OVERVIEW section to provide the exact number and details.\n"
        "2. Do NOT say 'the document only contains excerpts' if the overview or context provides the count or information.\n"
        "3. Provide direct, complete, and articulate answers.\n"
    )

    user_prompt_template = ""
    inputs = {"question": question}
    if context:
        user_prompt_template += "--- DOCUMENT CONTEXT ---\n{context}\n\n"
        inputs["context"] = context
    if web_text:
        user_prompt_template += "--- WEB SEARCH RESULTS ---\n{web_text}\n\n"
        inputs["web_text"] = web_text

    user_prompt_template += "Question: {question}\n\nAnswer:"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", user_prompt_template),
    ])

    models_to_try = [preferred_model] + [m for m in GEMINI_FALLBACK_CANDIDATES if m != preferred_model]

    last_error = None
    for model_name in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_key,
                temperature=temperature,
            )
            chain = prompt | llm | StrOutputParser()
            res = extract_plain_text(chain.invoke(inputs))
            if res and res.strip():
                st.session_state.active_model = model_name
                return res
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "NOT_FOUND" in err_str or "404" in err_str:
                time.sleep(0.3)
                continue
            else:
                raise e

    raise last_error


# -------------------------------------------------------------------- SIDEBAR
with st.sidebar:
    # ➕ New Chat
    if st.button("➕ New Chat", key="new_chat_btn"):
        if st.session_state.messages and st.session_state.username:
            summary_doc = ", ".join([f["name"] for f in st.session_state.indexed_files_summary]) or "general_chat"
            save_session(st.session_state.username, st.session_state.messages, summary_doc)
        st.session_state.messages = []
        if st.session_state.username:
            save_current_chat(st.session_state.username, [])
        st.rerun()

    st.markdown("### ⚙️ User Profile & Keys")
    username_input = st.text_input(
        "Your Name / Profile",
        value=st.session_state.username,
        placeholder="e.g. hazu",
    )

    gemini_input = st.text_input(
        "✨ Google Gemini API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="Paste your Gemini API key (AIza…)",
        help="Get a free API key at https://aistudio.google.com",
    )

    _c1, _c2 = st.columns(2)
    with _c1:
        if st.button("💾 Save Key"):
            if username_input.strip() and gemini_input.strip():
                st.session_state.api_key = gemini_input.strip()
                save_user_config(username_input.strip(), {
                    "gemini_api_key": st.session_state.api_key,
                })
                st.success(f"Key saved for {username_input.strip()}!")
            else:
                st.warning("Enter your name and Gemini key.")
    with _c2:
        if st.button("🗑️ Forget Key"):
            if st.session_state.username:
                try:
                    (user_dir(st.session_state.username) / "config.json").unlink(missing_ok=True)
                except Exception:
                    pass
            st.session_state.api_key = ""
            st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Model & Settings")
    model_choices = [
        "gemini-flash-lite-latest",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
    ]
    cur_idx = model_choices.index(st.session_state.active_model) if st.session_state.active_model in model_choices else 0
    chosen_model = st.selectbox(
        "Gemini Model",
        model_choices,
        index=cur_idx,
        help="gemini-flash-lite-latest has fast responses and high free quota limits.",
    )
    st.session_state.active_model = chosen_model

    web_search_enabled = st.checkbox(
        "🌐 Live Web Grounding",
        value=False,
        help="Search the internet in real-time when answering.",
    )

    with st.expander("📊 Advanced RAG Parameters"):
        chunk_size = st.slider("Chunk size (characters)", 400, 2000, 800, step=100)
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 100, step=50)
        top_k = st.slider("Retrieved chunks (top-k)", 4, 25, 10)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    # 🧠 MEMORY SECTION
    st.markdown("---")
    st.markdown("### 🧠 Chat Memory")
    st.caption(f"{len(st.session_state.messages)} message(s) in current chat.")

    if st.session_state.messages:
        _chat_text = "\n\n".join(f"{r.upper()}: {t}" for r, t in st.session_state.messages)
        st.download_button("⬇️ Export Chat", _chat_text, file_name="chat-history.txt")

    _sessions = list_sessions(st.session_state.username) if st.session_state.username else []
    if _sessions:
        st.markdown(f"**Saved Chats ({len(_sessions)}):**")
        _labels = {f"{c} — {q}": fn for fn, c, q, _n in _sessions}
        _chosen = st.selectbox("Saved History", list(_labels.keys()))
        _col1, _col2 = st.columns(2)
        with _col1:
            if st.button("📂 Load"):
                st.session_state.messages = load_session(st.session_state.username, _labels[_chosen])
                save_current_chat(st.session_state.username, st.session_state.messages)
                st.rerun()
        with _col2:
            if st.button("🗑️ Delete"):
                try:
                    (sessions_dir(st.session_state.username) / _labels[_chosen]).unlink(missing_ok=True)
                except Exception:
                    pass
                st.rerun()
    else:
        st.caption("No archived sessions yet.")

    if st.button("🧹 Clear Chat Display"):
        st.session_state.messages = []
        if st.session_state.username:
            save_current_chat(st.session_state.username, [])
        st.rerun()

# --- Sync user profile switch
_current_user = (username_input or "").strip()
if _current_user and _current_user != st.session_state.username:
    if st.session_state.active_user:
        save_current_chat(st.session_state.active_user, st.session_state.messages)
    st.session_state.username = _current_user
    set_last_user(_current_user)
    _cfg = load_user_config(_current_user)
    st.session_state.api_key = _cfg.get("gemini_api_key", "")
    st.session_state.messages = load_current_chat(_current_user)
    st.session_state.active_user = _current_user
    st.rerun()
elif st.session_state.active_user is None and st.session_state.username:
    _u = st.session_state.username
    _cfg = load_user_config(_u)
    if not st.session_state.api_key:
        st.session_state.api_key = _cfg.get("gemini_api_key", "") or os.getenv("GEMINI_API_KEY", "")
    if not st.session_state.messages:
        st.session_state.messages = load_current_chat(_u)
    st.session_state.active_user = _u

# Auto-update keys in session
if gemini_input and gemini_input.strip() != st.session_state.api_key:
    st.session_state.api_key = gemini_input.strip()

# --------------------------------------------------------------------- HEADER
st.markdown(
    """
    <div class="rag-header">
        <h1>✨ Hazu 3D Gemini Agent</h1>
        <p>Chat freely anytime, or upload multiple documents (PDF, Word, Excel, CSV, TXT) powered by Google Gemini intelligence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- MULTI-FILE UPLOAD ZONE
with st.expander("📁 Upload Documents (PDF, Word, Excel, CSV, TXT) — Optional", expanded=False):
    st.markdown(
        """
        <div style="font-size: 14px; color: #94a3b8; margin-bottom: 10px;">
            ✨ You do <b>NOT</b> need to upload files first to chat! Feel free to chat below immediately.
            Upload any combination of files whenever you want to query your documents.
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Choose files to index",
        type=["pdf", "docx", "doc", "xlsx", "xls", "csv", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        current_sig = tuple((f.name, f.size) for f in uploaded_files) + (chunk_size, chunk_overlap)
        if st.session_state.get("files_sig") != current_sig:
            if not st.session_state.api_key:
                st.error("🔑 Please enter your **Google Gemini API Key** in the sidebar to index documents.")
            else:
                prog = st.progress(0, "Processing documents…")
                try:
                    vs, chunks, overviews, summary = build_multi_vectorstore(
                        uploaded_files, chunk_size, chunk_overlap, st.session_state.api_key, prog
                    )
                    st.session_state.vectorstore = vs
                    st.session_state.all_chunks = chunks
                    st.session_state.doc_overviews = overviews
                    st.session_state.files_sig = current_sig
                    st.session_state.indexed_files_summary = summary
                    st.balloons()
                    st.success(f"Successfully indexed {len(uploaded_files)} file(s) into {len(chunks)} searchable chunks!")
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        st.error(
                            "⏳ **Embedding Rate Limit (429)**: The Google Gemini Free Tier limits embeddings to "
                            "30,000 tokens per minute. Please wait ~30 seconds for the free quota to reset, then click retry."
                        )
                    else:
                        st.error(f"Error indexing files: {err_msg[:400]}")
                    st.session_state.vectorstore = None
                    st.session_state.all_chunks = []
                    st.session_state.doc_overviews = []
        else:
            st.info(f"📂 {len(uploaded_files)} file(s) are actively indexed and ready for queries.")
    else:
        if st.session_state.vectorstore is not None:
            st.session_state.vectorstore = None
            st.session_state.all_chunks = []
            st.session_state.doc_overviews = []
            st.session_state.files_sig = None
            st.session_state.indexed_files_summary = []

# Status Indicator Bar
if st.session_state.vectorstore is not None:
    doc_count = len(st.session_state.indexed_files_summary)
    total_chunks_count = len(st.session_state.all_chunks)
    full_context_tag = " (Full-Document Mode active)" if total_chunks_count <= 30 else ""
    st.markdown(
        f'<div style="margin-bottom: 16px;"><span class="badge-3d badge-green">📚 Document Mode: {doc_count} file(s), {total_chunks_count} chunks{full_context_tag}</span> &nbsp; <span class="badge-3d badge-purple">✨ Engine: {st.session_state.active_model}</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div style="margin-bottom: 16px;"><span class="badge-3d badge-blue">💬 General Chat Mode (Upload files anytime)</span> &nbsp; <span class="badge-3d badge-purple">✨ Engine: {st.session_state.active_model}</span></div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ CHAT DISPLAY
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.write(text)

# ------------------------------------------------------------------ CHAT INPUT & EXECUTION
user_input = st.chat_input("Ask anything, or ask questions about your uploaded documents…")

if user_input:
    # 1. Verification of keys before answering
    if not st.session_state.api_key:
        st.error("🔑 Please enter your **Google Gemini API Key** in the sidebar to chat.")
        st.stop()

    # 2. Append User message
    st.session_state.messages.append(("user", user_input))
    save_current_chat(st.session_state.username, st.session_state.messages)
    with st.chat_message("user"):
        st.write(user_input)

    # 3. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner(f"🔍 Thinking with Gemini ({st.session_state.active_model})…"):
            try:
                # Document RAG Context (if files are indexed)
                context_str = ""
                source_docs = []
                if st.session_state.vectorstore is not None:
                    all_chunks = st.session_state.all_chunks
                    overviews = st.session_state.doc_overviews

                    # Detect if query asks for overview, counting, listing, or full sheet
                    q_lower = user_input.lower()
                    is_global_query = any(w in q_lower for w in [
                        "how many", "count", "total", "entries", "all", "list all", "summary",
                        "overview", "columns", "rows", "full sheet", "entire", "number of"
                    ])

                    # FULL-CONTEXT MODE: If file has <= 30 chunks, pass all chunks into Gemini's 1M context!
                    if len(all_chunks) <= 30:
                        source_docs = all_chunks
                    elif is_global_query:
                        # For global queries on larger files, retrieve high top-k and always prepend overviews
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": min(20, len(all_chunks))})
                        retrieved = retriever.invoke(user_input)
                        # Ensure overviews are prepended
                        source_docs = overviews + [d for d in retrieved if d not in overviews]
                    else:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
                        retrieved = retriever.invoke(user_input)
                        source_docs = overviews + [d for d in retrieved if d not in overviews]

                    context_str = "\n\n---\n\n".join(d.page_content for d in source_docs)

                # Optional Web Search Context
                web_text_str = ""
                web_results = []
                if web_search_enabled:
                    web_results, _ = perform_web_search(user_input, max_results=4)
                    if web_results:
                        web_text_str = "\n\n".join(
                            f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}"
                            for r in web_results
                        )

                # Generate Answer with 429 quota protection
                answer = generate_ai_answer(
                    question=user_input,
                    context=context_str,
                    web_text=web_text_str,
                    gemini_key=st.session_state.api_key,
                    preferred_model=st.session_state.active_model,
                    temperature=temperature,
                )

                # Show answer directly in chat
                st.write(answer)
                st.session_state.messages.append(("assistant", answer))
                save_current_chat(st.session_state.username, st.session_state.messages)

                # Source Citations below in 3D expanders
                if source_docs:
                    with st.expander(f"📚 Document Sources ({len(source_docs)} chunks used)"):
                        for i, doc in enumerate(source_docs, 1):
                            src = doc.metadata.get("source", "Document")
                            page_or_sheet = doc.metadata.get("page", doc.metadata.get("sheet", ""))
                            rows_range = doc.metadata.get("rows_range", "")
                            doc_type = doc.metadata.get("type", "")

                            extra_parts = []
                            if page_or_sheet != "":
                                extra_parts.append(f"Sheet/Page: {page_or_sheet}")
                            if rows_range != "":
                                extra_parts.append(f"Rows: {rows_range}")
                            if doc_type == "overview":
                                extra_parts.append("Master Overview")

                            extra_str = f" ({', '.join(extra_parts)})" if extra_parts else ""
                            st.markdown(f"**{i}. {src}{extra_str}**")
                            st.caption(doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else ""))

                if web_results:
                    with st.expander(f"🌐 Live Web Sources ({len(web_results)} results)"):
                        for r in web_results:
                            st.markdown(f"[{r.get('title', 'Source')}]({r.get('href', '#')})")
                            st.caption((r.get("body", "") or "")[:250])

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    st.error(
                        "⚠️ **Gemini Quota Exceeded (429 RESOURCE_EXHAUSTED)**\n\n"
                        "The free-tier request limit for this API key has been reached. "
                        "You can switch models in the sidebar or grab a fresh free API key from "
                        "[Google AI Studio](https://aistudio.google.com) and paste it into the sidebar."
                    )
                elif "401" in err_msg or "API_KEY_INVALID" in err_msg:
                    st.error("🔑 Invalid API Key. Please verify your Gemini API key in the sidebar.")
                else:
                    st.error(f"Error: {err_msg[:400]}")
