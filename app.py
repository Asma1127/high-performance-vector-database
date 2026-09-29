
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st

from src.vectordb.index import make_index


# -------------------- CONFIG --------------------

ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "documents.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="VectorDB | Semantic Workspace",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- PREMIUM UI CSS --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #f7f8fc;
    color: #172033;
}
.block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #202b3e;
}
[data-testid="stSidebar"] * {
    color: #e5eaf3;
}
[data-testid="stSidebar"] .stButton button {
    background: #202b3e;
    color: #f8fafc;
    border: 1px solid #344156;
    border-radius: 10px;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #2c3a52;
    border-color: #65748b;
}
h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    letter-spacing: -0.7px;
    color: #172033;
}
h1 { font-weight: 800 !important; }
h2, h3 { font-weight: 700 !important; }

.hero {
    background: linear-gradient(120deg, #18243b 0%, #283a63 58%, #5146a8 100%);
    border-radius: 22px;
    padding: 32px 34px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 14px 36px rgba(35, 45, 85, 0.14);
}
.hero * { color: white !important; }
.hero-kicker {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    opacity: .75;
    font-weight: 700;
}
.hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    line-height: 1.2;
    font-weight: 800;
    margin: 12px 0 9px 0;
}
.hero-sub {
    color: #dbe4f5 !important;
    font-size: 15px;
    max-width: 680px;
    line-height: 1.7;
}
.hero-chip {
    display: inline-block;
    border: 1px solid rgba(255,255,255,.22);
    background: rgba(255,255,255,.10);
    border-radius: 30px;
    padding: 7px 12px;
    margin: 18px 7px 0 0;
    font-size: 12px;
    color: #f2f5ff;
}
.section-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #718096;
    font-weight: 700;
    margin-bottom: 8px;
}
.metric-card {
    background: white;
    border: 1px solid #e8ebf2;
    border-radius: 16px;
    padding: 19px 20px;
    min-height: 112px;
    box-shadow: 0 4px 15px rgba(25, 35, 60, .035);
}
.metric-label {
    color: #788399;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 10px;
}
.metric-value {
    color: #19243a;
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
}
.metric-note {
    color: #8993a7;
    font-size: 11px;
    margin-top: 4px;
}
.panel {
    background: white;
    border: 1px solid #e8ebf2;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 5px 18px rgba(25, 35, 60, .035);
    margin-bottom: 18px;
}
.panel-title {
    font-family: 'Manrope', sans-serif;
    color: #1c2940;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 5px;
}
.panel-sub {
    color: #8993a7;
    font-size: 12px;
    margin-bottom: 18px;
}
.result-card {
    background: white;
    border: 1px solid #e7eaf1;
    border-radius: 15px;
    padding: 19px 21px;
    margin: 12px 0;
    box-shadow: 0 4px 15px rgba(25,35,60,.035);
}
.result-title {
    color: #1c2940;
    font-family: 'Manrope', sans-serif;
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 8px;
}
.result-text {
    color: #566176;
    font-size: 13px;
    line-height: 1.75;
}
.score-pill {
    display: inline-block;
    background: #eef2ff;
    color: #5146a8;
    border: 1px solid #dfe4ff;
    border-radius: 30px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 800;
}
.small-meta {
    color: #929bad;
    font-size: 11px;
    margin-top: 12px;
}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    border-radius: 11px !important;
    border: 1px solid #dfe4ed !important;
    background: #fff !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #7770d6 !important;
    box-shadow: 0 0 0 2px rgba(119,112,214,.12) !important;
}
.stButton button, .stFormSubmitButton button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}
div[data-testid="stFileUploader"] {
    border: 1px dashed #cbd3e2;
    border-radius: 13px;
    padding: 8px;
    background: #fafbfe;
}
hr { border-color: #e9edf4; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# -------------------- DATABASE --------------------

def connect():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            dimension INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    return con


# -------------------- EMBEDDINGS --------------------

@st.cache_resource(show_spinner=False)
def get_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2"), "Sentence Transformers"
    except Exception:
        return None, "Hashing fallback"


@st.cache_resource(show_spinner=False)
def get_fallback_embedder():
    from src.vectordb.embedding import Embedder
    return Embedder(dim=384)


def embed(texts):
    model, _ = get_model()
    if model is not None:
        vectors = model.encode(
            texts,
            normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)

    fallback = get_fallback_embedder()
    return fallback.embed_batch(texts).astype(np.float32)


# -------------------- DOCUMENT OPERATIONS --------------------

def add_document(title, content):
    title = title.strip()
    content = content.strip()

    if not title or not content:
        raise ValueError("Title and content are required.")

    vector = embed([content])[0].astype(np.float32)

    con = connect()
    try:
        cur = con.execute(
            """
            INSERT INTO documents
            (title, content, embedding, dimension)
            VALUES (?, ?, ?, ?)
            """,
            (title, content, vector.tobytes(), len(vector))
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def delete_document(doc_id):
    con = connect()
    try:
        cur = con.execute(
            "DELETE FROM documents WHERE id = ?",
            (int(doc_id),)
        )
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()


def get_documents():
    con = connect()
    try:
        return con.execute(
            """
            SELECT id, title, content, created_at
            FROM documents
            ORDER BY id DESC
            """
        ).fetchall()
    finally:
        con.close()


def search(query, top_k):
    con = connect()
    try:
        rows = con.execute(
            """
            SELECT id, title, content, embedding, dimension, created_at
            FROM documents
            """
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return []

    q = embed([query])[0].astype(np.float32)
    index = make_index(len(q))
    documents = {}

    for rid, title, content, blob, dim, created in rows:
        if blob is None or dim is None:
            continue

        vector = np.frombuffer(
            blob,
            dtype=np.float32,
            count=dim
        )

        if len(vector) != len(q):
            continue

        index.add(rid, vector)
        documents[rid] = {
            "id": rid,
            "title": title,
            "content": content,
            "created": created
        }

    matches = index.search(q, top_k)
    results = []

    for doc_id, score in matches:
        if doc_id not in documents:
            continue
        item = documents[doc_id].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def seed():
    con = connect()
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
    finally:
        con.close()

    if count == 0:
        samples = [
            (
                "Vector databases",
                "A vector database stores numerical embeddings and retrieves records using similarity search."
            ),
            (
                "Semantic search",
                "Semantic search understands the meaning of a query instead of matching only exact keywords."
            ),
            (
                "SQLite storage",
                "SQLite is a lightweight relational database suitable for local applications and prototypes."
            ),
            (
                "Embedding model",
                "An embedding model converts text into a numerical vector representation."
            ),
            (
                "Cosine similarity",
                "Cosine similarity compares the direction of two vectors and is commonly used for semantic retrieval."
            ),
        ]
        for title, content in samples:
            add_document(title, content)


# -------------------- HELPERS --------------------

def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def section_heading(title, subtitle=None):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="panel-sub">{subtitle}</div>', unsafe_allow_html=True)


# -------------------- INITIALIZE --------------------

seed()
model, model_name = get_model()
docs = get_documents()
total = len(docs)


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 4px 22px 4px;">
        <div style="font-size:12px;letter-spacing:2px;color:#9ca9c0;font-weight:700;">
            SEMANTIC AI
        </div>
        <div style="font-family:Manrope;font-size:23px;font-weight:800;color:white;margin-top:5px;">
            ✳ VectorDB
        </div>
        <div style="font-size:12px;color:#9ca9c0;margin-top:5px;">
            Intelligent retrieval workspace
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### WORKSPACE")
    page = st.radio(
        "Navigation",
        ["Overview", "Semantic Search", "Documents", "Add Data"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("#### SYSTEM")
    st.markdown(
        f"<div style='font-size:12px;color:#cbd5e1;'>"
        f"● <span style='color:#86efac;'>Local database connected</span><br><br>"
        f"Storage: SQLite<br>"
        f"Embedding: {model_name}<br>"
        f"Dimensions: 384"
        f"</div>",
        unsafe_allow_html=True
    )
    st.divider()
    st.caption("VectorDB • Local prototype")


# -------------------- OVERVIEW --------------------

if page == "Overview":
    st.markdown(f"""
    <div class="hero">
        <div class="hero-kicker">AI-powered information retrieval</div>
        <div class="hero-title">Search by meaning.<br>Discover what matters.</div>
        <div class="hero-sub">
            A semantic search workspace that transforms text into vector
            representations and retrieves relevant information using similarity.
        </div>
        <span class="hero-chip">✳ Semantic embeddings</span>
        <span class="hero-chip">⌘ Vector similarity</span>
        <span class="hero-chip">◈ SQLite storage</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">WORKSPACE OVERVIEW</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("TOTAL DOCUMENTS", total, "Stored in your local database")
    with c2:
        metric_card("STORAGE ENGINE", "SQLite", "Persistent document storage")
    with c3:
        metric_card("EMBEDDING MODEL", "MiniLM" if model else "Fallback", model_name)

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        section_heading(
            "Start exploring",
            "Ask a question in natural language and retrieve related documents."
        )
        with st.form("overview_search"):
            q = st.text_input(
                "Search your knowledge base",
                placeholder="e.g. How does semantic search work?"
            )
            submitted = st.form_submit_button(
                "⌕  Search knowledge base",
                use_container_width=True,
                type="primary"
            )
        if submitted and q.strip():
            st.session_state["search_query"] = q.strip()
            st.session_state["navigate_search"] = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        section_heading("System architecture", "Your current search pipeline")
        st.markdown("""
        **01** &nbsp; Natural-language query

        ↓

        **02** &nbsp; Sentence Transformer embedding

        ↓

        **03** &nbsp; SQLite document store

        ↓

        **04** &nbsp; Cosine similarity ranking

        ↓

        **05** &nbsp; Top-K results
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    section_heading("Recently added", "Latest documents in your workspace")
    if not docs:
        st.info("No documents yet. Add data to get started.")
    else:
        for rid, title, content, created in docs[:4]:
            st.markdown(f"""
            <div style="padding:12px 0;border-bottom:1px solid #edf0f5;">
                <div style="font-weight:800;color:#25324a;">{title}</div>
                <div style="font-size:12px;color:#8792a5;margin-top:4px;">
                    Document #{rid} · {created}
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# -------------------- SEARCH PAGE --------------------

elif page == "Semantic Search":
    st.title("Semantic Search")
    st.caption("Find relevant information using natural-language queries.")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    section_heading("Search workspace", "Describe what you are looking for.")
    query = st.text_input(
        "Your query",
        value=st.session_state.get("search_query", ""),
        placeholder="e.g. Explain how vector similarity works",
        label_visibility="collapsed"
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.selectbox("Results", [3, 5, 10], index=1)
    with col2:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        run_search = st.button(
            "⌕  Search documents",
            type="primary",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if run_search and query.strip():
        st.session_state["search_query"] = query.strip()
        st.session_state["last_results"] = search(query.strip(), top_k)

    if st.session_state.get("last_results") is not None:
        results = st.session_state["last_results"]
        st.markdown(
            f'<div class="section-label">SEARCH RESULTS · {len(results)} FOUND</div>',
            unsafe_allow_html=True
        )
        if not results:
            st.info("No matching documents found. Try another query or add more data.")
        else:
            for item in results:
                st.markdown(f"""
                <div class="result-card">
                    <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;">
                        <div class="result-title">{item['title']}</div>
                        <span class="score-pill">Similarity {item['score']:.3f}</span>
                    </div>
                    <div class="result-text">{item['content']}</div>
                    <div class="small-meta">Document #{item['id']} · {item['created']}</div>
                </div>
                """, unsafe_allow_html=True)


# -------------------- DOCUMENTS PAGE --------------------

elif page == "Documents":
    st.title("Documents")
    st.caption("Browse and manage documents stored in your local database.")

    st.markdown(f"**{total}** documents in your workspace")
    if not docs:
        st.info("Your database is empty. Add a document to begin.")
    else:
        for rid, title, content, created in docs:
            with st.container(border=True):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**{title}**")
                    st.caption(f"Document #{rid} · {created}")
                    st.write(content[:350] + ("..." if len(content) > 350 else ""))
                with col2:
                    if st.button("Delete", key=f"delete_{rid}"):
                        delete_document(rid)
                        st.success("Document deleted.")
                        st.rerun()


# -------------------- ADD DATA PAGE --------------------

elif page == "Add Data":
    st.title("Add Data")
    st.caption("Expand your knowledge base with text documents.")

    manual, upload = st.tabs(["✎  Write manually", "↑  Upload .txt"])

    with manual:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        section_heading("Create a document", "Add a title and the text you want to search.")
        with st.form("manual_add", clear_on_submit=True):
            title = st.text_input("Document title", placeholder="e.g. Introduction to vector databases")
            content = st.text_area(
                "Document content",
                height=220,
                placeholder="Paste or write your document content here..."
            )
            submit = st.form_submit_button(
                "＋  Save document",
                type="primary",
                use_container_width=True
            )
        if submit:
            if title.strip() and content.strip():
                try:
                    add_document(title, content)
                    st.success("Document saved and embedded successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save document: {e}")
            else:
                st.warning("Please enter both a title and document content.")
        st.markdown('</div>', unsafe_allow_html=True)

    with upload:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        section_heading("Upload text file", "Upload a UTF-8 .txt file to add it to your database.")
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=["txt"],
            help="Only .txt files are supported in this version."
        )
        if uploaded_file is not None:
            st.caption(f"Selected: {uploaded_file.name}")
            if st.button("↑  Upload and save", type="primary", use_container_width=True):
                try:
                    file_content = uploaded_file.getvalue().decode("utf-8").strip()
                    if not file_content:
                        st.warning("The selected file is empty.")
                    else:
                        add_document(Path(uploaded_file.name).stem, file_content)
                        st.success("File uploaded and saved.")
                        st.rerun()
                except UnicodeDecodeError:
                    st.error("This file is not UTF-8 encoded.")
                except Exception as e:
                    st.error(f"Could not upload file: {e}")
        st.markdown('</div>', unsafe_allow_html=True)