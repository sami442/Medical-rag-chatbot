import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
st.set_page_config(
    page_title="MedSource — Research Q&A",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Distinct visual style: slate blue + warm gold, sans-serif, card-based chat
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Code+Pro:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
    .stApp { background: #F4F2ED; color: #2B2D3A; }
    .block-container { max-width: 880px; padding-top: 1.5rem; }

    .app-title {
        font-size: 2.2rem; font-weight: 700; color: #2B2D3A; margin: 0;
    }
    .app-sub {
        color: #6B6E80; font-size: 0.95rem; margin-top: 0.2rem;
    }

    .source-card {
        background: white; border: 1px solid #DEDBD0; border-radius: 10px;
        padding: 1.2rem 1.4rem; margin: 1rem 0;
    }
    .source-status {
        font-family: 'Source Code Pro', monospace; font-size: 0.8rem;
        color: #6B6E80; margin-top: 0.5rem;
    }
    .source-status.ready { color: #3A7D5C; }

    .chat-bubble-user {
        background: #2B2D3A; color: #F4F2ED; border-radius: 12px 12px 2px 12px;
        padding: 0.8rem 1.1rem; margin: 0.5rem 0; max-width: 80%;
        margin-left: auto; font-size: 0.95rem;
    }
    .chat-bubble-bot {
        background: white; border: 1px solid #DEDBD0; color: #2B2D3A;
        border-radius: 12px 12px 12px 2px; padding: 0.8rem 1.1rem;
        margin: 0.5rem 0; max-width: 80%; font-size: 0.95rem; line-height: 1.5;
    }

    section[data-testid="stSidebar"] { background: #2B2D3A !important; }
    section[data-testid="stSidebar"] * { color: #F4F2ED !important; }

    .stButton > button {
        background: #C9952C; color: #2B2D3A; border: none; border-radius: 8px;
        padding: 0.6rem 1.4rem; font-weight: 600;
    }

    footer, [data-testid="stToolbar"], #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<p class='app-title'>MedSource</p>
<p class='app-sub'>Ask questions about any medical article, grounded in its actual content.</p>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This tool fetches a live web article, retrieves the most "
        "relevant passages for your question using semantic search, "
        "and asks Gemini to answer using only that retrieved context."
    )
    st.markdown("---")
    st.markdown("**Developer**")
    st.markdown("Samina Mazhar")
    st.markdown("BS Artificial Intelligence")
    st.markdown("---")
    st.markdown("[GitHub](https://github.com/sami442)")
    st.markdown("[Hugging Face](https://huggingface.co/mazharsamina26)")


@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource(show_spinner=False)
def get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')


def fetch_article_text(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "sup"]):
        tag.decompose()
    content_div = soup.find("div", {"id": "mw-content-text"})
    paragraphs = content_div.find_all("p") if content_div else soup.find_all("p")
    text = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def build_index(chunks, embedder):
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index


def retrieve(question, chunks, index, embedder, k=3):
    q_embedding = embedder.encode([question])
    distances, indices = index.search(np.array(q_embedding).astype('float32'), k)
    return [chunks[i] for i in indices[0]]


def answer_question(question, chunks, index, embedder, gemini_model):
    relevant = retrieve(question, chunks, index, embedder)
    context = "\n\n".join(relevant)
    prompt = f"""Answer the question based only on the following context.
If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    response = gemini_model.generate_content(prompt)
    return response.text


# ---- Session state ----
if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.messages = []
    st.session_state.source_url = None

embedder = get_embedder()
gemini_model = get_gemini_model()

# ---- Source input ----
st.markdown("<div class='source-card'>", unsafe_allow_html=True)
url_input = st.text_input(
    "Article URL",
    placeholder="Paste a Wikipedia or medical article URL, e.g. https://en.wikipedia.org/wiki/Diabetic_retinopathy",
)
load_clicked = st.button("Load Source")

if load_clicked and url_input:
    if gemini_model is None:
        st.error("Gemini API key not configured. Add GEMINI_API_KEY in app secrets.")
    else:
        with st.spinner("Fetching and indexing article..."):
            try:
                text = fetch_article_text(url_input)
                if len(text) < 200:
                    st.warning("Very little text extracted — this source may not be scrapeable. Try a different URL.")
                else:
                    chunks = chunk_text(text)
                    index = build_index(chunks, embedder)
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.source_url = url_input
                    st.session_state.messages = []
                    st.success(f"Indexed {len(chunks)} passages from this article.")
            except Exception as e:
                st.error(f"Could not fetch this URL: {e}")

if st.session_state.chunks:
    st.markdown(
        f"<p class='source-status ready'>● source loaded — {st.session_state.source_url}</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown("<p class='source-status'>○ no source loaded yet</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---- Chat interface ----
if st.session_state.chunks:
    for msg in st.session_state.messages:
        bubble_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
        st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    question = st.chat_input("Ask a question about this article...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            answer = answer_question(
                question,
                st.session_state.chunks,
                st.session_state.index,
                embedder,
                gemini_model,
            )
        st.session_state.messages.append({"role": "bot", "content": answer})
        st.rerun()
else:
    st.info("Load an article above to start asking questions.")

st.markdown("---")
st.markdown(
    "Maintained by Samina Mazhar, BS Artificial Intelligence · "
    "[GitHub](https://github.com/sami442) · "
    "[Hugging Face](https://huggingface.co/mazharsamina26)"
)
