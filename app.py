import streamlit as st
import google.generativeai as genai

st.set_page_config(
    page_title="MedSource — Medical AI Assistant",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Code+Pro:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
    .stApp { background: #F4F2ED; color: #2B2D3A; }
    .block-container { max-width: 860px; padding-top: 1.5rem; }

    h1 { color: #2B2D3A !important; font-weight: 700 !important; font-size: 2rem !important; }

    .chat-bubble-user {
        background: #2B2D3A; color: #F4F2ED;
        border-radius: 12px 12px 2px 12px;
        padding: 0.8rem 1.1rem; margin: 0.5rem 0;
        max-width: 78%; margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-bubble-bot {
        background: white; border: 1px solid #DEDBD0; color: #2B2D3A;
        border-radius: 12px 12px 12px 2px;
        padding: 0.8rem 1.1rem; margin: 0.5rem 0;
        max-width: 78%; font-size: 0.95rem; line-height: 1.6;
    }

    section[data-testid="stSidebar"] { background: #2B2D3A !important; }
    section[data-testid="stSidebar"] * { color: #F4F2ED !important; }
    section[data-testid="stSidebar"] hr { border-color: #3A3D52 !important; }

    footer, [data-testid="stToolbar"], #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---- Gemini setup ----
@st.cache_resource(show_spinner=False)
def get_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

SYSTEM_PROMPT = """You are MedSource, a knowledgeable and friendly medical AI assistant.
You answer questions about medical conditions, symptoms, treatments, medications,
anatomy, physiology, diseases, and general health topics clearly and accurately.
Always remind users that your answers are for informational purposes only and
that they should consult a licensed healthcare professional for personal medical advice.
Keep answers concise, well-structured, and easy to understand."""

def answer(messages, gemini):
    # Build conversation history for multi-turn chat
    history = []
    for msg in messages[:-1]:  # all except last
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = gemini.start_chat(history=history)
    response = chat.send_message(
        f"{SYSTEM_PROMPT}\n\nUser question: {messages[-1]['content']}"
        if len(messages) == 1
        else messages[-1]["content"]
    )
    return response.text

# ---- Header ----
st.title("MedSource")
st.caption("Your medical AI assistant — ask about symptoms, conditions, treatments, and more.")

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### What can I help with?")
    st.markdown("""
- 🩺 Symptoms & conditions
- 💊 Medications & treatments
- 🫀 Anatomy & physiology
- 🧬 Diseases & disorders
- 🏥 Medical procedures
- 🥗 Health & nutrition
    """)
    st.markdown("---")
    st.markdown("**Example questions:**")
    st.markdown("""
- *What are the symptoms of diabetes?*
- *How does chemotherapy work?*
- *What causes high blood pressure?*
- *What is the difference between Type 1 and Type 2 diabetes?*
- *How is pneumonia diagnosed?*
    """)
    st.markdown("---")
    st.markdown("⚕️ **Disclaimer**")
    st.markdown(
        "MedSource provides general medical information only. "
        "Always consult a licensed healthcare professional "
        "for personal medical advice and diagnosis."
    )
    st.markdown("---")
    st.markdown("**Developer**")
    st.markdown("Samina Mazhar")
    st.markdown("BS Artificial Intelligence, IUB")
    st.markdown(
        "[GitHub](https://github.com/sami442) · "
        "[Hugging Face](https://huggingface.co/mazharsamina26)"
    )

# ---- Load Gemini ----
gemini = get_gemini()
if gemini is None:
    st.error("Gemini API key not configured. Add GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Chat history ----
for msg in st.session_state.messages:
    bubble = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
    st.markdown(
        f"<div class='{bubble}'>{msg['content']}</div>",
        unsafe_allow_html=True
    )

# ---- Chat input ----
question = st.chat_input("Ask a medical question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        reply = answer(st.session_state.messages, gemini)
    st.session_state.messages.append({"role": "bot", "content": reply})
    st.rerun()

# ---- Footer ----
st.markdown("---")
st.markdown(
    "Maintained by Samina Mazhar, BS Artificial Intelligence · "
    "[GitHub](https://github.com/sami442) · "
    "[Hugging Face](https://huggingface.co/mazharsamina26)"
)
