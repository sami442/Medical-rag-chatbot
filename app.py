import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(
    page_title="MedSource — Portfolio Assistant",
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
    .chat-wrap { margin-bottom: 1rem; }

    section[data-testid="stSidebar"] { background: #2B2D3A !important; }
    section[data-testid="stSidebar"] * { color: #F4F2ED !important; }
    section[data-testid="stSidebar"] hr { border-color: #3A3D52 !important; }

    .stButton > button {
        background: #C9952C; color: #2B2D3A; border: none;
        border-radius: 8px; padding: 0.5rem 1.2rem; font-weight: 600;
    }

    footer, [data-testid="stToolbar"], #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---- Pre-loaded knowledge base about Samina's portfolio ----
KNOWLEDGE_BASE = """
# About Samina Mazhar
Samina Mazhar is a BS Artificial Intelligence student at Islamia University Bahawalpur (IUB), Pakistan.
GPA: 3.61/4.0 — Top 5% of batch.
Ehsaas Scholarship 2020-24 (Fully Funded).
PM Youth Laptop Scheme 2023 awardee.
GitHub: github.com/sami442
Hugging Face: huggingface.co/mazharsamina26
Research interests: Computer Vision, Medical AI, Deep Learning.

# Project 1: Brain MRI Tumor Segmentation
Repository: github.com/sami442/medical-image-segmentation
Live App: medical-image-segmentation-jc6hrzsdhjimse9d47n5uz.streamlit.app
Architecture: U-Net CNN for semantic segmentation.
Dataset: Kaggle LGG MRI Segmentation dataset — 110 patients, 3929 brain MRI images. Real clinical dataset.
Task: Binary pixel-wise segmentation — tumor region vs. normal brain tissue.
Performance: 99.37% accuracy, Dice coefficient 0.3147.
The Dice score started at 0.0136 and improved significantly after switching from binary cross-entropy loss to a combined BCE + Dice loss function to handle class imbalance between tumor and non-tumor pixels.
Model saved as both .h5 (full model on Hugging Face) and .tflite (0.12MB, for Streamlit deployment).
Design: Dark neon aesthetic — NeuroScan AI branding, teal/green accents.
Challenges: Class imbalance (tumor pixels are rare), domain adaptation, TFLite conversion.

# Project 2: CancerShield AI — Multi-Cancer Detection
Repository: github.com/sami442/multi-cancer-detection
Live App: multi-cancer-detection-9jme9mlzxhhllkct4ec3ft.streamlit.app
Architecture: Two independent SVM (Support Vector Machine) classifiers — one per cancer type.
Datasets:
- Breast Cancer: Wisconsin Diagnostic dataset (UCI), 569 samples, 30 features. Real clinical dataset.
- Ovarian Cancer: Coimbra dataset (UCI), 116 samples, 9 biomarker features. Real clinical dataset.
Performance:
- Breast Cancer SVM: 98.25% accuracy
- Ovarian Cancer SVM: 87.50% accuracy
Models saved as .pkl files with StandardScaler.
Design: White/red clinical aesthetic — CancerShield AI branding.
This project uses tabular/structured data (biomarkers and diagnostic measurements) rather than images, making it technically distinct from the image-based projects.

# Project 3: MediScan — Multi-Disease Screening Console
Repository: github.com/sami442/multi-disease-imaging-ai
Live App: multi-disease-imaging-ai.streamlit.app
Architecture: Four independent CNN (Convolutional Neural Network) binary classifiers, one per disease.
Design: Dark canvas with terracotta and teal accents — clinical chart / triage board aesthetic.
Fonts: Fraunces serif + JetBrains Mono — deliberately distinct from other two apps.

Disease 1 — Pneumonia Detection:
Dataset: Kaggle Chest X-ray dataset (Paul Mooney), 5860 chest X-ray images. Real clinical dataset.
Task: Binary classification — Normal vs. Pneumonia.
Accuracy: 82.25%

Disease 2 — Diabetic Retinopathy Detection:
Dataset: APTOS-2019 Kaggle dataset (Gaussian-filtered), 3662 retina fundus images. Real clinical dataset.
Task: Binary classification — No DR vs. Has DR (simplified from original 5-class).
Accuracy: 92.77%

Disease 3 — Skin Lesion Classification:
Dataset: HAM10000 (Human Against Machine with 10000 training images), 10015 dermoscopy images. Real clinical dataset.
Task: Binary classification — Benign vs. Malignant. High recall on malignant class prioritized for screening safety.
Accuracy: 78.00%

Disease 4 — COVID-19 Detection:
Dataset: COVID-19 Radiography Database, 21165 chest X-ray images. Real clinical dataset.
Task: Binary classification — Normal vs. Abnormal (COVID + Lung Opacity + Viral Pneumonia combined).
Accuracy: 75.83%
Transfer learning with ResNet50 was attempted but performed worse (~55-66%) due to ImageNet-to-X-ray domain mismatch. Reverted to custom CNN as the better model — this is a documented learning point about when transfer learning helps vs. hurts.
All 4 models converted to TFLite for lightweight Streamlit deployment.
TF version used: tensorflow-cpu==2.17.0 (required to support TFLite opcode version 12).

# Technical Stack Across All Projects
Languages: Python 3.10
Frameworks: TensorFlow/Keras (CNN training), Scikit-learn (SVM), Streamlit (deployment)
Model formats: .h5 (training), .tflite (deployment), .pkl (SVM)
Hosting: Streamlit Cloud (free tier), Hugging Face (large model storage)
Version control: GitHub (sami442)
All datasets are real, publicly available clinical datasets from Kaggle and UCI — no synthetic data used.
"""

# ---- Chunk and index the knowledge base at startup ----
@st.cache_resource(show_spinner=False)
def build_knowledge_index():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Split by double newlines to preserve semantic blocks
    chunks = [c.strip() for c in KNOWLEDGE_BASE.split('\n\n') if c.strip()]
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return embedder, chunks, index


@st.cache_resource(show_spinner=False)
def get_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')


def retrieve(question, chunks, index, embedder, k=4):
    q_emb = embedder.encode([question])
    _, indices = index.search(np.array(q_emb).astype('float32'), k)
    return [chunks[i] for i in indices[0]]


def answer(question, chunks, index, embedder, gemini):
    relevant = retrieve(question, chunks, index, embedder)
    context = "\n\n".join(relevant)
    prompt = f"""You are a helpful assistant that answers questions about 
Samina Mazhar's AI/ML portfolio and background.
Answer based on the context below. Be specific and helpful.
If the answer isn't in the context, say so honestly.

Context:
{context}

Question: {question}

Answer:"""
    return gemini.generate_content(prompt).text


# ---- Header ----
st.title("MedSource")
st.caption("Ask me anything about Samina's medical AI projects, datasets, models, or background.")

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### What can I answer?")
    st.markdown("""
- 🧠 Brain Tumor Segmentation project
- 🏥 CancerShield — breast & ovarian cancer
- 🩺 MediScan — 4-disease screening console
- 📊 Datasets, accuracy, architectures used
- 👩‍💻 Samina's background & skills
    """)
    st.markdown("---")
    st.markdown("**Example questions:**")
    st.markdown("""
- *What dataset was used for brain tumor segmentation?*
- *What accuracy did the pneumonia model achieve?*
- *Why did ResNet50 underperform on COVID-19?*
- *What is Samina's GPA?*
- *How many diseases does MediScan screen for?*
    """)
    st.markdown("---")
    st.markdown("**Developer**")
    st.markdown("Samina Mazhar")
    st.markdown("BS Artificial Intelligence, IUB")
    st.markdown("[GitHub](https://github.com/sami442) · [Hugging Face](https://huggingface.co/mazharsamina26)")

# ---- Load resources ----
with st.spinner("Loading knowledge base..."):
    embedder, chunks, index = build_knowledge_index()
gemini = get_gemini()

if gemini is None:
    st.error("Gemini API key not configured. Add GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Chat history ----
st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)
for msg in st.session_state.messages:
    bubble = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
    st.markdown(f"<div class='{bubble}'>{msg['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---- Chat input ----
question = st.chat_input("Ask about any of Samina's projects...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        reply = answer(question, chunks, index, embedder, gemini)
    st.session_state.messages.append({"role": "bot", "content": reply})
    st.rerun()

# ---- Footer ----
st.markdown("---")
st.markdown(
    "Maintained by Samina Mazhar, BS Artificial Intelligence · "
    "[GitHub](https://github.com/sami442) · "
    "[Hugging Face](https://huggingface.co/mazharsamina26)"
)
