# 🩺 MedSource — Medical AI Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Gemini](https://img.shields.io/badge/Gemini-2.5--flash-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medical-rag-chatbot-hzsybqc3unhgstjmrmazoq.streamlit.app/)

## 🚀 Live Demo
👉 [**Try the App Here**](https://medical-rag-chatbot-hzsybqc3unhgstjmrmazoq.streamlit.app/)

## 📌 Overview
MedSource is a conversational medical AI assistant powered by Google Gemini.
It answers questions about medical conditions, symptoms, treatments, medications,
anatomy, physiology, and general health topics in a clean, multi-turn chat interface.
Built as a demonstration of LLM integration and conversational AI design.

## ✨ Features
- 💬 Multi-turn conversation — remembers context across questions
- 🩺 Covers symptoms, conditions, treatments, medications, anatomy
- ⚕️ Built-in medical disclaimer on every session
- 🎨 Clean chat-bubble interface with warm gold/slate design
- ⚡ Fast responses powered by Gemini 2.5 Flash

## 🛠️ Tech Stack
| Technology | Purpose |
|------------|---------|
| Python 3.10 | Core language |
| Google Gemini 2.5 Flash | LLM for answer generation |
| Streamlit | Web interface and deployment |
| Streamlit Secrets | Secure API key management |

## 🖥️ App Interface
The app opens directly to a chat interface — no setup required.
Just type any medical question and get an instant, structured answer.

**Example questions you can ask:**
- *What are the symptoms of diabetic retinopathy?*
- *How does chemotherapy work?*
- *What is the difference between Type 1 and Type 2 diabetes?*
- *How is pneumonia diagnosed and treated?*
- *What causes high blood pressure?*

## 🚀 How to Run Locally
1. Clone the repository
```bash
git clone https://github.com/sami442/medical-rag-chatbot.git
cd medical-rag-chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add your Gemini API key
Create a file `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

4. Run the app
```bash
streamlit run app.py
```

## 🔑 Getting a Gemini API Key
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy and add to your Streamlit secrets

## ⚕️ Medical Disclaimer
MedSource provides **general medical information only** and is intended
for educational and informational purposes. It is not a substitute for
professional medical advice, diagnosis, or treatment. Always consult a
licensed healthcare professional for personal medical concerns.

## 🔗 Related Projects
- 🧠 [Brain MRI Tumor Segmentation](https://github.com/sami442/medical-image-segmentation) · [Live App](https://medical-image-segmentation-jc6hrzsdhjimse9d47n5uz.streamlit.app/)
- 🏥 [CancerShield — Multi-Cancer Detection](https://github.com/sami442/multi-cancer-detection) · [Live App](https://multi-cancer-detection-9jme9mlzxhhllkct4ec3ft.streamlit.app/)
- 🩺 [MediScan — 4-Disease Screening Console](https://github.com/sami442/multi-disease-imaging-ai) · [Live App](https://multi-disease-imaging-ai.streamlit.app/)

## 👩‍💻 Author

**Samina Mazhar**
BS Artificial Intelligence
Islamia University Bahawalpur
Bahawalpur, Punjab, Pakistan

### 🔗 Connect:
- 🐙 **GitHub:** [@sami442](https://github.com/sami442)
- 🤗 **Hugging Face:** [mazharsamina26](https://huggingface.co/mazharsamina26)

## 📄 License
This project is licensed under the MIT License.
