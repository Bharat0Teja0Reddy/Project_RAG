# 🛡️ LocalLens

**LocalLens** is a completely offline, highly secure, browser-based AI document analyzer and chatbot. Built with privacy as the ultimate priority, no data ever leaves your device. It leverages cutting-edge local AI models (Llama 3.2 & LLaVA) via Ollama, combined with a powerful vector database (ChromaDB) to give you a ChatGPT-like experience completely offline.

---

## ✨ Features

- 🔒 **100% Offline & Private:** No cloud APIs, no data telemetry, no internet required. Your documents and chats stay strictly on your local machine.
- 📚 **Universal Document Support:** Chat directly with your files. Supports `.pdf`, `.docx`, `.xlsx`, `.csv`, `.pptx`, `.md`, and `.txt`.
- 👁️ **Image & OCR Processing:** Upload images (`.png`, `.jpg`) and scanned documents. LocalLens uses Tesseract OCR and the LLaVA vision model to extract and comprehend visual data.
- 💬 **Persistent Chat History:** Seamlessly switch between past conversations with a ChatGPT-style sidebar that automatically summarizes and saves your chat sessions.
- 📊 **Dynamic Flowcharts:** Ask the AI to visualize complex workflows, and it will generate live, interactive Mermaid.js diagrams and flowcharts directly in the chat UI.
- 🎨 **Premium UI/UX:** A stunning, fully responsive "Full-Bleed" interface featuring smooth glassmorphism, animated neural cores (Three.js), and personalized operator greetings.

---

## 🛠️ Technology Stack

- **Frontend:** Vanilla HTML, CSS, JavaScript (Zero-build architecture)
- **Backend:** Python, FastAPI
- **AI Engine:** [Ollama](https://ollama.ai/) (Running `llama3.2` for text & `llava` for vision)
- **Vector Database:** ChromaDB (Local persistent storage)
- **Document Processing:** Langchain, PyMuPDF, Unstructured, PyTesseract

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.10+** installed on your system.
2. **Ollama** installed and running on your local machine.
   - Pull the required models by running these commands in your terminal:
     ```bash
     ollama run llama3.2
     ollama run llava
     ```
3. *(Optional but highly recommended)* **Tesseract-OCR** installed for advanced image and scanned PDF processing.

### Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/LocalLens-ai.git
   cd LocalLens-ai
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application!**
   Simply double-click the `Start_LocalLens.bat` file, or run the server manually:
   ```bash
   uvicorn api_server:app --host 127.0.0.1 --port 8000
   ```

4. Open your web browser and navigate to `index.html` to access the terminal!

---

## 📂 Architecture Note

When you upload a document, LocalLens splits the text into chunks, generates vector embeddings (using a lightweight CPU-bound embedding model to prevent GPU overload), and securely stores them in a local `chroma_db` folder. Your persistent chat histories are securely stored in `chats.json`. Both of these are automatically ignored by Git to protect your privacy.
