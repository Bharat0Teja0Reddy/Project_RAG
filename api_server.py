import os
import tempfile
import base64
import requests
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- CRITICAL FOR TRUE OFFLINE MODE ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# --------------------------------------

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

import json
import uuid

# Optional: Point to Tesseract binary if not in system PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI()

# Allow all origins so the browser doesn't block the upload with a CORS "Failed to fetch" error
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHATS_FILE = "chats.json"

def load_chats():
    if not os.path.exists(CHATS_FILE):
        return {}
    try:
        with open(CHATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_chats(chats):
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, indent=4)

from typing import Optional

class ChatRequest(BaseModel):
    message: str
    document_name: Optional[str] = None  # None or "All Documents" means search everything
    mode: str = "pdf"
    chat_id: Optional[str] = None
class DeleteRequest(BaseModel):
    document_name: str

vector_db = None
db_loaded = False

print("Loading Local Embedding Model (Offline Mode)...")
try:
    # CRITICAL FIX: 'device': 'cpu' stops the RTX 3050 from crashing during PDF uploads!
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
except Exception as e:
    print(f"CRITICAL ERROR loading embeddings: {e}")
    embeddings = None

print("Loading local vector database...")
try:
    if embeddings:
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        db_loaded = True
        print("Database loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load ChromaDB. Error: {e}")

@app.get("/documents")
async def get_documents():
    if db_loaded and vector_db:
        try:
            data = vector_db.get()
            unique_sources = set()
            for meta in data.get("metadatas", []):
                if meta and "source" in meta:
                    unique_sources.add(meta["source"])
            return {"documents": list(unique_sources)}
        except Exception as e:
            print(f"Error reading documents: {e}")
    return {"documents": []}

@app.post("/clear")
async def clear_database():
    global vector_db, db_loaded
    try:
        if db_loaded and vector_db:
            # Fetch all existing document IDs and delete them to wipe the slate clean
            existing_data = vector_db.get()
            ids_to_delete = existing_data.get("ids", [])
            if ids_to_delete:
                vector_db.delete(ids=ids_to_delete)
            return {"message": "Database wiped successfully."}
        return {"message": "Database is already empty."}
    except Exception as e:
        print(f"❌ CLEAR ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats")
async def get_chats():
    chats = load_chats()
    return [{"chat_id": k, "title": v.get("title", "New Chat")} for k, v in chats.items()]

@app.get("/chats/{chat_id}")
async def get_chat_history(chat_id: str):
    chats = load_chats()
    if chat_id in chats:
        return chats[chat_id]
    raise HTTPException(status_code=404, detail="Chat not found")

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    chats = load_chats()
    if chat_id in chats:
        del chats[chat_id]
        save_chats(chats)
        return {"message": "Deleted"}
    return {"message": "Not found"}

@app.post("/delete")
async def delete_document(request: DeleteRequest):
    global vector_db, db_loaded
    try:
        if db_loaded and vector_db:
            print(f"Deleting all chunks for document: {request.document_name}")
            existing_data = vector_db.get()
            ids_to_delete = []
            
            # Find all chunks that match the specific document name
            for i, metadata in enumerate(existing_data.get("metadatas", [])):
                if metadata and metadata.get("source") == request.document_name:
                    ids_to_delete.append(existing_data["ids"][i])
            
            if ids_to_delete:
                vector_db.delete(ids=ids_to_delete)
                return {"message": f"Successfully deleted {request.document_name}."}
            return {"message": "Document not found in database."}
        return {"message": "Database is empty."}
    except Exception as e:
        print(f"❌ DELETE ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    
    # Eject Vision model from VRAM to make room for Chat model
    try:
        requests.post("http://localhost:11434/api/generate", json={"model": "llava", "keep_alive": 0}, timeout=2)
    except:
        pass
        
    chats = load_chats()
    chat_id = request.chat_id
    
    if not chat_id or chat_id not in chats:
        chat_id = str(uuid.uuid4())
        
        # Ask LLM to generate a quick 3-5 word summary title for this query
        try:
            title_payload = {
                "model": "llama3.2",
                "messages": [
                    {"role": "system", "content": "You are a helpful AI that summarizes user queries into very short, concise titles. Output ONLY the title (maximum 5 words), no quotes, no extra text."},
                    {"role": "user", "content": f"Summarize this query into a title: {request.message}"}
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 10}
            }
            title_resp = requests.post("http://localhost:11434/api/chat", json=title_payload, timeout=5)
            title = title_resp.json().get("message", {}).get("content", "").strip().replace('"', '')
            if not title:
                title = "New Conversation"
        except Exception as e:
            print("Failed to generate title:", e)
            title = "New Conversation"
            
        chats[chat_id] = {"title": title, "messages": []}
    
    chat_history = chats[chat_id]["messages"]

    try:
        if request.mode == "general":
            system_message = "You are a highly intelligent, friendly, and helpful AI assistant (similar to ChatGPT or Gemini). Explain concepts in the simplest way possible, use analogies, scenarios, and examples to make your answers easy to understand for humans. You do not have access to any specific uploaded documents right now."
            llm_user_msg = request.message
        else:
            context_text = "No context found."
            
            if db_loaded and vector_db:
                # Increase K to provide much more context to the LLM (helps with summarization)
                search_kwargs = {"k": 12}
                if request.document_name and request.document_name != "All Documents":
                    # Strictly filter ChromaDB search to ONLY the selected document
                    search_kwargs["filter"] = {"source": request.document_name}
                
                results = vector_db.similarity_search(request.message, **search_kwargs)
                if results:
                    context_text = "\n\n".join([doc.page_content for doc in results])

            system_message = """You are an expert, friendly Document Analyzer (like ChatGPT or Gemini). Answer the user's question using the provided Context.
            Break down complex information into simple, easy-to-understand explanations. Use bullet points, analogies, and real-world scenarios if it helps make the concept clearer.
            If the user asks for a summary, provide a comprehensive, easy-to-read summary based on the available Context chunks. Do not complain that you don't have the full document.
            If the answer to a specific factual question is completely absent from the Context, kindly state that you cannot answer it based on the document.
            If the user asks for a flowchart, graph, or diagram, generate valid Mermaid.js code enclosed in ```mermaid blocks.
            Do not use outside general knowledge for factual document questions."""
            
            # Inject context into the user message for strong attention
            llm_user_msg = f"Context:\n{context_text}\n\nQuestion: {request.message}"

        chat_history.append({"role": "user", "content": request.message})
        
        # Build ephemeral message list for this request
        messages_for_llm = [{"role": "system", "content": system_message}] + chat_history[:-1] + [{"role": "user", "content": llm_user_msg}]
        
        payload = {
            "model": "llama3.2", 
            "messages": messages_for_llm,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        response.raise_for_status()
        
        ai_response = response.json().get("message", {}).get("content", "Error: No response generated.")
        chat_history.append({"role": "assistant", "content": ai_response})
        
        chats[chat_id]["messages"] = chat_history
        save_chats(chats)
        
        return {"response": ai_response, "chat_id": chat_id}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_db, db_loaded, embeddings
    
    try:
        ext = file.filename.split('.')[-1].lower()
        if ext not in ["pdf", "txt", "md", "png", "jpg", "jpeg", "docx", "csv", "xlsx", "pptx"]:
            raise Exception("Unsupported file type.")
            
        print(f"\n--- Processing Upload: {file.filename} ---")
        
        # --- DEDUPLICATION LOGIC ---
        if db_loaded and vector_db:
            print(f"Checking for existing file '{file.filename}'...")
            try:
                existing_data = vector_db.get()
                ids_to_delete = []
                for i, metadata in enumerate(existing_data.get("metadatas", [])):
                    if metadata and metadata.get("source") == file.filename:
                        ids_to_delete.append(existing_data["ids"][i])
                
                if ids_to_delete:
                    print(f"Found existing chunks for '{file.filename}'. Overwriting...")
                    vector_db.delete(ids=ids_to_delete)
            except Exception as e:
                 print(f"Warning: Could not perform deduplication check: {e}")
        # ---------------------------

        # Eject Chat model from VRAM to make room for reading files
        try:
            requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "keep_alive": 0}, timeout=2)
        except:
            pass

        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        clean_filename = file.filename
        documents = []

        if ext == "pdf":
            print("Reading PDF...")
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Check if PDF is essentially empty (scanned image)
            total_chars = sum(len(doc.page_content.strip()) for doc in docs)
            if total_chars < 50:
                print("PDF appears to be a scanned image. Falling back to Tesseract OCR...")
                try:
                    pdf_document = fitz.open(tmp_path)
                    ocr_text = ""
                    for page_num in range(len(pdf_document)):
                        page = pdf_document.load_page(page_num)
                        pix = page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes()))
                        text = pytesseract.image_to_string(img)
                        ocr_text += f"\n--- Page {page_num + 1} ---\n" + text
                    
                    if ocr_text.strip():
                        docs = [Document(page_content=ocr_text)]
                    else:
                        raise Exception("OCR failed to find text.")
                except Exception as e:
                    print(f"OCR Error: {e}")
                    raise Exception("Could not read PDF. If it's scanned, ensure Tesseract-OCR is installed in C:\\Program Files\\Tesseract-OCR\\tesseract.exe")

            for doc in docs:
                doc.metadata["source"] = clean_filename
            documents.extend(docs)
            
        elif ext in ["txt", "md"]:
            print("Reading Text...")
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = clean_filename
            documents.extend(docs)
            
        elif ext == "docx":
            print("Reading Word Document...")
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = clean_filename
            documents.extend(docs)
            
        elif ext == "csv":
            print("Reading CSV...")
            loader = CSVLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = clean_filename
            documents.extend(docs)
            
        elif ext == "xlsx":
            print("Reading Excel...")
            loader = UnstructuredExcelLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = clean_filename
            documents.extend(docs)
            
        elif ext == "pptx":
            print("Reading PowerPoint...")
            loader = UnstructuredPowerPointLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = clean_filename
            documents.extend(docs)
                
        elif ext in ["png", "jpg", "jpeg"]:
            print("👁️ Waking Vision Model (Llava)...")
            with open(tmp_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            payload = {
                "model": "llava",
                "prompt": "Describe this image in high detail. Transcribe all text.",
                "images": [encoded_string],
                "stream": False,
                "keep_alive": 0 
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            desc = response.json().get("response", "")
            documents = [Document(page_content=f"IMAGE DESCRIPTION ({clean_filename}):\n{desc}", metadata={"source": clean_filename})]

        os.remove(tmp_path)

        if not documents:
            raise Exception("No readable text found in this file.")

        print("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            raise Exception("No content chunks generated.")

        if not embeddings:
            raise Exception("System Error: Embeddings model failed to load. Check console.")

        print("Saving to database...")
        if db_loaded and vector_db:
            vector_db.add_documents(chunks)
        else:
            vector_db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")
            db_loaded = True

        print("--- Upload Complete! ---")
        return {"message": "Success"}

    except Exception as e:
        print(f"❌ UPLOAD ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting AirGap AI Server...")
    # CHANGED: 127.0.0.1 bypasses Windows Firewall completely instead of 0.0.0.0
    uvicorn.run(app, host="127.0.0.1", port=8000)