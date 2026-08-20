# 🩺 Medical RAG Assistant

An AI-powered assistant that answers medical questions using **only** the content of documents you upload — no outside knowledge, no hallucinations, every answer traceable to its exact source file and page number.

Built with **Retrieval-Augmented Generation (RAG)**: your PDFs are chunked, embedded, and searched semantically. Only the most relevant retrieved passages are passed to the language model to generate an answer — grounded strictly in your own trusted documents.

---

## 🚨 The Problem

General-purpose AI models (like ChatGPT) answer medical questions from general training knowledge, which causes real risks:

- **Hallucinations** — inaccurate or outdated medical information
- **No source verification** — answers can't be traced back to a trustworthy reference
- **Clinical risk** — unsupported answers are especially dangerous in healthcare contexts
- **Institutional need** — hospitals, researchers, and clinics need answers grounded *only* in approved documents (protocols, guidelines, research papers)

## ✅ The Solution

A full RAG pipeline purpose-built for medical PDFs:

```
PDF Upload → Text Extraction (OCR-capable) → Chunking → Embeddings
    → Semantic Search → Grounded Answer Generation → Cited Sources
```

Every answer comes with the **exact source file, page number, and similarity/confidence score** — so nothing is taken on faith.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **Document-Grounded Answers** | Responses are generated *only* from retrieved document content — never from the model's general knowledge |
| 🔗 **Source Citations** | Every answer includes file name, page number, and similarity score |
| 🚫 **Smart Refusal** | If no strong enough match exists in the document, the system declines instead of guessing (confidence threshold) |
| 🛡️ **Safety Guardrails** | Automatically refuses personal diagnosis or medication dosage requests |
| 🔒 **Prompt Injection Detection** | Resists attempts (in the user's question or hidden in the document itself) to override system instructions |
| 💡 **Auto-Suggested Questions** | After processing a document, the assistant generates example questions grounded in the actual condition discussed in the file |
| 🌍 **Bilingual** | Fully supports both Arabic and English — matches the language of the question |
| 💬 **Multi-Chat Sessions** | Multiple independent conversations, each with its own uploaded document(s) and history |
| 🌙 **Dark / Light Mode** | Toggle between themes |
| 📎 **Technical Details View** | Optional expandable view showing raw retrieved chunks, similarity scores, and JSON response for debugging/transparency |

---

## 🏗️ How It Works

1. **Upload** — User uploads one or two PDF medical documents (scanned or digital).
2. **Extract & Chunk** — [Docling](https://github.com/docling-project/docling) with OCR support (`RapidOCR`) extracts text; `HybridChunker` splits it into overlapping, context-aware chunks.
3. **Embed** — Each chunk is converted into a vector using `Qwen3-Embedding-0.6B` (via `sentence-transformers`).
4. **Ask** — User submits a question (Arabic or English).
5. **Retrieve** — The question is embedded and compared against all chunk embeddings using cosine similarity (NumPy) to find the top-K most relevant passages.
6. **Safety Check** — The question is screened for prompt-injection attempts and unsafe clinical requests (personal diagnosis/dosage).
7. **Generate** — Only if the best match clears the confidence threshold, the retrieved chunks are passed to **Google Gemini**, instructed to answer *strictly* from that context and to ignore any instructions embedded within the document itself.
8. **Cite** — The answer is displayed with its confidence score and citations (file, page, chunk).

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** — web interface
- **[Docling](https://github.com/docling-project/docling) + RapidOCR** — PDF parsing and OCR (including scanned documents)
- **[Sentence Transformers](https://www.sbert.net/)** (`Qwen/Qwen3-Embedding-0.6B`) — semantic embeddings
- **[Google Gemini API](https://ai.google.dev/)** — grounded answer generation
- **NumPy** — cosine similarity search (brute-force, no vector DB needed at this scale)
- **Python**

> **Why NumPy instead of a vector database?** At the scale of a handful of uploaded PDFs (hundreds–low thousands of chunks), a brute-force cosine similarity search in NumPy is exact, fast (milliseconds), and avoids the operational overhead of running a separate vector DB. A dedicated vector database (FAISS, Pinecone, Chroma, etc.) would make sense at a much larger scale — many documents, persistent storage across sessions/users, or millions of vectors.

---

## 📂 Project Structure

```
├── hackthon.py          # Main Streamlit application
├── requirements.txt      # Python dependencies
├── data/                 # (gitignored) local data / temp files
└── robot_avatar.png      # UI asset
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- A [Google Gemini API key](https://ai.google.dev/)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key
Create a `.env` file or set the environment variable directly:
```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="your-api-key-here"

# macOS/Linux
export GEMINI_API_KEY="your-api-key-here"
```

### 5. Run the app
```bash
streamlit run hackthon.py
```

---

## 🖥️ Usage

1. Open the app in your browser (Streamlit will give you a local URL).
2. Expand **⚙️ Settings** in the sidebar.
3. Upload one (or two) PDF medical document(s).
4. Click **🔄 Process Documents** and wait for the success message.
5. Ask a question in Arabic or English — or click one of the auto-suggested questions.
6. Read the grounded answer along with its source citations.

---

## 🩹 Safety & Responsible AI

This tool is designed as a **documentation support assistant**, not a replacement for professional medical advice:

- It automatically **refuses** requests for personal diagnosis or medication dosage.
- It refuses to comply with prompt-injection attempts, whether typed by the user or hidden inside an uploaded document.
- Every answer can be verified against the original source document — nothing is presented without a traceable reference.

---

## 💼 Proposed Commercialization

**Target customers:**
- Hospitals and medical institutions
- Medical researchers and research centers
- Clinics and healthcare providers
- Healthcare education and training organizations

**Potential business models:**
- B2B subscription for healthcare organizations
- Institutional licensing
- Custom deployment for organizations
- Enterprise support and maintenance

---

## 🔭 Future Work

- Improve extraction accuracy for tables and medical images within PDFs
- Support for additional/external medical knowledge sources
- More rigorous automated evaluation of answer quality
- Migrate to a dedicated vector database if scaling to many concurrent documents/users

---

## 📜 License

*(Add your license here, e.g. MIT)*
