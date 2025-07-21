# 🎥 YouTube Chatbot – Chat With Any YouTube Video Using AI

This project enables users to **interact with YouTube videos** using a conversational AI interface. Ask any question, and the chatbot will respond based on the video content. It leverages **dual-mode transcription**, **chunking**, **semantic + lexical retrieval**, and a **powerful LLM** to provide accurate, source-backed answers.

---

## 🚀 Features

- 🧠 Ask questions about any YouTube video
- 🔍 Uses both **semantic** (FAISS) and **keyword-based** (BM25) search
- 📝 Highlights relevant content in the full transcript
- 💬 Maintains chat context using memory
- 📎 Displays original video source for each answer
- 🔁 Fallback to Whisper transcription if API fails

---

## 🛠️ Tech Stack

| Component        | Technology                  |
|------------------|------------------------------|
| UI               | Streamlit                    |
| LLM              | HuggingFaceH4/zephyr-7b-beta |
| Embeddings       | HuggingFaceEmbeddings        |
| Vector Store     | FAISS                        |
| Retriever        | BM25 + FAISS (ensemble)      |
| Memory           | LangChain BufferMemory       |
| Transcript       | YouTubeTranscriptAPI + Whisper |

---

## 🔗 Workflow Overview

1. **📹 YouTube Video → Transcript**  
   - Tries `YouTubeTranscriptAPI` to fetch subtitles  
   - If it fails, uses `Whisper` for local transcription

2. **📄 Text Splitting**  
   - Transcripts are chunked using `RecursiveCharacterTextSplitter`  
   - Parameters:  
     - `chunk_size = 200`  
     - `chunk_overlap = 60`

3. **🔎 Dual Retrieval Setup**  
   - `BM25Retriever`: Keyword-based lexical matching  
   - `FAISS`: Semantic similarity search using embeddings  
   - Combined using `EnsembleRetriever` with 50:50 weighting

4. **🤖 Conversational Chain**  
   - Uses LangChain's `ConversationalRetrievalChain`  
   - Memory: `ConversationSummaryBufferMemory`  
   - Tracks chat history  
   - Returns both answer and supporting transcript snippet

5. **🖥️ Frontend Chat UI**  
   - Built with Streamlit chat interface  
   - Highlights transcript sections matching the retrieved chunks  
   - Source (video title) shown at the bottom in red

---

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Het01/Youtube-Chatbot.git
cd Youtube-Chatbot
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/Mac
source venv/bin/activate
```

3. **Install the dependencies:**

```bash
pip install -r requirements.txt
```
4. **Start the Streamlit app:**

```bash
streamlit run app.py
```

5. **Visit in your browser:**

```bash
http://localhost:8501
```


