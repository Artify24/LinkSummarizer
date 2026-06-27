# 📖 Universal URL Summarizer & Chat

A powerful AI-powered application that summarizes content from **YouTube videos** and **websites**, then allows users to **chat with the extracted information** using Retrieval-Augmented Generation (RAG).

Built with **LangChain**, **Groq LLM**, **Hugging Face Embeddings**, **FAISS**, and **Streamlit**.

---

## 🚀 Live Demo

🌐 **Application:** https://linksummarizer-o2pw7enunzspw4ocjnpjme.streamlit.app/

---

## ✨ Features

* 📺 Summarize YouTube videos
* 🌐 Summarize any website/article
* 🤖 Conversational AI powered by Groq
* 🧠 Retrieval-Augmented Generation (RAG)
* 💬 Chat with the summarized content
* ⚡ Fast semantic search using FAISS
* 🔍 Context-aware conversations with chat history
* 🎨 Clean and responsive Streamlit interface

---

## 🏗️ Tech Stack

### Frontend

* Streamlit

### Backend / AI

* LangChain
* Groq (openai/gpt-oss-120b)
* Hugging Face Embeddings
* FAISS Vector Database

### Document Loaders

* YouTube Loader
* Unstructured URL Loader

### Language

* Python

---

## 🧠 Architecture

```text
                 URL
                  │
                  ▼
       Website / YouTube Loader
                  │
                  ▼
        Extract Document Content
                  │
                  ▼
     Recursive Character Splitter
                  │
                  ▼
        HuggingFace Embeddings
                  │
                  ▼
            FAISS Vector DB
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
Document Summary        Conversational RAG
      │                       │
      ▼                       ▼
      Groq LLM         History Aware Retriever
                  │
                  ▼
          Chat with Document
```

---

## 📂 Project Structure

```
.
├── app.py
├── .env
├── requirements.txt
├── README.md
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

---

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/universal-url-summarizer.git
```

Move into the project

```bash
cd universal-url-summarizer
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. Enter a YouTube or Website URL.
2. The application extracts the content.
3. The document is divided into smaller chunks.
4. Hugging Face generates vector embeddings.
5. FAISS stores those embeddings.
6. Groq generates a detailed summary.
7. Users can ask questions about the document.
8. A history-aware retriever provides context-aware answers.

---

## 📸 Features in Action

* Website Summarization
* YouTube Video Summarization
* AI Generated Summary
* Conversational RAG
* Chat History
* Semantic Search
* Fast Vector Retrieval

---

## 🔮 Future Improvements

* PDF Support
* DOCX Support
* Multiple URL Analysis
* Export Summary as PDF
* Voice Chat
* Streaming Responses
* Citation & Source References
* Multi-language Support
* Authentication
* Conversation Export

---

## 🛠️ Libraries Used

* streamlit
* langchain
* langchain-community
* langchain-classic
* langchain-groq
* langchain-huggingface
* faiss-cpu
* sentence-transformers
* python-dotenv
* validators
* unstructured

---

## 👨‍💻 Author

**Zain Sayed**

If you found this project useful, consider giving it a ⭐ on GitHub.

---

