# 🤖 Yours Truly Human Resources Chatbot

An intelligent **HR Chatbot** built with **LangChain**, **FAISS**, and **Google Generative AI**, offering real-time, conversational answers to HR-related queries based on HR policy documents.

The chatbot features a **Streamlit-based UI** and supports **conversational memory**, making it capable of following up on previous questions just like ChatGPT. It uses a locally stored **FAISS vector database** to perform semantic search over HR policies from a comprehensive HR website.

---

## ✨ Features

- 🔍 **Semantic Search** over HR content using vector embeddings and FAISS
- 💬 **Conversational Memory**: remembers previous questions to provide context-aware answers
- 🤖 **Powered by Google Gemini** (via `ChatGoogleGenerativeAI`)
- 📄 **Supports Large-Scale HR Data**: HR content from hundreds of HTML files embedded and indexed
- ⚙️ **Streamlit UI**: Simple and interactive user interface
- ⚡ **Local Vector DB**: Fast and efficient retrieval using FAISS
- 🧠 **LLM + Retriever Chain** using LangChain’s advanced chaining features
- ✅ **Graceful Fallback**: Returns “I don't know” when an answer isn’t found

---

## 🛠️ Tools & Technologies Used

| Tool / Library               | Purpose                                      |
|-----------------------------|----------------------------------------------|
| `LangChain`                 | Core framework for chaining LLM + retrieval  |
| `Google Generative AI`      | LLM for generating human-like responses      |
| `FAISS`                     | Efficient local vector similarity search     |
| `HuggingFace Embeddings`    | Sentence-level semantic embeddings           |
| `Streamlit`                 | Web UI framework                             |
| `BeautifulSoup`             | HTML parsing                                 |
| `dotenv`                    | Secure environment variable management       |
| `sentence-transformers`     | Transformer-based embeddings (`MiniLM`)      |
| `Python`                    | Core backend language                        |

---

## 📁 Folder Structure

```
hr-chatbot/
│
├── faiss_index/ # Generated FAISS vector store
├── hr-policies/ # Folder containing scraped HR HTML pages
├── query.py # Streamlit UI & chatbot logic
├── main.py # Load the HTML pages and create FAISS index
├── requirements.txt # Python dependencies
├── .env # Environment variables (Google API key)
├── hr-logo.png # Logo displayed in Streamlit UI
└── README.md # You're here!
```
---
## 🧠 How It Works

1. **HTML Scraping**  
   Uses `BeautifulSoup` to extract content from downloaded HR policy HTML files.

2. **Text Splitting**  
   The content is split into smaller chunks using `RecursiveCharacterTextSplitter` for better embedding.

3. **Embedding & FAISS Indexing**  
   The chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS index.

4. **Conversational QA**  
   On user input, the chatbot:
   - Reformulates the question using history-aware retriever
   - Retrieves relevant chunks from the FAISS DB
   - Uses Google Gemini (via LangChain) to answer based on retrieved context

5. **Streamlit Frontend**  
   Provides a simple chat-like interface for interacting with the bot.

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/sudarshan-raveendranath/HR-Chatbot.git
```

### 2. Create a Virtual Environment and install dependencies
```bash
python -m venv hr-chatbot-env
source hr-chatbot-env/bin/activate  # On Windows: hr-chatbot-env\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory and add your Google API key:
```plaintext
GOOGLE_API_KEY=your_google_api_key
```

### 4. (Optional) Rebuild the FAISS Index (Run only once)
```bash
# In main.py or a separate file
upload_htmls()
```

### 5. Start the Chatbot
```bash
streamlit run query.py
```
