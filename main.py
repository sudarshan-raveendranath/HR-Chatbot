import os
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def upload_htmls():
    """This function does the followings:
    1. Reads recursively through the given folder hr-policies (within the current folder)
    2. Loads the pages (Documents)
    3. Loaded documents are split into smaller chunks
    4. Embeddings are created for documents
    5. Vector store is created using the embeddings
    """

    #Load all the HTML pages in the given folder structure recursively using the Directory loader
    #use wget.exe -r -l inf -k -p -E -A.html -P hr-policies https://www.hrhelpboard.com/hr-policies.html
    documents = []
    for root, _, files in os.walk("hr-policies"):
        for file in files:
            if file.endswith(".html"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        html = f.read()
                        soup = BeautifulSoup(html, "html.parser")
                        text = soup.get_text(separator="\n")
                        documents.append(Document(page_content=text, metadata={"source": path}))
                except Exception as e:
                    print(f"Error loading file {path}: {e}")

    print(f"{len(documents)} pages loaded")

    #Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators = ["\n\n", "\n", " ", ""]
    )

    split_documents = text_splitter.split_documents(documents = documents)
    print(f"{len(split_documents)} chunks created")

    print(split_documents[0].metadata)

    #Upload chunks as vector embeddings into FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index_path = "faiss_index"
    if os.path.exists(os.path.join(index_path, "index.faiss")) and \
            os.path.exists(os.path.join(index_path, "index.pkl")):
        print("FAISS index already exists. Skipping creation.")
    else:
        print("Creating new FAISS index...")
        vector_db = FAISS.from_documents(split_documents, embeddings)
        vector_db.save_local(index_path)
        print("FAISS index saved.")

def faiss_query():
    """
    This function does the following:
    1. Load the local FAISS DB
    2. Trigger a semantic similarity search using a query
    3. This retrieves semantically matching Vectors from the DB
    """

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)

    query = "Explain  the candidate onboarding process"
    docs = new_db.similarity_search(query)

    #Print all the extracted Vectors from the above query
    for doc in docs:
        print("##---- Page ----##")
        print(doc.metadata['source'])
        print("##---- Content ----##")
        print(doc.page_content)

if __name__ == "__main__":
    #The below code 'upload_htmls()' is executed only once and then commented as the Vector Database is…
    #experiments
    #upload_htmls()
    #The below function is experimental to trigger a semantic search on the Vector DB
    faiss_query()
