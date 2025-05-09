#This program is intended to create a Chatbot that accesses a FAISS vector DB that contains a large HR website
#with tons of HR policies, practices and domain knowledge. The chatbot will allow the user to query on any
#HR-related information in a conversational form with conversational memory like ChatGPT.
#The UI of the chatbot is created using Streamlit, and the chatbot is powered by Langchain and Google Generative AI.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import os
import sys
import torch
from dotenv import load_dotenv
load_dotenv()

# Block Streamlit from watching internal torch modules
for mod in list(sys.modules):
    if mod.startswith("torch.classes"):
        sys.modules[mod] = None

def build_chat_history(chat_history_list):
    #This function takes sin the chat history messages in a list of tuples format
    #and turns it into a series of Human and AI message objects
    chat_history = []
    for message in chat_history_list:
        chat_history.append(HumanMessage(content=message[0]))
        chat_history.append(AIMessage(content=message[1]))

    return chat_history

def query(question, chat_history):
    """
    This function does the following;
    1. Receives 2 parameters - 'question' - a string and 'chat_history' - a list of tuples containing accumulating question-answer pairs
    2. Load the local FAISS DB where the entire website is stored as vector embeddings
    3. Create a ConversationalBufferMemory object with 'chat_history'
    4. Create a ConversationalRetrievalChain object with the FAISS DB and the LLM (LLM lets us create retriever objects against the FAISS DB)
    5. Invoke the retriever object with the 'question' and 'chat_history' to get answer
    6. Return the response
    """

    chat_history = build_chat_history(chat_history)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)
    llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", google_api_key = os.getenv("GOOGLE_API_KEY"), temperature = 0)

    condense_question_system_template = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. DO NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is. "
    )

    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("human", "{chat_history}"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, new_db.as_retriever(), condense_question_prompt,
    )

    system_prompt = (
        "You are an assistant for question-answering tasks on HR policy. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return convo_qa_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history,
        }
    )

def show_ui():
    """
    This function does the following:
    1. Implements Streamlit UI
    2. Implements two session variables - 'messages' - to contain the accumulating questions and answers to be displayed on the UI and
    'chat_history' - to contain the accumulating question-answer pairs in a list of tuples format to be served to the retriever objects as chat_history.
    3. For each user query, the response is obtained by invoking the 'query' function and the chat histories are built up
    """

    st.title("Yours Truly Human Resources Chatbot")
    st.image("hr-logo.png")
    st.subheader("Please ask your HR-related questions below:")

    #Initiating chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    #Display chat messages from history on app rerun
    for messages in st.session_state.messages:
        with st.chat_message(messages["role"]):
            st.markdown(messages["content"])

    #Accept user input
    if prompt := st.chat_input("Ask a question about HR policies and practices"):
        #Invoke the function with the retriever with chat history and display responses in the chat container
        with st.spinner("Thinking..."):
            response = query(question = prompt, chat_history = st.session_state.chat_history)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])

            #Append the user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.extend([(prompt, response["answer"])])

if __name__ == "__main__":
    #Run the Streamlit app
    show_ui()