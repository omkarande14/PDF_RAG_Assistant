import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration
st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise RAG Assistant (Llama 3 8B)")
st.caption("Powered by Ollama, LangChain & ChromaDB | 100% On-Premise & Secure")

# 2. Sidebar for Document Upload
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Initialize Vector DB in Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 3. Process PDF Document
if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("Processing PDF and generating embeddings..."):
        # Save uploaded file to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load Document
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # Split Document into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Generate Embeddings & Store in ChromaDB
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        # Clean up temporary file
        os.remove(tmp_file_path)
        st.sidebar.success("PDF processed and indexed successfully!")

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Processing
if user_prompt := st.chat_input("Ask a question about your document or anything else..."):
    # Display User Message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = OllamaLLM(model="llama3:8b")

            if st.session_state.vector_store:
                # RAG Mode: Retrieve Context & Format Prompt directly via LCEL
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(user_prompt)
                
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert AI assistant. Use the following context to answer the question concisely.\n\nContext:\n{context}"),
                    ("human", "{input}")
                ])
                
                chain = prompt_template | llm
                ai_response = chain.invoke({"context": context, "input": user_prompt})
            else:
                # Direct Chat Mode (No PDF Uploaded)
                ai_response = llm.invoke(user_prompt)

            st.markdown(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})