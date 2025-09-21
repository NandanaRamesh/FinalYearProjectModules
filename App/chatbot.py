import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("Please set the GEMINI_API_KEY environment variable.")
    st.stop()

# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf}: {e}")
            return None
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Use Streamlit's caching to avoid reprocessing the vector store on every rerun
@st.cache_resource
def get_vector_store(text_chunks, api_key):
    # Pass the API key explicitly to the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# --- Core QA Logic ---

def get_stuff_chain(api_key):
    prompt_template = """
    Answer the user's question as detailed as possible from the provided context.
    Make sure to provide all the details from the text. If the answer is not in the provided context,
    just say, "Answer is not available in the context." Do not provide any other information or wrong answers.

    Context:
    {context}
    
    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Pass the API key explicitly to the chat model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    chain = create_stuff_documents_chain(model, prompt)
    return chain

def get_fallback_chain(api_key):
    prompt = PromptTemplate(
        template="You are an expert financial advisor. Provide a detailed, easy-to-read, and personalized answer to the following question. Use structured formats like tables or bullet points where applicable.\n\nUser asked: {question}",
        input_variables=["question"]
    )
    # Pass the API key explicitly to the fallback model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    return {"model": model, "prompt": prompt}

# --- Main Streamlit App ---

def chatbot_page():
    st.title("Get Your Doubts Cleared")

    # Hardcoded PDF paths
    pdf_paths = [
        os.path.join("assets", "upi.pdf"),
        os.path.join("assets", "WHAT_ARE_MUTUAL_FUNDS.pdf"),
        os.path.join("assets", "SEBI_Financial_Lessons.pdf")
    ]
    
    # Check if the FAISS index already exists to avoid re-creation
    if os.path.exists("faiss_index"):
        st.session_state.vector_store_ready = True
    else:
        st.info("Processing PDF files. This may take a moment...")
        for path in pdf_paths:
            if not os.path.exists(path):
                st.error(f"File not found: {path}. Please check your hardcoded paths.")
                st.session_state.vector_store_ready = False
                return
        
        raw_text = get_pdf_text(pdf_paths)
        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks, GEMINI_API_KEY)
            st.success("Vector store created successfully! You can now ask questions.")
            st.session_state.vector_store_ready = True
        else:
            st.error("Could not extract text from PDFs. Please check your files.")
            st.session_state.vector_store_ready = False

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    user_question = st.chat_input("Ask a question about your documents...", disabled=not st.session_state.vector_store_ready)
    
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
        except Exception as e:
            st.error(f"Failed to load vector store: {e}")
            return
        
        with st.spinner("Thinking..."):
            qa_chain = get_stuff_chain(GEMINI_API_KEY)
            response = qa_chain.invoke({"context": docs, "question": user_question})
            answer_text = response

            if "not available in the context" in answer_text.lower():
                st.info("The answer was not found in the documents. Using general knowledge as a fallback.")
                fallback_chain_info = get_fallback_chain(GEMINI_API_KEY)
                fallback_model = fallback_chain_info["model"]
                fallback_prompt = fallback_chain_info["prompt"]
                fallback_response = fallback_model.invoke(fallback_prompt.format(question=user_question))
                answer_text = fallback_response

            with st.chat_message("assistant"):
                st.write(answer_text)
            st.session_state.chat_history.append({"role": "assistant", "content": answer_text})