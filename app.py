import os
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain_groq import ChatGroq
from langchain_anthropic import AnthropicLLM

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.docstore.document import Document
from dotenv import load_dotenv
import tempfile
import os
import getpass

# Set page config at the beginning
st.set_page_config(page_title="Chat with your file", layout="wide")

# Add CSS styles
st.markdown("""
    <style>
        .main {
            background-color:  #000000;
            padding: 20px;
            color:#ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
        }
        .sidebar .sidebar-content h2 {
            color: #333333;
            background-color: #000000;
        }
        .stButton button {
            background-color: #0073e6;
            color: #ffffff;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #005bb5;
        }
        .message {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        .message.user {
            background-color: #e6f7ff;
        }
        .message.bot {
            background-color: #f0f0f0;
        }
        .chat-input {
            background-color: #ffffff;
            border: 1px solid #d9d9d9;
            border-radius: 10px;
            padding: 10px;
            width: 100%;
            box-sizing: border-box;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    load_dotenv()

    st.markdown("<h1 style='text-align: center; color: #0073e6;'>Elevate Your Document Experience with RAG GPT and Conversational AI</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #0073e6;'>🤖 Choose Your AI Model: Select from OpenAI, Google Gemini, ChatGroq, or Claude-2.1 for tailored responses.</h4>", unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "OpenAI"

    with st.sidebar:
        uploaded_files = st.file_uploader("🔍 Upload Your Files", type=['pdf', 'docx', 'csv'], accept_multiple_files=True)
        
        google_api_key = os.getenv("google_api_key")
        qdrant_api_key = os.getenv("qdrant_api_key")
        qdrant_url = os.getenv("qdrant_url")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        groq_api_key = os.getenv("groq_api_key")
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

        
        
        if not google_api_key or not qdrant_api_key or not qdrant_url or not openai_api_key or not groq_api_key or not ANTHROPIC_API_KEY:
            st.info("Please add your API keys to continue.")
            st.stop()

        model_choice = st.radio("Select the model to use", ("Google Gemini", "ChatGroq", "OpenAI","Claude-2.1"))
        st.session_state.selected_model = model_choice

        process = st.button("Process")
        if process:
            pages = get_files_text(uploaded_files)
            st.write("File loaded...")
            if pages:
                st.write(f"Total pages loaded: {len(pages)}")
                text_chunks = get_text_chunks(pages)
                st.write(f"File chunks created: {len(text_chunks)} chunks")
                if text_chunks:
                    vectorstore = get_vectorstore(text_chunks, qdrant_api_key, qdrant_url)
                    st.write("Vector Store Created...")
                    st.session_state.conversation = vectorstore
                    st.session_state.processComplete = True
                    st.session_state.session_id = os.urandom(16).hex()  # Initialize a unique session ID
                else:
                    st.error("Failed to create text chunks.")
            else:
                st.error("No pages loaded from files.")

    if st.session_state.processComplete:
        input_query = st.chat_input("Ask Question about your files.")
        if input_query:
            response_text = rag(st.session_state.conversation, input_query, openai_api_key, google_api_key,groq_api_key,ANTHROPIC_API_KEY, st.session_state.selected_model)
            st.session_state.chat_history.append({"content": input_query, "is_user": True})
            st.session_state.chat_history.append({"content": response_text, "is_user": False})

            response_container = st.container()
            with response_container:
                for i, message_data in enumerate(st.session_state.chat_history):
                    message(message_data["content"], is_user=message_data["is_user"], key=str(i))

def get_files_text(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        if file_extension == ".pdf":
            loader = PyMuPDFLoader(temp_file_path)
            pages = loader.load()
        elif file_extension == ".csv":
            loader = CSVLoader(file_path=temp_file_path)
            pages = loader.load()
        elif file_extension == ".docx":
            loader = Docx2txtLoader(temp_file_path)
            pages = loader.load()
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)
            pages = loader.load()
        else:
            st.error("Unsupported file format.")
            return []

        documents.extend(pages)

        # Remove the temporary file
        os.remove(temp_file_path)
        
    return documents

def get_vectorstore(text_chunks, qdrant_api_key, qdrant_url):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Qdrant.from_texts(texts=text_chunks, embedding=embeddings_model, collection_name="Machine_learning", url=qdrant_url, api_key=qdrant_api_key,force_recreate=True)
    return vectorstore

def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    texts = []
    for page in pages:
        chunks = text_splitter.split_text(page.page_content)
        texts.extend(chunks)
    return texts

def qdrant_client():
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        qdrant_key = st.secrets["qdrant_api_key"]
        URL = st.secrets["qdrant_url"]
        qdrant_client = QdrantClient(
        url=URL,
        api_key=qdrant_key,
        )
        qdrant_store = Qdrant(qdrant_client,"Machine_learning" ,embedding_model)
        return qdrant_store

vector_db = qdrant_client() 

def rag(vector_db, input_query, openai_api_key, google_api_key,groq_api_key,ANTHROPIC_API_KEY,selected_model):
    try:
        template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to provide a detailed and comprehensive answer to the question. If you don't know the answer, just say that you don't know. Offer as much relevant information as possible in your response.

Question: <{question}> 

Context:<{context}> 

Answer:
    """
        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()})

        if selected_model == "Google Gemini":
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=google_api_key)
        elif selected_model == "ChatGroq":
            model = ChatGroq(api_key=groq_api_key, model_name="Llama3-8b-8192")
        elif selected_model == "OpenAI":
            model = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
        elif selected_model == "Claude-2.1":
            model = ChatAnthropic(api_key=ANTHROPIC_API_KEY ,model="claude-2.1", temperature=0, max_tokens=1024)
        else:
            raise ValueError("Invalid model selected.")           
        output_parser = StrOutputParser()
        rag_chain = (
            setup_and_retrieval
            | prompt
            | model
            | output_parser
        )
        response = rag_chain.invoke(input_query)
        return response
    except Exception as ex:
        return str(ex)

if __name__ == '__main__':
    main()
