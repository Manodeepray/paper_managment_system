import streamlit as st
import os
from streamlit_pdf_viewer import pdf_viewer
from PyPDF2 import PdfReader
from llama_index.core import (
    SimpleDirectoryReader,
    ServiceContext,
    PromptHelper,
    VectorStoreIndex
)
from llama_index.llms.langchain import LangChainLLM
from langchain.chat_models import ChatAnthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Free LLM Service: Using Claude API (You can change to another LangChain-supported LLM)
llm = LangChainLLM(
    llm=ChatAnthropic(model_name="claude-2", temperature=0.7, max_tokens=512)
)

# Initialize Embedding Model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load PDF and Create Index
def load_pdf_and_create_index(uploaded_file):
    """Loads a PDF file and creates an index for retrieval."""
    with open(os.path.join("papers", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    pdf_reader = PdfReader(uploaded_file)
    pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    # Create index from PDF text
    documents = [pdf_text]
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    return index

# Query the Index
def query_index(index, user_input):
    """Queries the vector store index for relevant information."""
    return index.as_retriever().retrieve(user_input)

# UI Title
st.title("📚 Chatbot with RAG System (LlamaIndex)")

# Split UI into Two Columns
col1, col2 = st.columns([2, 3])

# Left Column (Chatbot)
with col1:
    st.header("🤖 Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["index"] = None

    # Function to simulate chatbot response
    def get_bot_response(user_input):
        if st.session_state["index"]:
            response = query_index(st.session_state["index"], user_input)
            return response[0].text if response else "No relevant information found."
        return "Please upload a PDF first!"

    # Input box for user queries
    user_input = st.text_input("You:", "")

    if user_input:
        st.session_state.messages.append(f"You: {user_input}")
        bot_response = get_bot_response(user_input)
        st.session_state.messages.append(f"Bot: {bot_response}")

    # Display Chat History
    for message in st.session_state.messages:
        st.write(message)

# Right Column (PDF Viewer & Uploader)
with col2:
    st.header("📄 PDF Viewer & Uploader")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.session_state["index"] = load_pdf_and_create_index(uploaded_file)
        st.success("✅ PDF successfully uploaded and indexed!")

