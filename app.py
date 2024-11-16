import streamlit as st
import os
from streamlit_pdf_viewer import pdf_viewer

# Set Streamlit page configuration to "wide"
st.set_page_config(layout="wide")

import streamlit as st
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, ServiceContext, LLMPredictor, PromptHelper
from langchain import OpenAI
from PyPDF2 import PdfReader
import os

# Initialize OpenAI API (replace "your-openai-api-key" with your OpenAI API key)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Initialize RAG-related components
def load_pdf_and_create_index(uploaded_file):
    # Save uploaded file to a directory
    with open(os.path.join("papers", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read the PDF
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    # Create nodes and index from the PDF text
    documents = [pdf_text]
    index = GPTSimpleVectorIndex.from_documents(documents)
    
    return index

def query_index(index, user_input):
    return index.query(user_input)

# Title of the app
st.title("Chatbot with RAG System (LlamaIndex)")

# Split the page into two columns
col1, col2 = st.columns([2, 3])

# Left column (Chatbot)
with col1:
    st.header("Chatbot")

    # Initialize session state for storing chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["index"] = None  # This will store the index from the PDF

    # Function to simulate a chatbot response
    def get_bot_response(user_input):
        # If an index is available, query it
        if st.session_state["index"]:
            response = query_index(st.session_state["index"], user_input)
            return response
        else:
            return "Please upload a PDF first!"

    # Input box for user to type their message
    user_input = st.text_input("You:", "")

    # If the user submits a message
    if user_input:
        # Add user's message to the chat history
        st.session_state.messages.append(f"You: {user_input}")

        # Get bot's response and add it to the chat history
        bot_response = get_bot_response(user_input)
        st.session_state.messages.append(f"Bot: {bot_response}")

    # Display the chat history
    for message in st.session_state.messages:
        st.write(message)

# Right column (PDF Viewer and PDF uploader)
with col2:
    st.header("PDF Viewer and Uploader")

    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Create index from uploaded PDF
        st.session_state["index"] = load_pdf_and_create_index(uploaded_file)
        st.success("PDF successfully uploaded and indexed!")

