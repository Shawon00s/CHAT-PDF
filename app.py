import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

# Initializes the page configuration
st.set_page_config(page_title="Chat with PDF", page_icon=":books:")

# To read the PDF files and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks using CharacterTextSplitter from langchain
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store using FAISS and HuggingFaceInstructEmbeddings
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large",
    model_kwargs={}
    )
    vectorstore = FAISS.from_texts(text=text_chunks, embedding = embeddings)
    return vectorstore

def main():
    load_dotenv()  # Load environment variables from .env file

    # Title of the app and providing text input area
    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask question about your PDFs: ")

    # Creating a sidebar for uploading files
    with st.sidebar:
        st.header("Upload your PDFs")
        pdf_docs = st.file_uploader("Upload PDF files here and click on the process button", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get the pdf texts
                raw_text = get_pdf_text(pdf_docs)  # Function to extract text from the uploaded PDFs

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)  # Function to split the text into chunks
                st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)  # Function to create a vector store using FAISS

if __name__ == '__main__':
    main()