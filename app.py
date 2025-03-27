import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

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
                st.write(raw_text) # Display the raw text
            # get the text chunks

            # create vector store

if __name__ == '__main__':
    main()