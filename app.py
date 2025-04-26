import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Helper Functions
def get_pdf_text_from_dir(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            try:
                with open(os.path.join(directory, filename), "rb") as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the Questions as detailed as possible from the provided context. 
    Make sure to provide all the details. If the answer is not in the provided context,
    just say, "Answer is not in the provided context." Do not provide the wrong answer.
    Context:\n {context}?\n
    Question:\n{question}\n 
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error processing your question: {e}")


# Main Function
def main():
    st.set_page_config(page_title="Chat With PDFs from Directory", layout="wide")
    st.header("BIO-Forte")

    # if st.button("Process PDFs in 'data' directory"):
    #     with st.spinner("Reading and processing PDFs..."):
    #         try:
    #             raw_text = get_pdf_text_from_dir("./data")
    #             if raw_text.strip():
    #                 text_chunks = get_text_chunks(raw_text)
    #                 get_vector_store(text_chunks)
    #                 st.success("PDFs processed successfully! You can now ask questions.")
    #             else:
    #                 st.warning("No text could be extracted from the PDFs.")
    #         except Exception as e:
    #             st.error(f"Error during PDF processing: {e}")

    user_question = st.text_input("Ask a question ")
    if user_question:
        user_input(user_question)


# Run the Streamlit App
if __name__ == "__main__":
    main()
