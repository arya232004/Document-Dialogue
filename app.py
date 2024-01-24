from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai 
from langchain.vectorstores import FAISS 
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def getpdf(pdf):
    text=""
    for pdf_no in pdf:
        pdf_read=PdfReader(pdf_no)
        for page in pdf_read.pages:
            text+=page.extract_text()
    return text                    

def get_chunks(data):
    text_split=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_split.split_text(data)
    return chunks
    
def get_vector(text_chunk):
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store=FAISS.from_texts(text_chunk, embedding=embeddings)
    vector_store.save_local('vector_store')
    

def get_converstaion_chain(): 
    prompt_template=""" Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    
    
    # if the answer is not in provided conext just say answer is not in the context, dont provide the wrong answer.
    # Context:\n {context}> \n
    # Question: {question} \n
    
    # Answer: 
    
    # """
    model=ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)
    prompt=PromptTemplate(template=prompt_template, input_variables=['context','question'])
    chain=load_qa_chain(model,chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("vector_store", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_converstaion_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF semalessly")

    user_question = st.text_input("Ask a Question from the PDF")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = getpdf(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector(text_chunks)
                st.success("Done")
                
if __name__ == "__main__":
    main()                        