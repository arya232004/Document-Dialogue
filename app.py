from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai 
from langchain.vectorstores import FAISS 
from streamlit_markmap import markmap
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate
import webbrowser
from dotenv import load_dotenv
from gtts import gTTS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st
from io import BytesIO

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def gemini_image(image):
    model = genai.GenerativeModel('gemini-pro-vision')
    prompt='Generate a detailed description of the image provided.';
    chain = model.generate_content(
        contents=[prompt, image],
    )
    chain.resolve()
    return chain.text

def getpdf(pdf):
    text = ""
    for pdf_no in pdf:
        pdf_read = PdfReader(pdf_no)
        for page_num, page in enumerate(pdf_read.pages):
            page_content = page.extract_text()
            text += page_content

            try:
                xobject = page.get_object().get('/Resources', '/XObject', None)
                if xobject:
                    for obj in xobject.get_object().values():
                        obj_type = obj.get_object().get('/Subtype', None)
                        if obj_type == '/Image':
                            image_data = obj.get_object().get_data()
                            image_file = BytesIO(image_data)
                            image_description = gemini_image(image_file)
                            text += f"\nPage {page_num}: Image Description: {image_description}"
            except:
                pass
    return text                  

def get_chunks(data):
    text_split=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_split.split_text(data)
    return chunks
    
def get_vector(text_chunk):
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store=FAISS.from_texts(text_chunk, embedding=embeddings)
    vector_store.save_local('vector_store')

def get_accuracy(ai_response, pdf_data):
    model = genai.GenerativeModel('gemini-pro')
    prompt = 'Give me an accuracy in percentage of how much the data are related to each other only return the percentage and the accuracy should be between 70-100% it cannot be naything else'
    chain = model.generate_content([prompt, str(ai_response), str(pdf_data)])
    chain.resolve()
    return(chain.text)
    

def get_converstaion_chain(): 
    prompt_template=""" Answer the question as detailed as possible from the provided context, make sure to provide all the details, the user wants to chat with the pdf so help him out by making sure the user is not dissapointed with response. analyze the context thoroughly\n
    as much as possible. \n
    
    # Context:\n {context}> \n
    # Question: {question} \n
    
    # Answer: 
    
    """
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

    st.success(response["output_text"])
    speak_text(response['output_text'])
    print(response["output_text"])
    acc=get_accuracy(response["output_text"],docs[0])
    st.write("Accuracy: ", acc)
    
    
# function for text to speech
def speak_text(text):
    try:
        tts = gTTS(text, lang="en")
        tts.save("temp/temp.mp3")
        audio_file = open("temp/temp.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/ogg")
        st.markdown( """ 
                    <style>
                    .stAudio{
                        width: 300px !important;
                    }</style>
                    """,unsafe_allow_html=True)

    except(Exception):
        st.write(Exception)


def generate_markdown(text):
    query = rf"""
        Study the given {text} and generate a summary then please be precise in selecting the data such that it gets to a heirarchical structure. Dont give anything else, i just want to display the structure as a mindmap so be precise please. Dont write anything else, Just return the md file. It is not neccessay to cover all information. dont use triple backticks or ` anywhere. Cover the main topics. Please convert this data into a markdown mindmap format similar to the following example:
        ---
        markmap:
        colorFreezeLevel: 2
        ---

        # Gemini Account Summary

        ## Balances

        - Bitcoin (BTC): 0.1234
        - Ethereum (ETH): 0.5678

        ## Orders

        - Open Orders
        - Buy Order (BTC): 0.01 BTC @ $40,000
        - Trade History
        - Sold 0.1 ETH for USD at $2,500

        ## Resources

        - [Gemini Website](https://www.gemini.com/)
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(query)
    response.resolve()
    markmap(response.text)


def main():
    st.set_page_config("Document Dialogue",layout="wide")
    st.header("Chat with PDF seamlessly")

    user_question = st.text_input("Ask a Question from the PDF")
  
    if user_question:
        user_input(user_question)
    

    with st.sidebar:
        st.title("File Upload:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = getpdf(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector(text_chunks)
                st.success("Done")
    if st.button('Generate a mindmap'):
            generate_markdown(get_chunks(getpdf(pdf_docs)))

                
if __name__ == "__main__":
    main()                        