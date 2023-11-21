import base64

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=HuggingFaceEmbeddings(model_name = embedding_model_name)
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorstore.as_retriever(),memory=memory)
    return conversation_chain

def generate_dataset(user_prompt):
    response=st.session_state.conversation({'question':user_prompt})
    dataset = response['chat_history']
    # Extract the 'content' field from the dataset
    content = [message.content for message in dataset]
    # Convert the content to a pandas DataFrame
    df = pd.DataFrame(content, columns=['content'])
    # Save the DataFrame to a CSV file
    df.to_csv('dataset.csv', index=False)
    # Download the CSV file
    st.markdown(get_table_download_link(df), unsafe_allow_html=True)

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    href = f'<a href="data:file/csv;base64,{b64}" download="dataset.csv">Download CSV File</a>'
    return href

def main():
    load_dotenv()
    st.set_page_config("DataGenerator-LangChain", page_icon=":books:")
    st.header("Generate your dataset :brain:")
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    user_prompt = "make a pair of input and output"
    pdf_docs = st.file_uploader("Upload the PDF Files here", accept_multiple_files=True)
    st.markdown('''
    - Streamlit
    - LangChain
    - OpenAI
    ''')
    if st.button('Process'):
        with st.spinner("Processing"):
            # Extract Text from PDF
            raw_text = get_pdf_text(pdf_docs)
            # Split the Text into Chunks
            text_chunks = get_text_chunks(raw_text)
            # Create Vector Store
            vectorstore=get_vector_store(text_chunks)
            # Create Conversation Chain
            st.session_state.conversation=get_conversation_chain(vectorstore)
            # Generate Dataset
            generate_dataset(user_prompt)
            st.success("Done! Click the link above to download your dataset.")

if __name__ == "__main__":
    main()