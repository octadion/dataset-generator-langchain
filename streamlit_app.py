import base64
import json
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import CSVLoader
import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')

def get_pdf_text(docs):
    if not isinstance(docs, list):
        pdf_reader = PdfReader(docs)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        text=""
        for pdf in docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text+=page.extract_text()
            return text
def get_csv_doc(docs):
    if docs is not None:
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open(os.path.join('temp', docs.name), 'wb') as f:
            f.write(docs.getbuffer())
        loader = CSVLoader(file_path=os.path.join('temp', docs.name), encoding="utf-8",
                           csv_args={'delimiter': ','})
        document = loader.load()
        return document
def get_text_chunks_pdf(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_text_chunks_csv(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks = text_splitter.split_documents(document)
    return chunks

def get_vector_store_pdf(text_chunks):
    embeddings=OpenAIEmbeddings()
    # embeddings=HuggingFaceEmbeddings(model_name = embedding_model_name)
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def get_vector_store_csv(text_chunks):
    embeddings=OpenAIEmbeddings()
    # embeddings=HuggingFaceEmbeddings(model_name = embedding_model_name)
    vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorstore.as_retriever(),memory=memory)
    return conversation_chain

def generate_dataset(user_prompt):
    response=st.session_state.conversation({'question':user_prompt})
    print(response)
    dataset = response['chat_history']
    content = [message.content for message in dataset]
    df = pd.DataFrame(content, columns=['content'])
    df_answer = df.iat[1, 0]
    answer_json = json.loads(df_answer.replace('\n', ''))
    with open('dataset.json', 'w') as f:
        json.dump(answer_json, f, indent=4)
    st.markdown(get_table_download_link(), unsafe_allow_html=True)

def get_table_download_link():
    with open('dataset.json', 'r') as f:
        json_data = f.read()
    b64 = base64.b64encode(json_data.encode()).decode()  # some strings
    href = f'<a href="data:file/json;base64,{b64}" download="dataset.json">Download JSON File</a>'
    return href

def main():
    user_prompt = ""
    load_dotenv()
    st.set_page_config("DataGenerator-LangChain", page_icon=":books:")
    st.header("Generate your dataset :brain:")
    dataset_count = st.number_input("Number of datasets to generate", min_value=3, max_value=1000)
    options = {
        "QnA": "Membuat QnA (Pertanyaan dan Jawaban)",
        "Summarization": "Membuat Rangkuman pasal atau sebuah topik pada pasal",
        "Element Extraction": "Melakukan Ekstraksi elemen atau identifikasi pasal",
        "Similarity Search": "Menemukan kesamaan pasal atau sebuah topik pasal, atau menemukan tumpang tindih pasal",
        "Revision": "Memberikan saran Revisi pasal berdasarkan menimbang dan mengingat pasal terkait serta berdasarkan pasal terdahulu",
        "Draft": "Membuat Draft pasal baru berdasarkan menimbang dan mengingat pasal terkait serta berdasarkan pasal terdahulu",
    }
    choice = st.selectbox("Task to generate:", list(options.keys()))
    detail_text = options[choice]
    st.write(f"Task: {detail_text}")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if choice == "Similarity Search" or choice == "Revision" or choice == "Draft":
        docs = st.file_uploader("Upload the PDF Files here", accept_multiple_files=True)
        if docs:
            main_document = docs[0]
            main_doc_name = main_document.name.replace('.pdf', '')
            other_documents = docs[1:]
            other_doc_names = [doc.name.replace('.pdf', '') for doc in other_documents]
            user_prompt = f"Anda adalah model yang mengubah isi teks menjadi berbagai tugas hukum dalam format JSON. " \
                      "Setiap JSON berisi ‘reference’ (tulis bunyi pasal dan ayat, serta nomor PP dan tahun), ‘instruction’ (instruksi atau pertanyaan), dan ‘output’ (jawaban). " \
                      f"Hanya merespons dengan JSON dan tanpa teks tambahan. Tugas dapat berupa {detail_text}, dengan {main_doc_name} sebagai dokumen utama, sedangkan {other_doc_names} sebagai dokumen pendukung. Hasilkan sebanyak {dataset_count} data. Pastikan setiap pertanyaan dan jawaban unik dan tidak berulang. \n"
    else:
        docs = st.file_uploader("Upload the PDF/CSV Files here")
        user_prompt = f"Anda adalah model yang mengubah isi teks menjadi berbagai tugas hukum dalam format JSON. " \
                      "Setiap JSON berisi ‘reference’ (tulis bunyi pasal dan ayat, serta nomor PP dan tahun), ‘instruction’ (instruksi atau pertanyaan), dan ‘output’ (jawaban). " \
                      f"Hanya merespons dengan JSON dan tanpa teks tambahan. Tugas dapat berupa {detail_text}. Hasilkan sebanyak {dataset_count} data. Pastikan setiap pertanyaan dan jawaban unik dan tidak berulang. \n"

    st.markdown('''
    - Streamlit
    - LangChain
    - OpenAI
    ''')
    if st.button('Process'):
        with st.spinner("Processing"):
            if docs.name.endswith(".pdf"):
                raw_text = get_pdf_text(docs)
                text_chunks = get_text_chunks_pdf(raw_text)
                vectorstore = get_vector_store_pdf(text_chunks)
            elif docs.name.endswith(".csv"):
                raw_text = get_csv_doc(docs)
                text_chunks = get_text_chunks_csv(raw_text)
                vectorstore = get_vector_store_csv(text_chunks)
            st.session_state.conversation=get_conversation_chain(vectorstore)
            # Generate Dataset
            generate_dataset(user_prompt)
            st.success("Done! Click the link above to download your dataset.")

if __name__ == "__main__":
    main()