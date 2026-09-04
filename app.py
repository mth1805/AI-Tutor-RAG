import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai
import docx

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
# Tải API key từ file .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_document_text(docs):
    text = ""
    for doc in docs:
        if doc.name.endswith(".pdf"):
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif doc.name.endswith(".docx"):
            doc_file = docx.Document(doc)
            for para in doc_file.paragraphs:
                text += para.text + "\n"
        elif doc.name.endswith(".txt"):
            text += doc.getvalue().decode("utf-8") + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Bạn là một Gia sư AI tận tâm dành cho sinh viên đại học. 
    Dựa vào các đoạn Tài liệu được cung cấp dưới đây, hãy trả lời Câu hỏi của sinh viên.
    QUY TẮC BẮT BUỘC: 
    1. Tuyệt đối KHÔNG đưa ra đáp án cuối cùng ngay lập tức.
    2. Hãy đưa ra gợi ý, hướng dẫn từng bước để sinh viên tự tìm ra câu trả lời.
    3. Nếu câu hỏi không nằm trong tài liệu, hãy nói: "Tài liệu bài giảng không đề cập đến vấn đề này, nhưng theo tôi..."
    
    Tài liệu: \n {context}?\n
    Câu hỏi: \n{question}\n
    
    Gia sư trả lời:
    """
    model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("🤖 **Gia sư AI:** ", response["output_text"])

def main():
    st.set_page_config(page_title="AI Tutor Pro", page_icon="🎓", layout="wide")
    
    # CSS Tùy chỉnh giao diện
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;} /* Ẩn menu mặc định của Streamlit */
        footer {visibility: hidden;} /* Ẩn chữ 'Made with Streamlit' */
        .stButton>button {
            background-color: #4CAF50; 
            color: white; 
            border-radius: 8px; 
            width: 100%;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("🎓 Trợ lý Gia sư AI - Đọc tài liệu")

    user_question = st.text_input("Sinh viên: Hỏi câu hỏi liên quan đến tài liệu tại đây...")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Kho tài liệu")
        docs = st.file_uploader("Upload tài liệu (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if st.button("Xử lý tài liệu"):
            with st.spinner("Đang học tài liệu..."):
                raw_text = get_document_text(docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Đã học xong! Bạn có thể đặt câu hỏi.")

if __name__ == "__main__":
    main()