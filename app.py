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
import base64
import time 

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
    
    # BATCHING
    
    # 1. Khởi tạo thanh tiến trình (Progress Bar) trên UI
    progress_text = f"Đang mã hóa {len(text_chunks)} đoạn văn bản ..."
    my_bar = st.progress(0, text=progress_text)
    
    # 2. Kích thước mỗi Batch
    batch_size = 50 # Xử lý 50 đoạn mỗi lần gửi
    
    # 3. Khởi tạo kho Vector với lô đầu tiên
    vector_store = FAISS.from_texts(text_chunks[:batch_size], embedding=embeddings)
    
    # 4. Vòng lặp xử lý các lô còn lại với
    for i in range(batch_size, len(text_chunks), batch_size):
        batch = text_chunks[i : i + batch_size]
        
        # Cho hệ thống ngủ 8 giây
        time.sleep(8) 
        
        # Nhúng lô tiếp theo vào kho chứa
        vector_store.add_texts(batch)
        
        # Tính toán % hoàn thành và cập nhật thanh tiến trình
        percent_complete = min((i + batch_size) / len(text_chunks), 1.0)
        my_bar.progress(percent_complete, text=progress_text)
        
    # 5. Lưu kho vector và dọn dẹp giao diện
    vector_store.save_local("faiss_index")
    my_bar.empty() # Ẩn thanh tiến trình khi hoàn tất
    
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
    
    return response["output_text"]

def display_pdf(uploaded_file):
    # Đọc file và mã hóa sang dạng base64
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    
    pdf_display = f'''
    <embed src="data:application/pdf;base64,{base64_pdf}" 
           width="100%" 
           height="600" 
           type="application/pdf">
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)
    

def main():
    st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="wide")
    
    # CSS Tùy chỉnh giao diện (Đã bao gồm ép User sang phải bằng !important)
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;} 
        .stButton>button {
            background-color: #4CAF50; 
            color: white; 
            border-radius: 8px; 
            width: 100%;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
        }
        
        /* --- TÙY CHỈNH KHUNG CHAT TRÁI/PHẢI --- */
        div[data-testid="stChatMessage"]:has(.user-tag) {
            flex-direction: row-reverse !important;
            text-align: right !important;
        }
        
        /* Đổi màu nền bong bóng chat của User */
        div[data-testid="stChatMessage"]:has(.user-tag) div[data-testid="stChatMessageContent"] {
            background-color: #d6eaf8 !important;
            color: black !important;
            border-radius: 15px !important;
            padding: 10px 15px !important;
        }
        
        /* Đổi màu nền bong bóng chat của AI */
        div[data-testid="stChatMessage"]:has(.ai-tag) div[data-testid="stChatMessageContent"] {
            background-color: #f1f3f4 !important;
            color: black !important;
            border-radius: 15px !important;
            padding: 10px 15px !important;
        }
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("🎓 Trợ lý Gia sư AI - Đọc tài liệu")

    # --- KHU VỰC SIDEBAR ---
    with st.sidebar:
        st.title("📂 Quản lý tài liệu")
        docs = st.file_uploader("Upload tài liệu (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if st.button("Xử lý tài liệu"):
            if docs:
                with st.spinner("Gia sư đang đọc tài liệu..."):
                    raw_text = get_document_text(docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Đã học xong tài liệu! Bạn có thể bắt đầu hỏi.")
            else:
                st.warning("Vui lòng upload tài liệu trước!")

    # --- CHIA ĐÔI MÀN HÌNH ---
    col1, col2 = st.columns([1, 1])

    # --- CỘT 1: HIỂN THỊ TÀI LIỆU ---
    with col1:
        st.subheader("📄 Xem trước tài liệu")
        if docs:
            # Menu chọn file để xem
            selected_file = st.selectbox("Chọn file để xem:", [doc.name for doc in docs])
            for doc in docs:
                if doc.name == selected_file:
                    if doc.name.endswith(".pdf"):
                        display_pdf(doc)
                    else:
                        st.info("Trình duyệt chỉ hiển thị bản gốc của PDF. Dưới đây là nội dung văn bản được trích xuất:")
                        st.text_area("", get_document_text([doc]), height=550)
        else:
            st.info("Vui lòng tải tài liệu lên từ thanh bên trái để xem trước.")

   # --- CỘT 2: GIAO DIỆN CHAT CÓ THANH CUỘN ---
    with col2:
        st.subheader("💬 Chat với Gia sư")
        
        chat_container = st.container(height=550, border=True)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 1. In lại tin nhắn CŨ
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        # Nhúng tag ẩn cho User
                        st.markdown(f"<span class='user-tag'></span> {message['content']}", unsafe_allow_html=True)
                    else:
                        # Nhúng tag ẩn cho AI
                        st.markdown(f"<span class='ai-tag'></span> {message['content']}", unsafe_allow_html=True)

        # 2. Xử lý tin nhắn MỚI
        if prompt := st.chat_input("Hãy đặt câu hỏi chi tiết về tài liệu của bạn..."):
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(f"<span class='user-tag'></span> {prompt}", unsafe_allow_html=True)

                with st.chat_message("assistant"):
                    with st.spinner("Gia sư đang suy nghĩ..."):
                        response = user_input(prompt) 
                        st.markdown(f"<span class='ai-tag'></span> {response}", unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
