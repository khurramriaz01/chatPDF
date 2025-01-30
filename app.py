import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = None
if "summary" not in st.session_state:
    st.session_state.summary = None

@st.cache_data
def process_single_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

def process_all_pdfs(uploaded_files):
    all_chunks = []
    for file in uploaded_files:
        chunks = process_single_pdf(file)
        all_chunks.extend(chunks)
    
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = np.array(embedder.encode(all_chunks))
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    
    return all_chunks, index

def generate_summary(chunks):
    combined_text = "\n".join(chunks)
    prompt = f"""Generate a comprehensive summary of the document(s) covering:
    - Key topics and themes
    - Important findings or conclusions
    - Notable data points or statistics
    - Overall purpose and scope
    
    Document content: {combined_text[:15000]}... [truncated]"""
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
        )
    )
    return response.text

def chat_interface():
    st.header("Chat with your PDFs")
    st.markdown("---")
    
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about the documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            try:
                embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                query_embedding = embedder.encode([prompt])[0]
                _, indices = st.session_state.index.search(np.array([query_embedding], dtype=np.float32), k=3)
                retrieved_chunks = " ".join([st.session_state.chunks[i] for i in indices[0]])
                
                prompt_text = f"""Context: {retrieved_chunks}
                Question: {prompt}
                - Provide detailed answer based on context
                - Acknowledge if information is missing
                - Use Feynman technique when requested"""
                
                response = model.generate_content(
                    prompt_text,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1024,
                    )
                )
                
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                        
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

def summary_interface():
    st.header("Document Summary")
    st.markdown("---")
    
    if not st.session_state.summary:
        with st.spinner("Generating summary..."):
            st.session_state.summary = generate_summary(st.session_state.chunks)
    
    summary_container = st.container(height=500)
    with summary_container:
        st.markdown(st.session_state.summary)
    
    if st.button("Regenerate Summary"):
        with st.spinner("Updating summary..."):
            st.session_state.summary = generate_summary(st.session_state.chunks)
        st.rerun()

def main():
    st.set_page_config(page_title="PDF Insight AI", page_icon="📘", layout="wide")
    
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploader"
        )
        
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                chunks, index = process_all_pdfs(uploaded_files)
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.uploaded_files = uploaded_files
                st.session_state.mode = None  # Reset mode on new upload
                st.session_state.summary = None  # Reset summary
            st.success(f"Processed {len(uploaded_files)} PDF(s)!")
        
        if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            st.markdown("**Uploaded Documents**")
            for file in st.session_state.uploaded_files:
                st.markdown(f"- {file.name}")
            
            if st.session_state.mode == "chat":
                if st.button("Switch to Summary"):
                    st.session_state.mode = "summary"
                    st.rerun()
            elif st.session_state.mode == "summary":
                if st.button("Switch to Chat"):
                    st.session_state.mode = "chat"
                    st.rerun()
    
    with right_col:
        if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            if not st.session_state.mode:
                st.header("Select Mode")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate Summary", use_container_width=True):
                        st.session_state.mode = "summary"
                        st.rerun()
                with col2:
                    if st.button("Start Chatting", use_container_width=True):
                        st.session_state.mode = "chat"
                        st.rerun()
            
            if st.session_state.mode == "chat":
                chat_interface()
            elif st.session_state.mode == "summary":
                summary_interface()

if __name__ == "__main__":
    main()