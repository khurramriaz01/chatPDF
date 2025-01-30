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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_data
def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = np.array(embedder.encode(chunks))
    
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    
    return chunks, index



def main():
    st.set_page_config(page_title="PDF Insight AI", page_icon="📘", layout="wide")
    
    # Split layout into two columns (1:3 ratio)
    left_col, right_col = st.columns([1, 3])
    
    with right_col:
        st.header("Chat with your PDF")
        st.markdown("---")
        
        # Chat container
        chat_container = st.container(height=500)
        
        # Display chat messages
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # User input at bottom
        if prompt := st.chat_input("Ask about the document..."):
            if "index" not in st.session_state:
                st.error("Please upload a PDF document first!")
                return
                
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant chunks
                    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                    query_embedding = embedder.encode([prompt])[0]
                    _, indices = st.session_state.index.search(np.array([query_embedding], dtype=np.float32), k=3)
                    retrieved_chunks = " ".join([st.session_state.chunks[i] for i in indices[0]])
                    
                    # Generate response
                    prompt_text = f"""Context: {retrieved_chunks}
                    Question: {prompt}
                    - You are greatest teacher of all time.
                    - Please provide a detailed answer based on the context above. 
                    - If the context doesn't contain relevant information, please indicate that.
                    - Use feynman technique only when user asks to explain any concepts."""
                    
                    response = model.generate_content(
                        prompt_text,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3,
                            max_output_tokens=1024,
                        )
                    )
                    
                    # Add AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(response.text)
                            
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    
    with left_col:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="uploader")
        
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                chunks, index = process_pdf(uploaded_file)
                st.session_state.chunks = chunks
                st.session_state.index = index
            st.success("PDF processed!")
            
        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown("""
        1. Upload a PDF document
        2. Ask questions in the chat
        3. Receive AI-powered answers
        """)


if __name__ == "__main__":
    main()