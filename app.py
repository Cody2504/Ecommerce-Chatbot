import streamlit as st
from src.configs.llm_config import get_chat_model, get_embeddings_model
from src.utils.vectorstore_utils import create_vectorstore, load_documents
from src.chains.llm_route_chain import invoke_llm_with_vectorstore
import os
from dotenv import load_dotenv

st.set_page_config(
    page_title="E-commerce Support Bot",
    page_icon="🛍️",
    layout="centered"
)

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

st.title("🛍️ E-commerce Support Chatbot")

@st.cache_resource
def initialize_resources():
    """Initialize and cache the chat model, embeddings model, and vectorstore."""
    chat_model = get_chat_model()
    embeddings_model = get_embeddings_model()
    docs = load_documents("data")
    vectorstore = create_vectorstore(docs, embeddings_model, "chroma_vectorstore")
    return chat_model, vectorstore

chat_model, vectorstore = initialize_resources()

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hello! I'm your support assistance. How can I help you?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list):
            for item in message["content"]:
                st.write(item)
        else:
            st.write(message["content"])

if prompt := st.chat_input("Question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Processing...")
        
        try:
            response, doc_info = invoke_llm_with_vectorstore(chat_model, vectorstore, prompt)
            
            message_placeholder.empty()
            if isinstance(response, list):
                for item in response:
                    st.write(item)
            else:
                st.write(response)
                
            if doc_info:
                st.caption(f"*Thông tin dựa trên tài liệu: {doc_info}*")
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            message_placeholder.error(f"Lỗi xảy ra: {str(e)}")
            import traceback
            st.error(traceback.format_exc())