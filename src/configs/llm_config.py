import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
import google.auth
from dotenv import load_dotenv

credentials, project_id = google.auth.default()

load_dotenv()
def get_embeddings_model():
    """Returns the embeddings model configuration."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

def get_chat_model():
    """Returns the chat model configuration."""
    
    return ChatVertexAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=512,
        max_retries=6,
        project=os.getenv("PROJECT_ID"),
        location="us-central1"
    )