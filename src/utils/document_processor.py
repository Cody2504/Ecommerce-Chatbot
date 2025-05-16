from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma
import os
import re

def get_document_type(file_path):
    """Xác định loại tài liệu dựa trên tên file."""
    file_name = os.path.basename(file_path).lower()
    file_name = os.path.splitext(file_name)[0]  # Remove extension
    return file_name

def extract_metadata(text, doc_type):
    """Trích xuất metadata từ nội dung tài liệu."""
    metadata = {"doc_type": doc_type}
    
    # Trích xuất các thông tin thêm dựa vào loại tài liệu
    if doc_type == "products":
        # Tìm tên sản phẩm
        product_match = re.search(r"Product Name: (.+)", text)
        if product_match:
            metadata["product_name"] = product_match.group(1)
            
        # Tìm thương hiệu
        brand_match = re.search(r"Brand: (.+)", text)
        if brand_match:
            metadata["brand"] = brand_match.group(1)
            
        # Tìm giá
        price_match = re.search(r"Price: (.+)", text)
        if price_match:
            metadata["price"] = price_match.group(1)
    
    elif doc_type == "faqs":
        # Đánh dấu là FAQ để dễ tìm kiếm
        metadata["content_type"] = "question_answer"
        
        # Tìm cụ thể câu hỏi nếu có
        q_match = re.search(r"Q: (.+?)\nA:", text)
        if q_match:
            metadata["question"] = q_match.group(1)
    
    elif doc_type in ["shipping", "returns", "ordering"]:
        # Đánh dấu là policy
        metadata["content_type"] = "policy"
        metadata["policy_type"] = doc_type
    
    elif doc_type == "common_issue":
        # Đánh dấu là issue
        metadata["content_type"] = "issue_solution"
        
        # Tìm tên vấn đề
        issue_match = re.search(r"Issue: (.+?)\nSolution:", text)
        if issue_match:
            metadata["issue"] = issue_match.group(1)
        
    return metadata

def get_text_splitter(doc_type):
    """Trả về text splitter phù hợp với loại tài liệu."""
    if doc_type == "products":
        # Sử dụng RecursiveCharacterTextSplitter cho sản phẩm để tách theo từng sản phẩm
        return RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\nProduct Name:", "\n\n"]
        )
    elif doc_type == "faqs":
        # Sử dụng RecursiveCharacterTextSplitter cho FAQs để tách theo Q&A
        return RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\nQ:", "\n\n"]
        )
    elif doc_type == "common_issue":
        # Tách theo các vấn đề phổ biến
        return RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["Issue:", "\n\n"]
        )
    else:
        # Mặc định dùng CharacterTextSplitter với kích thước nhỏ hơn
        return CharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=30
        )

def chunk_document(document, doc_type):
    """Chia nhỏ tài liệu thành các phần và giữ nguyên metadata."""
    text_splitter = get_text_splitter(doc_type)
    chunks = text_splitter.split_text(document.page_content)
    
    # Thêm metadata cho mỗi chunk
    docs_with_metadata = []
    for i, chunk in enumerate(chunks):
        # Lấy metadata từ document gốc và thêm thông tin về chunk
        metadata = document.metadata.copy() if hasattr(document, 'metadata') else {}
        metadata.update(extract_metadata(chunk, doc_type))
        metadata["chunk_id"] = i
        
        # Tạo Document mới với metadata đầy đủ
        from langchain_core.documents import Document
        docs_with_metadata.append(Document(page_content=chunk, metadata=metadata))
        
    return docs_with_metadata

def process_documents(docs):
    """Xử lý danh sách tài liệu, chia nhỏ chúng và thêm metadata."""
    processed_docs = []
    
    for doc in docs:
        # Lấy loại tài liệu từ tên file
        doc_type = get_document_type(doc.metadata.get("source", "unknown"))
        
        # Thêm doc_type vào metadata
        if not hasattr(doc, 'metadata'):
            doc.metadata = {}
        doc.metadata["doc_type"] = doc_type
        
        # Chia nhỏ tài liệu
        chunked_docs = chunk_document(doc, doc_type)
        processed_docs.extend(chunked_docs)
    
    return processed_docs

def create_optimized_vectorstore(docs, embeddings, persist_directory):
    """Tạo và lưu trữ vector store với các tài liệu đã được xử lý."""
    processed_docs = process_documents(docs)
    return Chroma.from_documents(
        documents=processed_docs, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
