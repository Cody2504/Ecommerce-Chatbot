from operator import itemgetter
from typing import Literal
from typing_extensions import TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, ListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain.chains import LLMRouterChain
from src.utils.custom_output_parser import CustomListOutputParser

def invoke_llm_with_vectorstore(llm, vectorstore, query):
    """Invokes the LLM with the relevant documentation retrieved from the vectorstore and returns only the content using a custom output parser."""
    # Retrieve the most relevant document from the vectorstore
    docs = vectorstore.as_retriever().invoke(query)
    context = docs[0].page_content if docs else "No relevant documentation found."

    # Create a prompt with the retrieved context
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for an e-commerce platform. Use the following context to answer the query."),
        ("system", "Context: {context}"),
        ("human", "{query}"),
    ])
    prompt = prompt_template.format(context=context, query=query)

    # Use custom output parser to return only the content as a list
    parser = CustomListOutputParser(separator="\n")
    result = llm.invoke(prompt)
    return parser.parse(str(result.content))


def route_to_doc_type(llm, query, doc_types):
    """Use LLM to classify the query into a doc_type (e.g., returns, faqs, ordering, etc.)."""
    system_msg = (
        "You are an e-commerce assistant. Classify the user's query into one of the following documentation types: "
        + ", ".join(doc_types) + ". "
        "Return only the most relevant type as a single word."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{query}"),
    ]).format(query=query)
    result = llm.invoke(prompt)
    return str(result.content).strip().lower()

def invoke_llm_with_vectorstore(llm, vectorstore, query):
    """Route query to the right doc_type, retrieve only relevant docs, and parse output."""
    # Define possible doc_types based on your corpus
    doc_types = ["returns", "refund", "faqs", "ordering", "products", "shipping", "common_issue"]
    doc_type = route_to_doc_type(llm, query, doc_types)
    # Retrieve only relevant docs for that doc_type
    docs = vectorstore.similarity_search(f"{doc_type} {query}", k=1)
    context = docs[0].page_content if docs else "No relevant documentation found."
    
    # Lấy thông tin về doc_type để trả về cho người dùng
    doc_info = None
    if docs:
        metadata = docs[0].metadata
        doc_source = metadata.get("source", "").split("/")[-1].split(".")[0] if "source" in metadata else doc_type
        doc_info = f"{doc_source}"
        if "product_name" in metadata:
            doc_info += f" - {metadata['product_name']}"
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for an e-commerce platform. Use the following context to answer the query."),
        ("system", "Context: {context}"),
        ("human", "{query}"),
    ])
    prompt = prompt_template.format(context=context, query=query)
    parser = CustomListOutputParser(separator="\n")
    result = llm.invoke(prompt)
    return parser.parse(str(result.content)), doc_info