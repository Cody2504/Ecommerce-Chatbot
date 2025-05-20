from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from src.utils.custom_output_parser import CustomListOutputParser

def create_few_shot_prompt_template():
    """
    Creates a few-shot prompt template for the e-commerce customer support chatbot.
    This approach provides examples of good responses to help guide the model.
    """
    examples = [
        {
            "query": "How do I return a product?",
            "context": "Our return policy allows customers to return products within 14 days of delivery. The product must be in its original packaging and condition. To initiate a return, log into your account, go to order history, select the item, and click 'Return Item'. A return shipping label will be provided for your convenience.",
            "answer": "You can return a product within 14 days of delivery. The item must be in its original packaging and condition. To start the return process:\n1. Log into your account\n2. Go to your order history\n3. Select the specific item\n4. Click 'Return Item'\n\nA return shipping label will be provided for you. If you need further assistance, please contact our customer service team."
        },
        {
            "query": "When will my order arrive?",
            "context": "Standard shipping typically takes 3-5 business days for domestic orders. Express shipping is available for an additional fee and delivers within 1-2 business days. International shipping may take 7-14 business days depending on the destination country. Once your order is shipped, you will receive a tracking number via email.",
            "answer": "Your order delivery time depends on the shipping method you selected:\n- Standard shipping: 3-5 business days (domestic)\n- Express shipping: 1-2 business days (domestic)\n- International shipping: 7-14 business days\n\nYou'll receive a tracking number by email once your order ships. You can also check your order status in your account dashboard."
        },
        {
            "query": "Do you have the SmartWatch Pro V3 in stock?",
            "context": "Product Name: SmartWatch Pro V3\nBrand: TechZone\nPrice: $89.00\nWarranty: 18 months\nFeatures: Heart rate monitor, sleep tracking, water resistant, Bluetooth\nCompatible Devices: Android & iOS\nReturnable: Within 7 days (if unopened)\nIn Stock: Limited stock available",
            "answer": "Yes, the SmartWatch Pro V3 is currently in stock, but with limited availability. This $89.00 smartwatch from TechZone features heart rate monitoring, sleep tracking, water resistance, and Bluetooth connectivity. It's compatible with both Android and iOS devices and comes with an 18-month warranty. If you're interested in purchasing, I'd recommend doing so soon while supplies last."
        }
    ]

    # Create the few-shot prompt template
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "Question: {query}"),
        ("human", "Context: {context}"),
        ("ai", "{answer}")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        input_variables=["query", "context"]
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledgeable e-commerce customer support assistant. Answer questions based on the provided context. Be helpful, concise, and friendly."),
        few_shot_prompt,
        ("human", "Question: {query}\nContext: {context}")
    ])

    return final_prompt


def invoke_retrieval_qa_chain(llm, vectorstore, query):
    from langchain_core.prompts import PromptTemplate
    
    template = """You are a knowledgeable e-commerce customer support assistant. Answer questions based on the provided context. Be helpful, concise, and friendly.

                Context: {context}

                Question: {question}

                Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  
        retriever=vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={
                "k": 1,         
                "fetch_k": 10,  
                "lambda_mult": 0.7  
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        },
        input_key="question"  
    )
    
    result = qa_chain.invoke({"question": query})
    
    answer = result.get("result", "I couldn't find relevant information to answer your question.")
    source_docs = result.get("source_documents", [])
    
    source_info = None
    if source_docs:
        doc = source_docs[0]
        metadata = doc.metadata
        source = metadata.get("source", "").split("/")[-1].split(".")[0] if "source" in metadata else "unknown"
        product_name = metadata.get("product_name", "")
        
        source_info = source
        if product_name:
            source_info += f" - {product_name}"
    
    parser = CustomListOutputParser(separator="\n")
    try:
        parsed_answer = parser.parse(answer)
        return parsed_answer, source_info
    except Exception:
        return answer, source_info

