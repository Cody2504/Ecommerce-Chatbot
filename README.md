# E-commerce Customer Support Chatbot

![E-commerce Support Chatbot](https://img.shields.io/badge/E--commerce-Support%20Chatbot-blue)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![Google Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash--8B-purple)


A customer support chatbot for e-commerce platforms, using Google Gemini 2.0 Flash, Streamlit, LangChain, and LangSmith to retrieve relevant information from a knowledge base about e-commerce products, ordering, shipping, returns, and generate helpful responses to customer queries.

## Key Features

- **Intelligent Queries**: Using large language models to understand and respond to customer questions
- **Information Retrieval**: Searching for relevant information from the database about products, shipping policies, and return policies
- **Query Classification**: Automatically classifying questions to route to the appropriate information source
- **Smart Document Chunking**: Custom chunking strategies based on document type:
  - Products: Chunked by product entries with product name as separator
  - FAQs: Chunked by Q&A pairs for precise answers
  - Common Issues: Split by issue-solution blocks
  - Policies: Optimized chunking for shipping/returns documents
- **Metadata Extraction**: Automatically extracting product names, brands, prices, and other relevant information
- **User-friendly Interface**: Easy-to-use chat interface built with Streamlit
- **Performance Monitoring**: Integration with LangSmith for tracking and evaluating LLM calls

## Project Structure

```
LangChainECommerce/
├── app.py                   # Main Streamlit application
├── data/                    # Text data for knowledge base
│   ├── products.txt         # Product information
│   ├── shipping.txt         # Shipping policies
│   ├── returns.txt          # Return policies
│   ├── faqs.txt             # Frequently asked questions
│   └── common_issue.txt     # Common issues and solutions
├── src/
│   ├── chains/              # LangChain chains
│   │   └── llm_route_chain.py  # Query routing and document retrieval logic
│   ├── configs/
│   │   └── llm_config.py    # LLM configuration
│   ├── models/              # Models and configurations
│   │   └── llm_config.py    # Configuration for ChatVertexAI and GoogleGenerativeAIEmbeddings
│   └── utils/
│       ├── custom_output_parser.py  # Custom output parser
│       ├── document_processor.py    # Document processing and chunking
│       └── vectorstore_utils.py     # Document loading and vectorstore functions
└── chroma_vectorstore/      # Vectorstore storage directory (gitignored)
```

## Approach

The project uses retrieval methods to improve response quality:

1. **LLM-based Routing**: Using LLM to classify queries into appropriate document types
2. **Metadata Extraction**: Extracting information from text to improve source attribution

## Installation

### Requirements

- Python 3.8+
- Google API key or Google Cloud Platform credentials

### Setup Steps

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd LangChainECommerce
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with the necessary information:
   ```
   GOOGLE_API_KEY=your_google_api_key
   PROJECT_ID=your_gcp_project_id
   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGSMITH_PROJECT=ecommerce_support_bot
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Start the application with the command `streamlit run app.py`
2. The chatbot interface will appear in your web browser
3. Enter questions about products, shipping, returns, or common issues
4. The chatbot will return relevant information from the knowledge base

## Future Improvements

The project can be extended in the following ways:

- Adding new retrieval methods and comparing their performance
- Integration with actual product databases
- Adding multi-language support
- Implementing session tracking and user analytics

## License

[MIT License](LICENSE)
