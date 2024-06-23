# Raza GPT

Raza GPT is a Streamlit-based web application that allows users to upload files (PDF, DOCX, CSV) and ask questions about the content using various AI models. 

## Features

- Upload multiple files (PDF, DOCX, CSV)
- Select from different AI models: OpenAI, Google Gemini, ChatGroq, Claude-2.1
- Process files, create text chunks, and build a vector store for efficient retrieval
- Ask questions about the uploaded files and receive detailed answers

## Requirements

- Python 3.7 or higher
- Streamlit
- Langchain
- PyMuPDF
- HuggingFace Transformers
- Qdrant Client
- Python-dotenv
- streamlit-chat

## Installation

1. **Clone the repository**

   ```sh
   git clone https://github.com/yourusername/raza-gpt.git
   cd raza-gpt
