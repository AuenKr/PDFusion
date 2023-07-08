
# PDFusion: A smart PDF reader and query answering system
Fusion of PDFs and artificial intelligence to answer questions based on PDFs.

This is a project that aims to make PDFs more interactive and intelligent. It allows users to upload PDF files and ask questions based on the content of the PDFs. It uses the langchain framework to train language models on the PDF data and generate natural language answers.

## Features

- Train a language model on the PDF data using langchain
- Ask questions based on the PDF content and get natural language answers

## Requirements
-  OpenAI Key

## Installation

To install PDFusion, you need to have Python and pip installed. Then, run the following command:

```bash
pip install -r requirements.txt
```

This will install PDFusion and its dependencies, including PyPDF2 and langchain.

Create .env file and your key in following format
```bash
OPENAI_API_KEY = <OpenAi_key>
```

## Usage

To use PDFusion, you need to have a PDF file that you want to upload and query. Then, run the following command:

```bash
streamlit run app.py
```

This will launch a web interface where you can upload your PDF file and ask questions based on its content. You can also save and load the trained language models using the buttons on the web interface.
