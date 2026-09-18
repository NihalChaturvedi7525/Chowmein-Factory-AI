# Chowmein Factory AI Assistant

Generative AI factory assistant using RAG, LangChain, LangGraph, FAISS and Gemini.

## Features

- PDF document loading
- Text chunking
- Hugging Face embeddings
- FAISS vector search
- RAG-based question answering
- LangGraph workflow
- Gemini LLM
- Source page display

## Architecture

User Question
-> LangGraph
-> Retriever
-> FAISS
-> Factory Policy Context
-> Gemini
-> Final Answer
-> Source Pages

## Technologies

Python
LangChain
LangGraph
RAG
FAISS
Hugging Face Embeddings
Gemini
PyPDFLoader

## How to Run

Install dependencies:

pip install -r requirements.txt

Keep Factory_Policy.pdf in the same folder as app.py.

Run:

python app.py

## Security

Never upload your Gemini API key to GitHub.

## Author

Nihal Chaturvedi
