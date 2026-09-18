import os
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)


PDF_FILE = "Factory_Policy.pdf"

loader = PyPDFLoader(PDF_FILE)
documents = loader.load()

print("Total Pages:", len(documents))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Total Chunks:", len(chunks))


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings ready!")


vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS ready!")


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


class FactoryState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve_node(state):

    docs = retriever.invoke(state["question"])

    context = "\n\n".join(
        f"[Source Page: {doc.metadata.get('page', 0) + 1}]\n"
        f"{doc.page_content}"
        for doc in docs
    )

    return {
        "question": state["question"],
        "context": context,
        "answer": ""
    }


def llm_node(state):

    prompt = f"""
You are a Chowmein Factory AI Assistant.

Answer the user's question using ONLY the factory policy context.

Do not make up information.

If the answer is not available in the context, say:

"I could not find this information in the factory policy."

Factory Policy Context:
{state["context"]}

User Question:
{state["question"]}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "question": state["question"],
        "context": state["context"],
        "answer": response.text
    }


graph_builder = StateGraph(FactoryState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("llm", llm_node)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "llm")
graph_builder.add_edge("llm", END)

factory_graph = graph_builder.compile()


print("\nChowmein Factory AI Assistant is ready!")

while True:

    question = input(
        "\nAsk your question (type 'exit' to stop): "
    )

    if question.lower() == "exit":
        print("Chat ended.")
        break

    result = factory_graph.invoke({
        "question": question,
        "context": "",
        "answer": ""
    })

    print("\nAI Answer:")
    print(result["answer"])

    docs = retriever.invoke(question)

    source_pages = []

    for doc in docs:

        page = doc.metadata.get("page", 0) + 1

        if page not in source_pages:
            source_pages.append(page)

    print("\nSource Pages:")

    for page in source_pages:
        print("-", page)
