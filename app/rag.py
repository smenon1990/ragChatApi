import os
from pypdf import PdfReader
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embeddings (THIS replaces embedding_model)
embeddings = OpenAIEmbeddings()

DATA_DIR = "/data"
os.makedirs(DATA_DIR, exist_ok=True)


def ingest_pdf(file_path: str):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(f"{DATA_DIR}/faiss")


def query_rag(question: str) -> str:
    vectorstore = FAISS.load_local(
        f"{DATA_DIR}/faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(question, k=4)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content
