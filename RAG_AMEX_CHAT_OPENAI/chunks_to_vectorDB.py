import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)
BASE_DIR = Path(__file__).parent.parent
print(BASE_DIR)
VECTOR_DB = str(BASE_DIR / "vector_db")
KNOWLEDGE_BASE = str(BASE_DIR / "amex_knowledge_base")
print(KNOWLEDGE_BASE)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def load_documents():
    docs = []
    for folder in glob.glob(str(Path(KNOWLEDGE_BASE) / "*")):
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for doc in loader.load():
            doc.metadata["doc_type"] = os.path.basename(folder)
            docs.append(doc)
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def build_vector_store(chunks):
    if os.path.exists(VECTOR_DB):
        Chroma(persist_directory=VECTOR_DB, embedding_function=embeddings).delete_collection()
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=VECTOR_DB)


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
    print(f"Done — {len(docs)} documents, {len(chunks)} chunks ingested.")