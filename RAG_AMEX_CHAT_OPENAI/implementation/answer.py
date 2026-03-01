from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_PATH = str(Path(__file__).parent.parent / "vector_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
llm = ChatOpenAI(model_name=MODEL, temperature=0)

SYSTEM_PROMPT = """You are a knowledgeable, friendly assistant representing American Express (Amex).
Use the context below to answer questions when relevant. If you don't know, say so.

Context:
{context}"""


def fetch_context(question: str) -> list[Document]:
    combined = question
    return retriever.invoke(combined)


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    prior_text = "\n".join(m["content"] for m in history if m["role"] == "user")
    query = (prior_text + "\n" + question).strip()

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content, docs