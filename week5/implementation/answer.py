from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gpt-4.1"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 12

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.

Rules for your answer:
1. Use ONLY the given context to answer. If the context does not contain the answer, say so directly.
2. Include every fact from the context that is relevant to the question — full names, exact titles,
   dates, and numbers must be copied exactly as written, never paraphrased, abbreviated, or shortened.
3. Do not omit any detail present in the context that helps answer the question completely.
4. Do not add information not found in the context, and do not add unrelated commentary or extra facts
   beyond what was asked.
5. Be direct and concise while still being complete.

Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
llm = ChatOpenAI(temperature=0, model_name=MODEL)

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def fetch_context(question: str, k: int = RETRIEVAL_K, fetch_k: int = 30) -> list[Document]:
    """
    Retrieve relevant context documents for a question, re-ranked by a cross-encoder.
    """
    initial_docs = vectorstore.similarity_search(question, k=fetch_k)

    if not initial_docs:
        return []

    reranker = get_reranker()
    pairs = [[question, doc.page_content] for doc in initial_docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs