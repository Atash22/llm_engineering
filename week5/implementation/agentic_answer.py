from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from implementation.answer import vectorstore, llm, fetch_context, SYSTEM_PROMPT


class AgentState(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    iteration: int
    is_relevant: bool


def retrieve_node(state: AgentState) -> AgentState:
    docs = fetch_context(state["question"])
    return {"documents": docs}


def grade_documents_node(state: AgentState) -> AgentState:
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    prompt = f"""
    Question: {state['question']}
    Retrieved context: {context[:2000]}

    Does this context contain enough information to answer the question well?
    Reply with only one word: "yes" or "no".
    """
    result = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    return {
        "iteration": state.get("iteration", 0) + 1,
        "is_relevant": result.startswith("yes"),
    }


def rewrite_query_node(state: AgentState) -> AgentState:
    prompt = f"""
    The following question did not retrieve good context: {state['question']}
    Rewrite it as a clearer, more specific search query to find better information.
    Reply with only the rewritten query, nothing else.
    """
    new_question = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {"question": new_question}


def generate_node(state: AgentState) -> AgentState:
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [HumanMessage(content=system_prompt),
                HumanMessage(content=state["question"])]
    response = llm.invoke(messages)
    return {"answer": response.content}


def should_continue(state: AgentState) -> str:
    if state["is_relevant"] or state["iteration"] >= 3:
        return "generate"
    return "rewrite_query"


graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade_documents", grade_documents_node)
graph.add_node("rewrite_query", rewrite_query_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges("grade_documents", should_continue, {
    "generate": "generate",
    "rewrite_query": "rewrite_query",
})
graph.add_edge("rewrite_query", "retrieve")
graph.add_edge("generate", END)

agentic_graph = graph.compile()


def agentic_answer_question(question: str, history: list[dict] = []):
    result = agentic_graph.invoke({
        "question": question,
        "documents": [],
        "answer": "",
        "iteration": 0,
        "is_relevant": False,
    })
    return result["answer"], result["documents"]
