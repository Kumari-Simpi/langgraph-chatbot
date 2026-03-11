"""chatbot.py — LangGraph agentic chatbot core logic
Tools: Arxiv, Wikipedia, Tavily
LLM: Groq (Qwen3-32b)
Memory: MemorySaver (per-session conversation history)
"""

import os  # noqa: F401
from dotenv import load_dotenv
load_dotenv()
#load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
from typing import Annotated

from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


def build_graph():
    """Build and return the compiled LangGraph with memory."""

    # ── System prompt ─────────────────────────────────────────────────────
    SYSTEM_PROMPT = """You are a helpful Research Assistant with access to three powerful tools:

1. **arxiv** – Search academic research papers on any scientific topic.
2. **wikipedia** – Look up factual information and background knowledge.
3. **tavily_search** – Search the web for recent news and current information.

When answering:
- Use the most appropriate tool(s) for the user's question.
- For research/academic topics → prefer arxiv.
- For factual background → prefer wikipedia.
- For recent news or current events → prefer tavily.
- Always be concise, accurate, and cite where information comes from.
- If you don't need a tool (e.g. casual greeting), respond directly.
"""

    # ── Tools ────────────────────────────────────────────────────────────
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper, description="Search arxiv for academic research papers")

    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    tavily_tool = TavilySearchResults(max_results=3)

    tools = [arxiv_tool, wiki_tool, tavily_tool]

    # ── LLM ──────────────────────────────────────────────────────────────
    llm = ChatGroq(model="qwen/qwen3-32b")
    llm_with_tools = llm.bind_tools(tools=tools)

    # ── State schema ─────────────────────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── Nodes ─────────────────────────────────────────────────────────────
    def tool_calling_llm(state: State):
        msgs = state["messages"]
        # Prepend system prompt on first turn only
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(msgs)
        return {"messages": [llm_with_tools.invoke(msgs)]}

    # ── Graph ─────────────────────────────────────────────────────────────
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")   # loop back → multi-step reasoning
    builder.add_edge("tool_calling_llm", END)

    # ── Memory (per-session conversation history) ─────────────────────────
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph


# Singleton — built once on import
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
