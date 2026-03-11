"""
app.py — Streamlit frontend for the LangGraph Research Chatbot
"""
import os                          # ← add this
from dotenv import load_dotenv     # ← add this
load_dotenv()

import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from chatbot import get_graph


# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Chatbot",
    page_icon="🔬",
    layout="centered",
)

st.title("🔬 Research Assistant Chatbot")
st.caption(
    "Powered by **LangGraph** · **Groq (Qwen3-32b)** · Tools: Arxiv · Wikipedia · Tavily"
)

# ── Session state ─────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    # List of dicts: {"role": "user"/"assistant"/"tool", "content": "...", "tool_name": "..."}
    st.session_state.chat_history = []

graph = get_graph()


# ── Helper: extract tool calls used ──────────────────────────────────────
def get_tool_names_used(messages):
    """Return list of tool names called in this response."""
    tool_names = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                tool_names.append(tc["name"])
    return tool_names


# ── Helper: get final AI text response ───────────────────────────────────
def get_final_ai_text(messages):
    """Return the last AIMessage that has text content (not just tool calls)."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return "*(No response)*"


# ── Display existing chat history ─────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🔬"):
            if msg.get("tools_used"):
                tools_str = " · ".join(f"`{t}`" for t in msg["tools_used"])
                st.caption(f"🛠 Tools used: {tools_str}")
            st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything — research papers, news, facts...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Run the graph
    with st.chat_message("assistant", avatar="🔬"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.session_id}}
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

        all_messages = result["messages"]
        tools_used = get_tool_names_used(all_messages)
        final_text = get_final_ai_text(all_messages)

        if tools_used:
            tools_str = " · ".join(f"`{t}`" for t in tools_used)
            st.caption(f"🛠 Tools used: {tools_str}")

        st.markdown(final_text)

    # Save to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": final_text,
        "tools_used": tools_used,
    })


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot is a **LangGraph agentic system** that intelligently routes 
    your questions to the right tool:

    | Tool | Best For |
    |------|----------|
    | 🧑‍🔬 **Arxiv** | Research papers |
    | 📖 **Wikipedia** | Facts & background |
    | 🌐 **Tavily** | News & current events |

    The bot maintains **conversation memory** across the session.
    """)

    st.divider()
    st.markdown("**Try asking:**")
    st.code("What is the paper 1706.03762?", language=None)
    st.code("Recent AI news?", language=None)
    st.code("What is quantum entanglement?", language=None)
    st.code("Latest research on LLMs?", language=None)

    st.divider()
    if st.button("🗑 Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
