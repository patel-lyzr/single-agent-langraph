"""
Single LangGraph agent using OpenAI.

A ReAct-style agent loop:
  user input → llm → (tool call? → run tool → llm) → response
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression. Example: '2 + 2 * 10'"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


@tool
def word_count(text: str) -> str:
    """Count the number of words in a text string."""
    return str(len(text.split()))


tools = [calculator, word_count]

# ---------------------------------------------------------------------------
# LLM + agent node
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful general-purpose assistant. "
    "Use tools when they are relevant to answer accurately."
)

llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"],
).bind_tools(tools)


def agent_node(state: State) -> State:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [llm.invoke(messages)]}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    sg = StateGraph(State)
    sg.add_node("agent", agent_node)
    sg.add_node("tools", ToolNode(tools))
    sg.set_entry_point("agent")
    sg.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    sg.add_edge("tools", "agent")
    return sg.compile()


graph = build_graph()

# ---------------------------------------------------------------------------
# BedrockAgentCore entrypoint
# ---------------------------------------------------------------------------

from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    prompt = payload.get("prompt", "")
    result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
    return {"result": result["messages"][-1].content}

if __name__ == "__main__":
    app.run()
