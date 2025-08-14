from typing import Dict, TypedDict, List, Optional, Union, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]
    
llm = ChatOpenAI(model="gpt-5-nano")
    
tools = []
        
@tool
def add(a: int, b: int) -> int:
    """This is an addition function that adds two numbers togethe

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        int: Sum of a and b
    """
    
    return a + b
    
tools.append(add)

llm = llm.bind_tools(tools=tools)

def llm_call(state: AgentState) -> AgentState:
    """ Node for LLM call

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """
    
    system_prompt = SystemMessage(content=
    """
    You are a helpful AI assistant. 

    CRITICAL TOOL USAGE RULES:
    1. ONLY call tools when the user's request DIRECTLY matches the tool's purpose
    2. Never use tools creatively or for unintended purposes
    3. If you don't have the right tool, say so instead of misusing available tools
    4. Think carefully before each tool call - does this EXACTLY match what the user asked?

    Available tools:
    - add: ONLY for addition operations (a + b)

    Examples of CORRECT usage:
    - User: "What's 5 + 3?" → Use add(5, 3)

    Examples of INCORRECT usage:
    - User: "What's 5 * 3?" → DO NOT use add(5, 3) repeatedly
    - User: "I like math" → DO NOT call any math tools
    """
    )
    
    response = llm.invoke([system_prompt] + list(state["messages"]))
    
    return {"messages" : [response]}

def should_continue(state: AgentState) -> str:
    messages = state['messages']
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)

graph.add_node("llm_call", llm_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "llm_call")

graph.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "continue" : "tool_node",
        "end" : END
    }
)

graph.add_edge("tool_node", "llm_call")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            
user_input = input("Enter: ")
inputs = {"messages": [("user", user_input)]}
print_stream(app.stream(inputs, stream_mode="values"))