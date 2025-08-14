from typing import Dict, TypedDict, List, Optional, Union, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

doc_content = ''

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]
    
llm = ChatOpenAI(model="gpt-5-nano")
    
tools = []

@tool
def update(content: str) -> bool:
    """Tool to update the document with the provided content

    Args:
        content (str): content 

    Returns:
        bool: return True for success, false for fail
    """
    global doc_content
    try:
        doc_content = content
        return True
    except Exception as e:
        print(f"Failure to update content with error: {e}")
        return False
    
@tool
def save(file_name: str) -> bool:
    """ save the document to a file on the system
    
    Args:
        file_name (str): name of the file user wants. IMPORTANT: This MUST be a .txt file

    Returns:
        bool: return True on success, False on failure
    """
    global doc_content
    try:
        with open(file_name, "w") as file:
            file.write(doc_content)
        return True
    except Exception as e:
        print(f"Failure in saving content to file with error: {e}")
        return False        

tools = [update, save]

llm = llm.bind_tools(tools=tools)

def agent(state: AgentState) -> AgentState:
    """This agent will create a new .txt file 

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """
    
    system_prompt = SystemMessage(content=
    """
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}                           
    """
    )
    
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\nUSER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = llm.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if we shoul continue

    Args:
        state (AgentState): State

    Returns:
        str: Conditional path to take
    """
    
    messages = state['messages']
    
    if not messages:
        return "continue"
    
    # Look for the most recent tool use and determine if we should END
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower():
            return "end"
    
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\nTOOL RESULT: {message.content}")
            
graph = StateGraph(AgentState)

graph.add_node("agent", agent)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "agent")
graph.add_edge("agent", "tool_node")

graph.add_conditional_edges(
    "tool_node",
    should_continue,
    {
        "continue" : "agent",
        "end" : END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()