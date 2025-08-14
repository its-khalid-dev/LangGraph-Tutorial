from typing import Dict, TypedDict, List, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]
    
llm = ChatOpenAI(model="gpt-5-nano")

def process(state: AgentState) -> AgentState:
    """Simple node to API call GPT model

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """
    
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    
    print(f"STATE: {state}")
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history: List[Union[HumanMessage, AIMessage]] = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({
        "messages" : conversation_history
    })
    conversation_history = result["messages"]
    user_input = input("\nEnter: ")
    
with open("log.txt", "w") as file:
    file.write("Conversation beginning")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"Human: {message}")
        else:
            file.write(f"AI: {message}")
    
    file.write("End of conversation")
            