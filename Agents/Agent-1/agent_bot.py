from typing import Dict, TypedDict, List, Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import tiktoken

load_dotenv()

class AgentState(TypedDict):
    messages : List[HumanMessage]
    
tokeniser = tiktoken.encoding_for_model("gpt-5-nano")
    
llm = ChatOpenAI(model="gpt-5-nano")

def process(state: AgentState) -> AgentState:
    """Simple node that embeds an OpenAI API call

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """
    response = llm.invoke(state['messages'])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("What do you want to ask today?\n\nUser: ")
while user_input != "exit":
    input_tokens = tokeniser.encode(user_input)
    print(f"Token IDs: {input_tokens}")
    print(f"Token count: {len(input_tokens)}")
    
    print("Breakdown:")
    for i, token_id in enumerate(input_tokens):
        decoded = tokeniser.decode([token_id])
        print(f"  {i}: {token_id} → '{decoded}'")
        print(f"Input token cost: {input_tokens}")
    agent.invoke({
        "messages" : [HumanMessage(content=user_input)]
    })
    user_input = input("Any other questions?\n\nUser: ")