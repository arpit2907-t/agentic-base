import os
from LLM import get_llm
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_researcher_agent():
    os.environ["TAVILY_API_KEY"] = "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"
    llm = get_llm()
    search_tool = TavilySearchResults(max_results=2)
    tools = [search_tool]
    System_Prompt = """You are a Senior Research Agent. 
    Your Goal is to find high-quality, verifiable data. 

    You will receive a specific 'Mini-Topic' and the 'Full Outline' from the Planner Agent. 
    Your job is to gather research specifically for that Mini-Topic while ensuring it fits the context of the Full Outline.

    STRICT RULES:
    1) Always check at least two different sources for numerical data.
    2) If you find conflicting information, report both views.
    3) You must provide the URL of your sources.
    4) If a search returns no results, try rephrasing 3 times before giving up.
    5) If all attempts fail, return an empty string.
    6) You must follow the outline for the respective mini-topic.
    7) Ensure the data is unique to this topic to avoid overlap with other sections.

    TOOLS:
    You have access to the following tools: {tool_names}

    Current Task: {input}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", System_Prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]).partial(
        tool_names=", ".join([t.name for t in tools])
    )

    Research_agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=Research_agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3
    )


if __name__ == "__main__":
    agent_executor = get_researcher_agent()
    test_input = (
        "Mini-Topic: Charging Infrastructure in Goa. "
        "Outline: 1. Current Stations, 2. Government Subsidies, 3. Future Expansion."
    )
    
    try:
        response = agent_executor.invoke({"input": test_input})
        print("\n--- FINAL REPORT ---\n")
        print(response["output"])
    except Exception as e:
        print(f"An error occurred: {e}")