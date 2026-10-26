from LLM import get_llm
from langchain_classic.agents import AgentExecutor , create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
def get_writer_agent():
    tools = []
    llm = get_llm()
    System_Prompt = """You are a Senior Writer Agent. 
    Your Goal is to write about mini-topics that you will be receving from the planner agent as per the outline also provided by it. 
    The data you will be using will be provided to you by the Researcher Agent.

    There are some rules that you have to follow.
    STRICT RULES
    1) You must follow the outline as created by the planner agent.
    2) Make sure to explain each mini-topic clearly.
    3) Make sure to also incorporate the sources of data you are using. The necessary sources will be provided to you by the researcher agent.
    4) If there exists conflicting data in the same mini topic, you must write about all the sides.
    5) You must write distinct things for all the mini-topics provided to you.
    6) You must not make up any facts and only write using the data provided by the Research Agent.

    USE THE FOLLOWING FORMAT:
    Thought : You should always think what to do 
    Action : Think about the mini-topic , the outline provided by the Planner Agent and use the data provided by the Research Agent to write.
    Thought : I have written enough as per the outline provided to me.
    You can repeat this thought action cycle as needed.
    Thought : I now have completed the draft for all the mini-topics.
    Final Answer : Draft Created.

    Current Task : {input}
    {agent_scratchpad}
    """

    prompt = ChatPromptTemplate.from_messages([
            ("system", System_Prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    Writer_agent = create_tool_calling_agent(llm ,tools, prompt)
    return AgentExecutor(
        agent = Writer_agent,
        tools=tools,
        verbose = False,
        handle_parsing_errors = True,
        max_iterations = 3
    )


