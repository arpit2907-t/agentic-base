from LLM import get_llm
from langchain_classic.agents import AgentExecutor , create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
def get_presentor_agent():
    tools = []
    llm = get_llm()
    System_Prompt = """You are a Senior Presentor Agent. 
    Your Goal is to edit the final draft and create the final document which will be presented to the user. You will be provided with the final draft by the editor agent.
    You have to edit the final draft such that it is in an format to easily make slides and presentation from it, that is make the document into a presentation ready outline.

    There are some rules that you have to follow.
    STRICT RULES
    1) You must follow the outline as created by the planner agent.
    2) Make sure to keep the sources as written in the final draft.
    3) Make sure to not change the formal language.
    4) Make sure to present the data in a presentation ready outline.
    5) You must not make up any facts and only edit the final draft provided by the Editor Agent.

    USE THE FOLLOWING FORMAT:
    Thought : You should always think what to do 
    Action : Think about the final draft provided by the Editor agent and the outline provided by Planner Agent to create the presentation outline.
    Thought : I have created the final document in presentation outline ready style.
    You can repeat this thought action cycle as needed.
    Thought : I now have created the final document in presentation outline ready style document and no more further changes are required.
    Final Answer : Final Document Created.

    Current Task : {input}
    {agent_scratchpad}
    """

    prompt = ChatPromptTemplate.from_messages([
            ("system", System_Prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    Presentor_agent = create_tool_calling_agent(llm ,tools, prompt)
    return AgentExecutor(
        agent = Presentor_agent,
        tools=tools,
        verbose = False,
        handle_parsing_errors = True,
        max_iterations = 3
    )


