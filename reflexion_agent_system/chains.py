from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.messages import HumanMessage

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

parser = JsonOutputToolsParser(return_id=True)


load_dotenv()

llm = ChatGroq(temperature=0, model_name="qwen-2.5-32b")


# Actor Agent Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries seperately** for searching improvements. Do not include them inside the reflection.
"""),
    MessagesPlaceholder(variable_name="messages"),
    ("system","Answer the user's question above using the required format."),
    ]
).partial(
    time= lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

validator = PydanticToolsParser(tools=[AnswerQuestion])

# Revisor chain instruction prompt
revise_instruction = """Revise your previous answer using the new information.
    - You should use the previous critique to add inportant information to your answer.
        - You must include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
    """

# Revisor chain
revisor_chain = actor_prompt_template.partial(
    first_instruction= revise_instruction
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer") 

# response = first_responder_chain.invoke({"messages": [HumanMessage(content="Write me a blog post on how small businesses can leverage AI to grow")]})

# print(response)