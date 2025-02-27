from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from schema import AnswerQuestion
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

load_dotenv()

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")


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

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") | pydantic_parser

response = first_responder_chain.invoke({"messages": [HumanMessage(content="Write me a blog post on how small businesses can leverage AI to grow")]})

print(response)