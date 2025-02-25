from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from trustcall import create_extractor
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()



# conversation
conversation = [HumanMessage(content="Hi, I am Aniket..."),
                AIMessage(content="Nice to meet you, Aniket."),
                HumanMessage(content="I really like biking around Pune."),]

# Schema
class UserProfile(BaseModel):
    """ User profile schema with typed fields """
    user_name: str = Field(description="The user's preferred name")
    interests: list[str] = Field(description="The list of user's interests")

# Initialize the model
model = ChatGroq(temperature=0, model="mixtral-8x7b-32768")

# Create the extractor
trustcall_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile"
)

# Instruction
system_msg = "Extract the user profile from the following conversation"

# Invoke the extractor
result = trustcall_extractor.invoke({"messages": [SystemMessage(content=system_msg)] + conversation})


print(result["responses"])