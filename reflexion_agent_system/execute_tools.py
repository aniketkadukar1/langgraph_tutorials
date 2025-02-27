from typing import List
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage
from chains import parser
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_community.tools import TavilySearchResults
from collections import defaultdict
import json

tavily_tool = TavilySearchResults(max_results=5)
tool_executor = ToolExecutor([tavily_tool])


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    last_ai_message = state[-1]
    parsed_tool_calls = parser.invoke(last_ai_message)

    ids = []
    tool_invocations = []

    for parsed_call in parsed_tool_calls:
        for query in parsed_call['args']['search_queries']:
            tool_invocations.append(

                ToolInvocation(
                    tool="tavily_search_result_json",
                    tool_input=query
                )
            )
            ids.append(parsed_call["id"])
    
    outputs = tool_executor.batch(tool_invocations)

    # Map each output to its corresponding ID and tool input
    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocation.tool_input] = output

    # Convert the mapped outputs to ToolMessage objects
    tool_messages = []
    for id_, mapped_output in outputs_map.items():
        tool_messages.append(
            ToolMessage(content=json.dumps(mapped_output),
            tool_call_id =id_)
        )

    return tool_messages
