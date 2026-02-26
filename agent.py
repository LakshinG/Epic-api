import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import tools

# Load environment variables
load_dotenv()

# Initialize the gemini model
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

llm = ChatOllama(model="llama3-groq-tool-use", temperature=0)

# Bind tools to the model
llm_with_tools = llm.bind_tools(tools)

def process_query(user_input: str, patient_id: str) -> Dict[str, Any]:
    """
    Processes a user query using the agent, executes tools if needed,
    and returns the final response along with data sources.

    Args:
        user_input (str): The doctor's natural language query.
        patient_id (str): The ID of the patient context.

    Returns:
        Dict: Contains 'response' (str) and 'data_sources' (List[Dict]).
    """

    # System prompt to set context
    # system_prompt gemini = f"""You are a helpful medical assistant.
    # You have access to tools to retrieve patient data for patient ID: {patient_id}.
    # ALWAYS use this patient_id when calling tools.
    # You can retrieve patient demographics, labs, and medications.

    # When answering, summarize the findings clearly.
    # If the user asks for data you just retrieved, present it nicely.
    # If you cannot find the answer in the tools, ask for clarification.
    # """

    system_prompt = """
    You are a secure Clinical Assistant. 
    You have access to real-time patient data via Epic FHIR tools.
    Your goal is to provide concise, medically accurate summaries for doctors.
    STRICT RULES:
    1. Only use information provided by the tools.
    2. If the tool returns a JSON bundle, analyze it fully for the requested data.
    3. If data is missing, state 'Information not available'—do not hallucinate.
    4. Keep patient privacy as a priority.
    """

    full_prompt = f"Patient ID: {patient_id}\nQuery: {user_input}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=full_prompt)
    ]

    data_sources = []

    print("\n--- DEBUG: Sending to LLM ---")
    print(f"Prompt: {full_prompt}")

    # First invocation
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Check for tool calls
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            # Find the matching tool implementation
            selected_tool = next((t for t in tools if t.name == tool_name), None)

            if selected_tool:
                # Execute the tool
                try:
                    # tool.invoke accepts a dict or args
                    tool_output = selected_tool.invoke(tool_args)
                except Exception as e:
                    tool_output = {
                        "resourceType": "OperationOutcome",
                        "issue": [{"severity": "error", "code": "exception", "diagnostics": str(e)}]
                    }

                # Append to data_sources (flatten list if it's a list)
                if isinstance(tool_output, list):
                    data_sources.extend(tool_output)
                else:
                    data_sources.append(tool_output)

                # Add ToolMessage to history
                # ToolMessage needs content as string
                messages.append(ToolMessage(
                    tool_call_id=tool_call_id,
                    content=str(tool_output),
                    name=tool_name
                ))
            else:
                # Handle hallucinated tool names
                error_msg = f"Error: Tool '{tool_name}' not found."
                messages.append(ToolMessage(
                    tool_call_id=tool_call_id,
                    content=error_msg,
                    name=tool_name
                ))

        # Second invocation to get final natural language response
        final_response_msg = llm_with_tools.invoke(messages)
        return {
            "response": final_response_msg.content,
            "data_sources": data_sources
        }
    else:
        # No tools called, just return the initial response
        return {
            "response": ai_msg.content,
            "data_sources": []
        }
