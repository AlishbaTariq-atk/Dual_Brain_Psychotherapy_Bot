from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import config as config
import os 

def initialize_llm():
    """Initialize OpenAI Chat LLM."""
    print("\nInitializing OpenAI LLM...")
    try:
        model_openai = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=os.getenv("OPEN_AI_KEY") # <- THIS LINE is required!
        )
        print("OpenAI LLM initialized successfully.")
        return model_openai
    except Exception as e:
        print(f"Error initializing OpenAI LLM: {e}")
        print("Exiting chatbot. Please ensure your OpenAI API key is set correctly and the model is accessible.")
        return None

def convert_to_langchain_messages(chat_history_list):
    """Converts custom chat history format to Langchain's BaseMessage objects."""
    langchain_messages = []
    for entry in chat_history_list:
        role = entry["role"]
        content = entry["parts"][0]["text"]
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "model":
            langchain_messages.append(AIMessage(content=content))
        elif role.lower() == "system":
            langchain_messages.append(SystemMessage(content=content))
    return langchain_messages


def call_llm_api(chat_history_list, llm_model):
    """Calls the specified LLM (OpenAI) to get a response."""
    try:
        langchain_messages = convert_to_langchain_messages(chat_history_list)
        response = llm_model.invoke(langchain_messages)
        return response.content
    except Exception as e:
        print(f"Error communicating with OpenAI LLM: {e}")
        return "There was an error connecting to the AI. Please try again."
