# main.py
import os
# from chatbot_porject import prompts
import config as config
from loader import load_all_knowledge_bases
from vector_store import build_vector_store, load_vector_store, save_vector_store
from llm_handler import initialize_llm
from conversation import ChatbotConversation
from dotenv import load_dotenv
from huggingface_hub import login
from prompts import get_greeting_message

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_HUB_TOKEN")
login(token=os.environ["HF_TOKEN"])

# Initialize shared objects
llm_model = None
vector_store = None
chatbot = None

def setup_knowledge_base():
    """Setup or load the knowledge base."""
    print("\nSetting up knowledge base...")

    vector_store = load_vector_store()
    if vector_store:
        print("Knowledge base loaded from existing vector store.")
        return vector_store

    print("No existing vector store found. Building from text files...")
    knowledge_files = [
        config.TEXT_FILE_FOR_KNOWLEDGE,
        config.BOOK_FILE_FOR_KNOWLEDGE,
        config.BOOK2_FILE_FOR_KNOWLEDGE,
        config.BOOK3_FILE_FOR_KNOWLEDGE
    ]

    missing_files = [f for f in knowledge_files if not os.path.exists(f)]
    if missing_files:
        print("\n❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease add these files to the data/ directory and run again.")
        return None

    combined_knowledge_content = load_all_knowledge_bases(knowledge_files)
    vector_store = build_vector_store(combined_knowledge_content)

    if vector_store:
        save_vector_store(vector_store)
        print("Knowledge base built and saved.")
    else:
        print("Warning: Vector store could not be built.")

    return vector_store

def initialize_chatbot():
    """Initialize LLM and vector store once for reuse."""
    global llm_model, vector_store, chatbot

    if not llm_model:
        llm_model = initialize_llm()
    if not vector_store:
        vector_store = setup_knowledge_base()
    if llm_model and vector_store:
        print(get_greeting_message())
        chatbot = ChatbotConversation(llm_model, vector_store)

def start_chatbot(user_message: str) -> str:
    """Process a single user message and return bot response."""
    global chatbot

    if chatbot is None:
        initialize_chatbot()

    if chatbot:
        return chatbot.process_user_input(user_message)  # ✅ FIXED METHOD NAME
    else:
        return "⚠️ Sorry, I couldn't initialize the chatbot."

if __name__ == "__main__":
    print("Running in CLI mode. Type 'bye' to exit.")
    initialize_chatbot()
    if not chatbot:
        print("Chatbot failed to start.")
    else:
        while True:
            msg = input("You: ")
            if msg.lower() == 'bye':
                chatbot.print_goodbye()  # Optional: gracefully end
                print("Bot: Take care! 😊")
                break
            print("Bot:", start_chatbot(msg))
