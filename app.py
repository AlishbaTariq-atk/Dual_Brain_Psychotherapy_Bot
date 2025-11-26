# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from conversation import ChatbotConversation
from vector_store import load_vector_store
from llm_handler import initialize_llm

app = FastAPI(title="Dual-Brain Therapy Chatbot")

# Shared objects
llm_model = initialize_llm()
vector_store = load_vector_store()
chat_sessions = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message.strip()

    if not user_message:
        return {"response": "Please enter a message."}

    # Create a session for the user if it doesn't exist
    if user_id not in chat_sessions:
        chatbot = ChatbotConversation(
            llm_model=llm_model,
            vector_store=vector_store
        )
        chatbot.print_greeting()  # print in server logs
        chat_sessions[user_id] = chatbot
        greeting_text = chatbot.get_greeting_message() if hasattr(chatbot, "get_greeting_message") else None

        # Send greeting back to client
        if greeting_text:
            return {"response": greeting_text}

    chatbot = chat_sessions[user_id]

    if user_message.lower() == "bye":
        # This will save session + print farewell in logs
        chatbot.print_goodbye()
        del chat_sessions[user_id]
        return {"response": "Session ended. Goodbye!"}

    # Process user input
    response = chatbot.process_user_input(user_message)

    return {
        "response": response,
        "chat_history": chatbot.chat_history
    }
