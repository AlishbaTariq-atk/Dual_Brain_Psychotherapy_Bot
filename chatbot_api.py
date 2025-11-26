"""
DBP Chatbot REST API
Simple API wrapper for the DBP chatbot to enable web-based integration
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bot_v1 import DBPBot

# Initialize FastAPI app
app = FastAPI(
    title="DBP Chatbot API",
    description="REST API for DBP Wellness Companion chatbot",
    version="1.0.0"
)

# Add CORS middleware to allow web requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active chat sessions
chat_sessions: Dict[str, DBPBot] = {}

# Request/Response models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    conversation_length: int

class SessionInfo(BaseModel):
    session_id: str
    conversation_length: int
    has_introduced: bool

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "DBP Chatbot API is running", "status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    """
    Send a message to the chatbot and get a response
    
    - **message**: The user's message to send to the chatbot
    - **session_id**: Optional session identifier (defaults to "default")
    """
    try:
        session_id = chat_message.session_id
        
        # Create new session if it doesn't exist
        if session_id not in chat_sessions:
            chat_sessions[session_id] = DBPBot()
        
        # Get bot instance for this session
        bot = chat_sessions[session_id]
        
        # Process the message
        response = bot.process_message(chat_message.message)
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            conversation_length=len(bot.conversation_history)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/sessions", response_model=List[SessionInfo])
async def get_sessions():
    """Get information about all active chat sessions"""
    sessions = []
    for session_id, bot in chat_sessions.items():
        sessions.append(SessionInfo(
            session_id=session_id,
            conversation_length=len(bot.conversation_history),
            has_introduced=bot.has_introduced
        ))
    return sessions

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific chat session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/sessions")
async def delete_all_sessions():
    """Delete all chat sessions"""
    chat_sessions.clear()
    return {"message": "All sessions deleted successfully"}

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a specific session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    bot = chat_sessions[session_id]
    return {
        "session_id": session_id,
        "conversation_history": bot.conversation_history,
        "conversation_length": len(bot.conversation_history)
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
