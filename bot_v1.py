"""
DBP Wellness Companion - Single File Implementation
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv
from docx2txt import process
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configuration
CONVERSATION_DIR = os.path.join(script_dir, "conversations")
SAFETY_FILTERS = {"clinical_terms": ["suicide", "self-harm", "abuse"]}
MAX_TURNS = 20
TEMPERATURE = 0.7

# Prompt templates
PROMPT_TEMPLATES = {
    "Introduction": """
    You are a compassionate and empathetic therapy assistant. Your role is to help clients explore their emotions in a safe and supportive environment.
    
    Start by introducing yourself and explaining how you can help. Ask an open-ended question to begin the conversation.
    """,
    "Ongoing": """
    Continue the conversation by responding to the user's message.
    
    Example questions:
    - What does this mean to you?
    - How does this relate to your current situation?
    """,
    "Visual Setup": """
    Guide the client through a visualization exercise. Ask them to imagine a safe space where they feel comfortable.
    
    Example questions:
    - What does your safe space look like?
    - What colors, sounds, or sensations do you notice?
    """,
    "Stressed Mind": """
    Help the client explore their stressed emotional part. Use empathetic language and validate their feelings.
    
    Example questions:
    - What is causing you stress right now?
    - How does this stress manifest in your body?
    """,
    "Mature Mind": """
    Guide the client to connect with their mature, compassionate self. Encourage them to offer understanding to their stressed part.
    
    Example questions:
    - What would your mature self say to your stressed part?
    - How can you show compassion to yourself in this situation?
    """,
    "Integration": """
    Help the client integrate insights from the session. Summarize key takeaways and suggest practical steps.
    
    Example questions:
    - What insights have you gained today?
    - What small step can you take this week to care for yourself?
    """
}

class DBPBot:
    """DBP Wellness Companion chatbot"""
    
    def __init__(self):
        self.conversations = self.load_conversations()
        self.vector_index = self.create_vector_index()
        self.llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.conversation_history = []
        self.has_introduced = False
        
    def load_conversations(self):
        """Load and parse conversation documents"""
        conversations = []
        for filename in os.listdir(CONVERSATION_DIR):
            if filename.endswith('.docx'):
                file_path = os.path.join(CONVERSATION_DIR, filename)
                text = process(file_path)
                
                # Parse conversation into turns
                turns = []
                current_speaker = None
                current_text = []
                
                for line in text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.endswith(':'):
                        # New speaker
                        if current_speaker is not None and current_text:
                            turns.append({
                                "speaker": current_speaker,
                                "text": "\n".join(current_text)
                            })
                            current_text = []
                        
                        current_speaker = line[:-1]  # Remove colon
                    else:
                        current_text.append(line)
                
                # Add last turn
                if current_speaker is not None and current_text:
                    turns.append({
                        "speaker": current_speaker,
                        "text": "\n".join(current_text)
                    })
                
                conversations.append({
                    "filename": filename,
                    "turns": turns
                })
        
        return conversations
        
    def create_vector_index(self):
        """Create vector index from conversation documents"""
        documents = []
        for conv in self.conversations:
            for turn in conv["turns"]:
                # Create Document objects with metadata
                documents.append(Document(
                    page_content=turn["text"],
                    metadata={"speaker": turn["speaker"]}
                ))
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        return FAISS.from_documents(documents, embeddings)
        
    def get_stage_prompt(self, stage: str) -> str:
        """Retrieve prompt template for the given stage"""
        return PROMPT_TEMPLATES[stage]
        
    def process_message(self, message: str) -> str:
        """Process user message and return bot response"""
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Determine which prompt template to use
        if not self.has_introduced:
            prompt_template = self.get_stage_prompt("Introduction")
            self.has_introduced = True
        else:
            prompt_template = self.get_stage_prompt("Ongoing")
        
        # Build context from conversation history
        context = ""
        if len(self.conversation_history) > 1:
            context = "Previous conversation:\n"
            for turn in self.conversation_history[-3:]:  # Last 3 turns for context
                role = "User" if turn["role"] == "user" else "Alex"
                context += f"{role}: {turn['content']}\n"
            context += "\n"
        
        # Format prompt with current context
        prompt = f"{prompt_template}\n\n{context}Current user message: {message}\n\nRespond as Alex:"
        
        # Generate response
        response = self.llm.generate_content(prompt)
        
        # Add bot response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response.text})
        
        return response.text

def main():
    """Main function to run the chatbot"""
    bot = DBPBot()
    print("DBP Wellness Companion: Hi, I'm here to help you explore your emotional mind.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Session ended. Remember: Listening to both parts of yourself is an act of strength.")
            break
            
        response = bot.process_message(user_input)
        print(f"\nBot: {response}")

if __name__ == "__main__":
    main()
