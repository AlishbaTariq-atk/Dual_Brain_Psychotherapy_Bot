import os
import json
import random
import re 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

# Load environment variables (for API key, if not using Canvas's auto-injection)
load_dotenv()

# Get absolute path to text.txt in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
TEXT_FILE_FOR_KNOWLEDGE = os.path.abspath(os.path.join(script_dir, "text.txt"))

# Wellness-safe phrasing guidelines (retained from previous iteration)
WELLNESS_APPROVED_PHRASES = [
    'Emotional overwhelm', 'Stressed Mind', 'Mature Mind', 'Discomfort',
    'Inner tension', 'Protective part of the self', 'Uncomfortable thoughts',
    'A part of you feels…', 'Let’s listen to that side for a moment…',
    'Emotional pressure', 'Let’s listen to that voice…'
]

WELLNESS_FORBIDDEN_TERMS = [
    'Anxiety symptoms', 'Fear response', 'Panic attack', 'You are traumatized',
    'You need therapy', 'We’ll fix this', 'Treatment', 'Diagnosis',
    'Mental illness', 'You are dissociating', 'Anxiety disorder',
    'Trauma response', 'Dissociation', 'You need help'
]

# --- Brain Dominance Assessment Questions ---
BRAIN_DOMINANCE_QUESTIONS = [
    {"id": 0, "question": "When solving a complex problem, do you prefer to break it down into smaller, logical steps, or approach it holistically?"},
    {"id": 1, "question": "How do you typically organize your tasks or thoughts? (e.g., using lists and detailed plans, or more flexible, mental maps)"},
    {"id": 2, "question": "When learning something new, do you prefer detailed instructions and facts, or a more conceptual overview and big picture?"},
    {"id": 3, "question": "How do you make important decisions? (e.g., based on data, analysis, and pros/cons, or intuition and gut feeling)"},
    #{"id": 4, "question": "What kind of books, articles, or media do you enjoy most? (e.g., non-fiction, technical manuals, news vs. fiction, poetry, visual arts)"},
    #{"id": 5, "question": "When faced with a new situation, do you tend to rely on your intuition first, or on logical analysis and past experiences?"},
    #{"id": 6, "question": "How do you express your creativity? (e.g., through structured problem-solving, writing, or through art, music, storytelling, abstract ideas)"},
    #{"id": 7, "question": "When remembering events, do you recall details sequentially and chronologically, or do you have a more vivid, sensory, and emotional memory?"},
    #{"id": 8, "question": "How do you prefer to communicate your ideas? (e.g., precise words, clear arguments, step-by-step explanations vs. metaphors, storytelling, non-verbal cues)"},
    #{"id": 9, "question": "Do you find yourself drawn more to patterns, connections, and overall themes, or to individual facts, details, and categories?"}
]


# --- RAG System Setup ---
# The parse_vtt_transcript function is no longer needed as we are reading a plain text file.
# def parse_vtt_transcript(vtt_content):
#     """Parses VTT content to extract only the text."""
#     # ... (function body removed) 



def get_all_local_transcripts_from_file(filename):
    """Reads and returns text from a single local text file."""
    print(f"\n--- Reading knowledge base from '{filename}' ---")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Successfully read '{filename}'.")
            return content
    except FileNotFoundError:
        print(f"Error: Knowledge base file '{filename}' not found. Please ensure it's in the same directory.")
        return "" # Return empty string if file not found
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return "" # Return empty string on other errors


def build_vector_store(text_content):
    """Builds a FAISS vector store from text content."""
    if not text_content.strip():
        print("No text content to build vector store. RAG will not be effective.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text_content])

    # Use Hugging Face embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store built successfully.")
    return vector_store


def retrieve_context(query, vector_store, k=3):
    """Retrieves relevant context from the vector store."""
    if vector_store is None:
        return "No knowledge base available."
    docs = vector_store.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    return context


def convert_to_langchain_messages(chat_history_list):
    """Converts custom chat history format to Langchain's BaseMessage objects."""
    langchain_messages = []
    for entry in chat_history_list:
        role = entry["role"]
        content = entry["parts"][0]["text"] # Assuming single part for simplicity
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "model":
            langchain_messages.append(AIMessage(content=content))
        elif role == "system":
            langchain_messages.append(SystemMessage(content=content))
    return langchain_messages

def call_llm_api(chat_history_list, model):
    """Calls the Gemini model to get a response."""
    try:
        # Convert custom chat history format to Gemini format
        gemini_messages = []
        for entry in chat_history_list:
            role = entry["role"]
            content = entry["parts"][0]["text"]
            if role == "user":
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
            elif role == "model":
                gemini_messages.append({"role": "model", "parts": [{"text": content}]})
            elif role == "system":
                # For Gemini, we can include system messages as user messages with note
                gemini_messages.append({"role": "user", "parts": [{"text": f"[System]: {content}"}]})
        
        # Generate response - pass the list directly, not wrapped in a dictionary
        response = model.generate_content(gemini_messages)
        return response.text
    except Exception as e:
        print(f"Error communicating with Gemini API: {e}")
        return "There was an error connecting to the AI. Please try again."

def run_chatbot():
    print("Hello! I'm your Dual-Brain Psychotherapy Chatbot. I'm here to help you explore your mind state and offer supportive insights.")
    print("Type 'bye' to exit at any time.")

    # Initialize Gemini model
    print("\nInitializing Gemini model...")
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Gemini model initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        print("Exiting chatbot. Please ensure GEMINI_API_KEY is set correctly.")
        return # Exit if model cannot be initialized

    # Initialize RAG system
    print("\nSetting up knowledge base from text file...")
    # Changed to load from a single text file
    combined_youtube_content = get_all_local_transcripts_from_file(TEXT_FILE_FOR_KNOWLEDGE)
    global vector_store # Make it accessible globally for the chatbot
    vector_store = build_vector_store(combined_youtube_content)
    if vector_store:
        print("Knowledge base ready. I can answer questions related to the provided text content.")
    else:
        print("No text knowledge base loaded. I will respond based on general knowledge.")

    chat_history = [] # Stores conversation for main chat
    right_brain_score = 0
    left_brain_score = 0
    brain_question_index = 0 # Start brain dominance assessment

    print("\nLet's begin by exploring your cognitive style. I'll ask you a few questions to understand if you lean more towards 'right-brained' or 'left-brained' thinking.")

    while True:
        if brain_question_index < len(BRAIN_DOMINANCE_QUESTIONS):
            # Brain dominance assessment phase
            current_question_data = BRAIN_DOMINANCE_QUESTIONS[brain_question_index]
            print(f"\nBot: {current_question_data['question']}")
            user_input = input("You: ").strip()

            if user_input.lower() == 'bye':
                print("Bot: Goodbye! Take care.")
                break

            # Evaluate user response for brain dominance
            # Prompt for brain dominance classification
            brain_dominance_prompt = f"""You are an AI assistant specialized in analyzing cognitive styles.
            Your task is to evaluate a user's answer to a question and classify its overall sentiment as indicating a "left_brain" or "right_brain" cognitive style.
            Left-brained characteristics include: logical, analytical, sequential, factual, detail-oriented, verbal, structured.
            Right-brained characteristics include: intuitive, holistic, creative, imaginative, non-verbal, emotional, pattern-oriented.
            If the answer is neutral or ambiguous, infer the style based on the general context of the question and the user's phrasing, leaning towards the most likely cognitive style it implies.
            Your response MUST be a single word: either "left_brain" or "right_brain". Do not include any other text, explanations, or punctuation.

            ---
            Question: "{current_question_data['question']}"
            User Answer: "{user_input}"
            ---
            Cognitive Style:"""

            # Use a fresh chat history for this classification to avoid context bleed
            # The prompt itself is the user's message for this classification task
            classification_chat_history = [{"role": "user", "parts": [{"text": brain_dominance_prompt}]}]
            # Removed 'await' keyword
            classification_result = call_llm_api(classification_chat_history, model)
            classification = classification_result.lower().strip()

            if "left_brain" in classification:
                left_brain_score += 1
                print("Bot: Understood. That points towards a more structured approach.")
            elif "right_brain" in classification:
                right_brain_score += 1
                print("Bot: Got it. That suggests a more intuitive perspective.")
            else:
                print("Bot: Thank you for sharing. I'll consider that for your cognitive style assessment.")

            brain_question_index += 1

            if brain_question_index == len(BRAIN_DOMINANCE_QUESTIONS):
                # End of brain dominance assessment
                dominance_message = ""
                if left_brain_score > right_brain_score:
                    dominance_message = f"Based on your responses (Left-brained score: {left_brain_score}, Right-brained score: {right_brain_score}), it seems you lean more towards a **left-brained** cognitive style, emphasizing logic and analysis."
                elif right_brain_score > left_brain_score:
                    dominance_message = f"Based on your responses (Left-brained score: {left_brain_score}, Right-brained score: {right_brain_score}), it appears you lean more towards a **right-brained** cognitive style, favoring intuition and creativity."
                else:
                    dominance_message = f"Your responses indicate a balanced cognitive style (Left-brained score: {left_brain_score}, Right-brained score: {right_brain_score}), suggesting you might use both logical and intuitive approaches equally."
                
                print(f"\nBot: {dominance_message}\nNow, how can I help you further? Feel free to ask me anything or share more about what's on your mind. I can also answer questions about the YouTube content I've processed.")
                # Add the assessment conclusion to the main chat history
                chat_history.append({"role": "model", "parts": [{"text": dominance_message}]})
        else:
            # Regular chat phase
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'bye':
                print("Bot: Goodbye! Take care.")
                break

            # Retrieve context from RAG system
            context_from_youtube = retrieve_context(user_input, vector_store)

            # Construct the main chatbot prompt with wellness safety guidelines and RAG context
            main_chat_system_prompt = f"""You are a warm, deeply empathetic, and non-judgmental AI companion. Your primary purpose is to offer understanding, encouragement, and to help the user explore their feelings and experiences. You are here to listen and provide insights from a general wellness and psychoeducational perspective.

            You are **NOT** a therapist, doctor, or medical professional. You **cannot** diagnose, treat, or offer any form of medical or psychological advice. Your responses must always maintain a supportive, non-clinical, and non-prescriptive tone.

            ---
            **Wellness-Safe Language Guidelines:**

            **Always use terms from the approved list for emotional states and internal experiences:**
            - {', '.join(WELLNESS_APPROVED_PHRASES)}

            **NEVER use the following forbidden terms (or their direct synonyms/variations):**
            - {', '.join(WELLNESS_FORBIDDEN_TERMS)}

            ---
            **Behavioral Instructions:**
            1.  **Acknowledge and Validate:** Always acknowledge the user's emotions and experiences first. Use phrases like "It sounds like...", "I hear that...", "That makes sense...".
            2.  **Rephrase Problematic Language:** If the user or the context might imply a forbidden term (e.g., intense fear, a feeling of 'losing control', or a clinical label), always rephrase your response using the **approved, wellness-safe language**. For instance, instead of "panic attack," use "emotional overwhelm" or "intense discomfort."
            3.  **Promote Self-Exploration:** Encourage the user to elaborate on their feelings or experiences by asking gentle, open-ended questions. Frame observations about the user's state as "a part of you feels..." or "it sounds like you're experiencing..." to maintain a non-prescriptive stance.
            4.  **Utilize Provided Knowledge (RAG):** When the user asks a question, check the `YouTube Transcript Context` provided below. If relevant information is present, synthesize it into your response, always maintaining your supportive, non-clinical persona and wellness-safe language.
            5.  **Handle Insufficient Context:** If the `YouTube Transcript Context` does not contain enough information to directly answer the user's question, state that you don't have specific information on that, but then pivot to offering general supportive conversation or encouraging self-reflection related to their query. Do not invent facts.
            6.  **Maintain Flow:** Ensure your responses are conversational and contribute to a continuous, supportive dialogue.

            ---
            **YouTube Transcript Context:**
            {context_from_youtube}

            ---
            """

            # Add current user input to chat history
            chat_history.append({"role": "user", "parts": [{"text": user_input}]})

            # Prepend system prompt to the current turn's chat history for consistent instruction
            current_turn_chat_history = [{"role": "system", "parts": [{"text": main_chat_system_prompt}]}] + chat_history

            # Removed 'await' keyword
            bot_response = call_llm_api(current_turn_chat_history, model)
            print(f"Bot: {bot_response}")

            # Add bot's response to chat history for future turns
            chat_history.append({"role": "model", "parts": [{"text": bot_response}]})

# To run the chatbot
if __name__ == "__main__":
    # Removed 'asyncio.run' and direct call 'run_chatbot()'
    run_chatbot()
