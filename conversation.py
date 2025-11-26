import os
import re
import json
from datetime import datetime
from pathlib import Path

from vector_store import retrieve_context
from llm_handler import call_llm_api
from prompts import get_system_prompt, get_greeting_message, get_goodbye_message
from entity_extractor import extract_entities_from_chat_history
from brain_assessor import analyze_brain_profile_from_chat


class ChatbotConversation:
    """Main chatbot conversation handler."""

    def __init__(self, llm_model, vector_store):
        self.llm_model = llm_model
        self.vector_store = vector_store
        self.chat_history = []
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        # Optional identifier (e.g., user name) – can be injected later (e.g., from UI)
        self.user_id: str | None = None

    def get_user_input(self):
        """Get user input from the console."""
        return input("\nYou: ").strip()

    def process_user_input(self, user_input):
        """Process user input and generate bot response."""
        # Retrieve relevant context from vector store
        context_from_knowledge_base = retrieve_context(user_input, self.vector_store)

        # Get system prompt with context
        system_prompt = get_system_prompt(context_from_knowledge_base)

        # Add user input to chat history
        self.chat_history.append({
            "role": "user",
            "parts": [{"text": user_input}]
        })

        # Prepare messages for LLM (system prompt + chat history)
        current_turn_chat_history = [
            {"role": "system", "parts": [{"text": system_prompt}]}
        ] + self.chat_history

        # Get response from LLM
        bot_response = call_llm_api(current_turn_chat_history, self.llm_model)

        # Add bot response to chat history
        self.chat_history.append({
            "role": "model",
            "parts": [{"text": bot_response}]
        })

        return bot_response

    def extract_number_from_text(self, text):
        """Extract numeric rating from text like '8 out of 10' or 'about seven'."""
        # Check for digits first
        match = re.search(r"\b\d+\b", text)
        if match:
            return int(match.group())

        # Map number words to digits
        words_to_numbers = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        for word in text.lower().split():
            if word in words_to_numbers:
                return words_to_numbers[word]

        return None

    def extract_envelope_values(self):
        """
        Go through chat history and pull out any Envelope Technique answers,
        storing both raw text and parsed numeric values.
        """
        envelope_data = {
            "RIGHT_ANXIETY": None,
            "RIGHT_DEPRESSION": None,
            "RIGHT_SUPPORTIVENESS": None,
            "RIGHT_CRAVINGS": None,
            "RIGHT_REMINDER": None,
            "LEFT_ANXIETY": None,
            "LEFT_DEPRESSION": None,
            "LEFT_SUPPORTIVENESS": None,
            "LEFT_CRAVINGS": None,
            "LEFT_REMINDER": None
        }

        # Detect lines with tags OR matching question patterns
        tag_pattern = re.compile(
            r"\[(RIGHT|LEFT)_(ANXIETY|DEPRESSION|SUPPORTIVENESS|CRAVINGS|REMINDER):\s*(.*?)\]",
            re.IGNORECASE
        )

        # We'll scan *all* messages
        for msg in self.chat_history:
            text = msg["parts"][0]["text"]

            # Match explicit tags like [RIGHT_ANXIETY: 8]
            matches = tag_pattern.findall(text)
            if matches:
                for side, metric, value in matches:
                    key = f"{side.upper()}_{metric.upper()}"
                    envelope_data[key] = {
                        "text": value.strip(),
                        "value": self.extract_number_from_text(value)
                    }
                continue

            # Match natural Q&A style by looking for question keywords
            for key in envelope_data.keys():
                metric = key.split("_")[1].lower()
                side = key.split("_")[0].lower()

                if metric in text.lower() and msg["role"] == "user":
                    envelope_data[key] = {
                        "text": text.strip(),
                        "value": self.extract_number_from_text(text)
                    }

        return envelope_data

    def print_greeting(self):
        """Print initial greeting message."""
        greeting = get_greeting_message()
        print(f"Bot: {greeting}")

    def print_goodbye(self):
        """Finalize session, save per-day file, and update all_sessions.json."""
        # Farewell message
        farewell = get_goodbye_message()
        print(f"Bot: {farewell}")

        # Extract session data
        entities = extract_entities_from_chat_history(self.chat_history)
        brain_profile = analyze_brain_profile_from_chat(self.chat_history)
        envelope_values = self.extract_envelope_values()

        session_summary = {
            "timestamp": datetime.now().isoformat(),
            "chat_history": self.chat_history,
            "entities": entities,
            "brain_profile": brain_profile,
            "envelope_assessment": envelope_values
        }

        # Determine filename using user identifier if provided
        if getattr(self, 'user_id', None):
            safe_id = ''.join(c for c in self.user_id if c.isalnum() or c in ('_', '-')).lower()
            filename = self.data_dir / f'sessions_{safe_id}.json'
        else:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = self.data_dir / f'sessions_{date_str}.json'

        if not os.path.exists(filename):
            with open(filename, "w") as f:
                json.dump([], f)

        with open(filename, "r") as f:
            daily_sessions = json.load(f)

        session_summary['session_id'] = len(daily_sessions) + 1
        if getattr(self, 'user_id', None):
            session_summary['user_id'] = self.user_id

        daily_sessions.append(session_summary)

        with open(filename, "w") as f:
            json.dump(daily_sessions, f, indent=4)

        # Save to all_sessions.json (master log)
        all_sessions_file = self.data_dir / "all_sessions.json"

        if not os.path.exists(all_sessions_file):
            with open(all_sessions_file, "w") as f:
                json.dump([], f)

        with open(all_sessions_file, "r") as f:
            all_sessions = json.load(f)

        all_sessions.append(session_summary)

        with open(all_sessions_file, "w") as f:
            json.dump(all_sessions, f, indent=4)

        print("\n✅ Conversation, brain profile, and envelope values saved successfully.")

    def run(self):
        """Run the main conversation loop."""
        self.print_greeting()

        while True:
            user_input = self.get_user_input()

            if user_input.lower() == 'bye':
                self.print_goodbye()
                break

            bot_response = self.process_user_input(user_input)
            print(f"Bot: {bot_response}")
