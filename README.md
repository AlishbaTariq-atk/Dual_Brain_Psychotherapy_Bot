# Basic DBP Bot

A Dual Brain Psychology (DBP) chatbot for emotional state detection, brain dominance assessment, and wellness support, featuring Retrieval-Augmented Generation (RAG) and JSON-based state transition logging.

## Features
- **Emotional State Detection**: Detects stressed vs. mature user states using pattern-based logic.
- **Brain Dominance Assessment**: Assesses logical vs. emotional dominance in user messages.
- **RAG (Retrieval-Augmented Generation)**: Integrates context from therapy documents using FAISS vector store and Google Generative AI Embeddings.
- **JSON State Logging**: All state transitions are logged as JSON in `state_transitions.json` for transparency and analysis.
- **Safety Filtering**: Configurable content moderation for user and bot messages.
- **Minimal, Clean Codebase**: All logic is consolidated into `core_functions.py` and `config_and_storage.py` for maintainability.

## Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd DBP_Project_Internship/DBP_Bot_Logic/src/wp3/unified_dbp_bot
```

### 2. Install Dependencies
Use the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root with the following keys:
```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
PROMPTLAYER_API_KEY=your_promptlayer_api_key
GOOGLE_API_KEY=your_google_api_key
```

### 4. (Optional) Ingest Documents for RAG
Place .txt, .docx, or .pdf files in the `Books/` directory for retrieval-augmented responses.

## Usage

### Start the Chatbot CLI
From the `src/wp3` directory, run:
```bash
python -m unified_dbp_bot.main chat
```

### Interact
Type your message and press Enter. Type `exit` or `quit` to end the session.

## File Structure
- `core_functions.py`: Main chatbot logic, brain assessment, RAG, and pipeline.
- `config_and_storage.py`: Prompt templates, config, safety filtering, and JSON-based storage.
- `main.py`: CLI entry point.
- `state_transitions.json`: Stores all state transitions as JSON objects.
- `requirements.txt`: Python dependencies.

## Logging and Data
- All state transitions are logged in `state_transitions.json`.
- No SQL/SQLite database is used for state tracking (legacy code removed).
- Session IDs are always strings and unique per session.

## Security & Safety
- Safety filtering is enabled by default (see `config_and_storage.py`).
- API keys must be kept secret and **never** committed to version control.

