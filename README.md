# Dual-Brain Psychology Chatbot

A Python-based chatbot that helps users explore their emotional patterns using the Dual Brain Psychology (DBP) framework. This project supports both command-line (CLI) and API (FastAPI) usage, integrates LLMs (OpenAI/Google), and provides entity extraction and brain profile analysis.

---

## Features
- **Empathetic DBP Dialogue:** All bot responses are grounded in Fred Schiffer’s Dual Brain Psychology, avoiding clinical or diagnostic language.
- **Named Entity Recognition:** Uses spaCy to extract entities from user messages.
- **LLM Integration:** Supports OpenAI (GPT-4o, GPT-3.5-turbo) and Google Generative AI via LangChain.
- **Knowledge Base Retrieval:** Uses FAISS vector store for context-aware responses.
- **Session Management:** Maintains per-user chat history and brain profile analysis (CLI and API).
- **Data Logging:** Saves chat sessions and brain profile summaries to JSON files in `data/`.
- **API & CLI:** Interact via REST API or command-line.

---

## Project Structure

```
Chat_Bot/
├── app.py                # FastAPI server (REST API)
├── main.py               # CLI entry point
├── conversation.py       # Chat session logic, history, greeting/goodbye
├── entity_extractor.py   # spaCy NER and brain profile analysis
├── brain_assessor.py     # Brain dominance/emotion analysis helpers
├── llm_handler.py        # LLM (OpenAI/Google) interface
├── prompts.py            # System, greeting, and goodbye prompts (DBP logic)
├── config.py             # Environment/config management
├── loader.py             # Knowledge base loading utilities
├── vector_store.py       # FAISS vector store setup and retrieval
├── requirements.txt      # Python dependencies
├── render.yaml           # Render.com deployment config (ignore for now)
├── data/                 # Knowledge files and session logs
├── specs/                # (Optional) API or other specs
├── start.sh              # (Optional) Startup script (ignore for now)
└── README.md             # This file
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://gitlab.com/omar.khan/dbpchat
cd DBP_Bot_Logic/Chat_Bot
```

### 2. Install Dependencies
It is strongly recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in `Chat_Bot/` with your API keys:
```
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-genai-key
HUGGINGFACE_HUB_TOKEN=your-huggingface-token
```

### 4. Prepare Knowledge Base
Place your knowledge/text files in the `data/` directory. Update paths in `config.py` if needed.

---

## Usage

### CLI Mode
```bash
python main.py
```
- The bot will greet you and you can start chatting.
- Type `bye` to end the session. Session summaries are saved in `data/`.

### API Mode (FastAPI)
```bash
uvicorn app:app --reload
```
- **POST /chat** — Start or continue a chat session. JSON body:
    ```json
    { "user_id": "string", "message": "string" }
    ```
- **Session End:** Send message `"bye"` to end and save the session in "data/" folder.

---

## Deployment
- **Render.com:** Use `render.yaml` for deployment. The app will start on port 10000.
- **Custom:** Use `uvicorn app:app --host 0.0.0.0 --port <PORT>` for self-hosting.

---

## Integration
- **Dart/Flutter or Other Clients:** Call the `/chat` endpoint with `user_id` and `message` for chat, and implement session logic client-side.

---

## Troubleshooting
- For dependency issues, recreate your virtual environment and reinstall requirements.
- Ensure all API keys are set in `.env`.
- If spaCy model errors occur, run:
  ```bash
  python -m spacy download en_core_web_sm
  ```
- For vector store errors, ensure knowledge files exist in `data/`.

---

## License
MIT License (or specify your license here)
