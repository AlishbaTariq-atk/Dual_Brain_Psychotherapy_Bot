"""Basic DBP Bot CLI entry point

Usage:
    python -m unified_dbp_bot.main chat [--user USER_ID]
    python -m unified_dbp_bot.main build-index

Commands:
    chat         Start an interactive console conversation with the bot.
    build-index  Re-scan therapy documents and rebuild the FAISS vector store.
"""
from __future__ import annotations

# ========== Standard Library Imports ========== #
import argparse
import pickle
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config_and_storage import Config
from .core_functions import Chatbot
from unified_dbp_bot.core_functions import build_vector_store as rebuild_index

def _load_vector_store(chatbot: Chatbot) -> None:
    """Load persisted FAISS index into chatbot.rag_system if present."""
    index_dir = Path(Config.DATA_DIR) / "faiss_index"
    if index_dir.exists():
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            chatbot.rag_system.vector_store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
            print(f"[INFO] Loaded vector store from {index_dir}")
        except Exception as exc:
            print(f"[WARN] Failed to load vector store ({exc}). You may need to rebuild it.")
    else:
        print("[INFO] No vector store found – you can build one with 'build-index'.")



def run_chat(user_id: str) -> None:
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

    bot = Chatbot()
    _load_vector_store(bot)
    bot.start_session(user_id)

    print("\nType your message. Type 'exit' or 'quit' to end the session.\n")
    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break
        if user_msg.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        bot_reply = bot.handle_user_message(user_msg)
        print(f"Bot: {bot_reply}\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified DBP Bot CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    chat_p = sub.add_parser("chat", help="Start interactive chat session")
    chat_p.add_argument("--user", dest="user_id", default="anon", help="User identifier")

    sub.add_parser("build-index", help="Rebuild FAISS vector store from documents")

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    if args.command == "chat":
        run_chat(args.user_id)
    elif args.command == "build-index":
        rebuild_index()
    else:  
        print("Unknown command", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()