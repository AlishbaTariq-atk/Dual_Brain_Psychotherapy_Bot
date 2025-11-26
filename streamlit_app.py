import streamlit as st
import main  # reuse chatbot + saving logic in main.py
from conversation import ChatbotConversation
from vector_store import load_vector_store
from llm_handler import initialize_llm
from prompts import get_greeting_message, get_goodbye_message
import os, json, glob
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure chatbot initialized once (creates main.chatbot)
if "chatbot_initialized" not in st.session_state:
    main.initialize_chatbot()
    st.session_state.chatbot = main.chatbot  # reference for goodbye call
    # Add greeting message once
    greeting = get_greeting_message()
    st.session_state.chat_history = [{"user": "", "bot": greeting}]
    st.session_state.chatbot_initialized = True

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- SESSION SETUP ----
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

user_name = st.sidebar.text_input("User name / ID", value=st.session_state.get("user_name", ""))
user_name = user_name.strip()
st.session_state["user_name"] = user_name

# Block further options until we have a name
if user_name:
    session_action = st.sidebar.radio("Session", ("New", "Continue"), index=0)
    user_file = DATA_DIR / f"sessions_{''.join(c for c in user_name if c.isalnum() or c in ('_', '-')).lower()}.json"

    selected_session = None
    if session_action == "Continue" and user_file.exists():
        with open(user_file, "r") as f:
            sessions_list = json.load(f)
        if sessions_list:
            labels = [f"{i+1}: {s['timestamp']} ({len(s.get('chat_history', []))} msgs)" for i, s in enumerate(sessions_list)]
            sel_idx = st.sidebar.selectbox("Pick previous session", range(len(labels)), format_func=lambda i: labels[i])
            selected_session = sessions_list[sel_idx]
        else:
            st.sidebar.info("No previous sessions found; a new one will be created.")
    elif session_action == "Continue":
        st.sidebar.info("No session file yet; new will be created.")

    # Once set, mark session initialized
    if "session_ready" not in st.session_state and (session_action == "New" or selected_session):
        st.session_state.session_ready = True
        # Set user_id on chatbot
        if not getattr(st.session_state.chatbot, "user_id", None):
            st.session_state.chatbot.user_id = user_name
        # If continuing, preload chat history
        if selected_session:
            st.session_state.chatbot.chat_history = selected_session.get("chat_history", [])
            # convert to simple display history for UI
            simplified = []
            for item in selected_session.get("chat_history", []):
                if item["role"] == "user":
                    simplified.append({"user": item["parts"][0]["text"], "bot": ""})
                else:
                    if simplified:
                        simplified[-1]["bot"] = item["parts"][0]["text"]
            st.session_state.chat_history = simplified or st.session_state.chat_history

# If not ready, keep chat disabled regardless of mode
ready = st.session_state.get("session_ready", False)

mode = st.sidebar.radio("Select mode", ("Chat", "Admin"))

# ---------------- CHAT MODE ----------------
if mode == "Chat":
    st.title("Dual-Brain Psychology Chatbot")

    # Event handler for sending message
    def send_message():
        user_input = st.session_state.user_input  # get current value
        if user_input.strip() == "":
            return

        # Check for exit keyword
        if user_input.lower().strip() == "bye":
            # Get farewell string to show in UI
            farewell = get_goodbye_message()
            st.session_state.chatbot.print_goodbye()  # still saves session
            bot_response = farewell
            st.session_state.chat_history.append({"user": user_input, "bot": bot_response})
            st.success("Session saved. You may close the tab.")
            st.session_state.disable_input = True
            return

        # Inject user_id once if provided
        if st.session_state.get("user_name") and not getattr(st.session_state.chatbot, "user_id", None):
            st.session_state.chatbot.user_id = st.session_state["user_name"]

        # Get bot response via main.start_chatbot (ensures same pipeline)
        response = main.start_chatbot(user_input)
        # Append to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": response})
        # Clear the input field safely
        st.session_state.user_input = ""  # <-- safe here, inside callback

    # Text input widget (disable before ready or after bye)
    if not ready:
        st.warning("Please enter your User ID and choose to create a new session or continue an old one in the sidebar.")

    st.text_input(
        "You:",
        key="user_input",
        on_change=send_message,
        disabled=not ready or st.session_state.get("disable_input", False)
    )

    # Display chat history
    for chat in st.session_state.chat_history:
        # Skip empty user for greeting row
        if chat["user"]:
            st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

# ---------------- ADMIN MODE ---------------
else:
    st.title("Admin Panel – Session Explorer")

    DATA_DIR = Path("data")
    session_files = sorted(glob.glob(str(DATA_DIR / "sessions_*.json")), reverse=True)

    if not session_files:
        st.info("No session files found in 'data/'. Start some chats first.")
    else:
        file_choice = st.selectbox("Select session file", session_files)

        with open(file_choice, "r") as f:
            sessions = json.load(f)

        # Build human-readable labels
        options = [f"{i}: {s['timestamp']} – {len(s.get('chat_history', []))} msgs" for i, s in enumerate(sessions)]
        idx = st.selectbox("Select session", range(len(options)), format_func=lambda i: options[i])
        session = sessions[idx]

        st.subheader("Brain Profile")
        st.json(session.get("brain_profile", {}))

        st.subheader("Envelope Assessment")
        st.json(session.get("envelope_assessment", {}))

        # 2D transition matrix visual
        trans = session.get("brain_profile", {}).get("transitions", {})
        if trans:
            array = np.array(trans.get("normalized", []))
            if array.size:
                st.subheader("State Transition Matrix (normalized)")
                st.table(pd.DataFrame(array, index=trans.get("labels", []), columns=trans.get("labels", [])))

        # Chat history
        st.subheader("Chat History")
        for msg in session.get("chat_history", []):
            role = msg["role"].capitalize()
            text = msg["parts"][0]["text"] if msg.get("parts") else ""
            st.markdown(f"**{role}:** {text}")
