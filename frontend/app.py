"""
project/
│── app.py         # Streamlit frontend
│── backend.py         # Your modelling logic
│── models/        # (optional) fine-tuned weights, config, etc.
"""

import streamlit as st
import time
import random
from datetime import datetime
from backend import get_response

st.set_page_config(
    page_title="AI Assistant",
    page_icon = "💬💊",
    layout="centered")

# Custom CSS for asethetic styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(10deg, #607E98 0%, #FBE6A3 100%);
        #The gradient flows diagonally (with 10 degrees) from a colour a to colour b
    }
    
    .stChatMessage {
        background-color: white; 
        border-radius: 30px; #Rounded corners on each chat message 
        padding: 1rem; #Internal spacing (16px) inside each message bubble
        margin: 0.5rem 0; #Vertical spacing (8px) between messages, no horizontal margin
    }

    .stChatMessage p {
        color: black;
    }
    
    .stChatInputContainer {
        border-top: 2px solid #f0f0f0; #Light gray line (2px thick) separating the input area from chat messages
        padding-top: 1rem; #Spacing (16px) above the input box for breathing room
    }
    
    h1 {
        color: black;
        text-align: center;
        font-weight: 600;
        margin-bottom: 2rem; #Space (32px) below headings
    }
    
    .chat-header {
        text-align: center; 
        color: black;
        font-size: 2.5rem; #Large text (40px) for the main title
        margin-bottom: 0.5rem; Small spacing (8px) before subtitle
    }
    
    .subtitle {
        text-align: center;
        color: #607E98;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
            
    .chat-history-item {
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            cursor: pointer;
            background-color: rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="chat-header">💬💊 Your Bio-Medical Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your interactive conversation partner</div>', unsafe_allow_html=True)

# initialise session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# initialise first chat history
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.chat_history[st.session_state.current_chat_id] = {
        "messages": [{"role": "assistant", "content": "How can I help you?"}],
        "title": "New Chat",
        "timestamp": datetime.now()
    }

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

if "editing_chat_id" not in st.session_state:
    st.session_state.editing_chat_id = None

# Get current chat messages
current_messages = st.session_state.chat_history[st.session_state.current_chat_id]["messages"]

# Sidebar
with st.sidebar:
    # Clear chat button with confirmation
    st.markdown("### ⚙️ Options")
    
    if not st.session_state.confirm_clear:
        if st.button("🗑️ Clear Current Chat", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("⚠️ Are you sure? This will permanently delete this chat.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes", use_container_width=True):
                # Delete current chat
                if len(st.session_state.chat_history) > 1:
                    del st.session_state.chat_history[st.session_state.current_chat_id]
                    # Switch to most recent remaining chat
                    st.session_state.current_chat_id = sorted(
                        st.session_state.chat_history.keys(),
                        key=lambda x: st.session_state.chat_history[x]["timestamp"],
                        reverse=True
                    )[0]
                else:
                    # If it's the last chat, reset it
                    st.session_state.chat_history[st.session_state.current_chat_id] = {
                        "messages": [{"role": "assistant", "content": "How can I help you?"}],
                        "title": "New Chat",
                        "timestamp": datetime.now()
                    }
                st.session_state.confirm_clear = False
                st.rerun()
        
        with col2:
            if st.button("❌ No", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This biomedical assistant saves your conversation history. Start a new chat anytime!")

    st.markdown("----")
    
    st.markdown("### 💬 Chat Sessions")

    # New Chat button
    if st.button("➕ New Chat", use_container_width = True):
        new_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.current_chat_id = new_chat_id
        st.session_state.chat_history[new_chat_id] = {
            "messages": [{"role": "assistant", "content": "How can I help you?"}],
            "title": "New Chat",
            "timestamp": datetime.now()
        }
        st.session_state.confirm_clear = False
        st.rerun()

    st.markdown("----")

    # Display saved chats
    st.markdown("### 📚 Previous Chats")

    #Sort chats by timestamp (most recent first)
    sorted_chats = sorted(
        st.session_state.chat_history.items(),
        key=lambda x: x[1]["timestamp"],
        reverse=True
    )

    for chat_id, chat_data in sorted_chats:
        # Create a preview title from first user message or use default
        chat_title = chat_data["title"]
        if chat_title == "New Chat" and len(chat_data["messages"]) > 1:
            first_user_msg = next((m["content"] for m in chat_data["messages"] if m["role"] == "user"), None)
            if first_user_msg:
                chat_title = first_user_msg[:30] + "..." if len(first_user_msg) > 30 else first_user_msg

       # Check if this chat is being edited
        if st.session_state.editing_chat_id == chat_id:
            new_title = st.text_input(
                "Edit title:",
                value=chat_title,
                key=f"edit_{chat_id}",
                label_visibility="collapsed"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅", key=f"save_{chat_id}", use_container_width=True):
                    if new_title.strip():
                        st.session_state.chat_history[chat_id]["title"] = new_title.strip()
                    st.session_state.editing_chat_id = None
                    st.rerun()
            with col2:
                if st.button("❌", key=f"cancel_{chat_id}", use_container_width=True):
                    st.session_state.editing_chat_id = None
                    st.rerun()
        else:
            # Show chat with edit and select buttons
            col1, col2 = st.columns([4, 1])
            with col1:
                # Highlight current chat
                if chat_id == st.session_state.current_chat_id:
                    if st.button(f"🔵 **{chat_title}**", key=f"chat_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.confirm_clear = False
                        st.rerun()
                else:
                    if st.button(f"💬 {chat_title}", key=f"chat_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.confirm_clear = False
                        st.rerun()
            with col2:
                if st.button("✏️", key=f"edit_btn_{chat_id}", use_container_width=True):
                    st.session_state.editing_chat_id = chat_id
                    st.rerun()   

# Display chat history
for message in current_messages:
    avatar = "👤" if message["role"] == "user" else "🌟"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# User input
prompt = st.chat_input("Ask me something biomedical questions...")
if prompt:
    #Display user message
    with st.chat_message("user", avatar = "👤"):
        st.markdown(prompt)
        current_messages.append({"role": "user", "content": prompt})

    #Call BioBERT / RAG response
    response = get_response(prompt)

    # Display assistant response
    with st.chat_message('assistant', avatar = "🌟"):
        st.markdown(response)
    current_messages.append({"role": "assistant", "content": response})

# Update timestamp
st.session_state.chat_history[st.session_state.current_chat_id]["timestamp"] = datetime.now()

