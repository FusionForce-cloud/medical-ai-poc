import streamlit as st
from agents.crew import run_multi_agent_turn
from logging_config import logger

st.set_page_config(page_title="Post-Discharge Nephrology Assistant")

st.title("🩺 Post-Discharge Nephrology AI Assistant")
st.caption("This AI assistant is for educational purposes only. "
           "Always consult qualified healthcare professionals for medical advice.")


# ---------------------------
# Session memory initialization
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I'm your post-discharge care assistant. What's your full name?"}
    ]

if "patient_name" not in st.session_state:
    st.session_state.patient_name = None
# ---------------------------


# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add the user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process via multi-agent system
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = run_multi_agent_turn(prompt, st.session_state)
            logger.info(f"Final reply to user: {reply}")
            st.markdown(reply)

    # Add final reply to history
    st.session_state.messages.append({"role": "assistant", "content": reply})
