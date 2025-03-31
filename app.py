import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Page setup
st.set_page_config(page_title="Simple Chatbot", page_icon="🤖")
st.title("Simple Chatbot")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    gemini_key = st.text_input("Gemini API Key:", type="password")
    openai_key = st.text_input("OpenAI API Key:", type="password")
    
    # Start chat button
    if st.button("Start Chat"):
        if gemini_key:
            try:
                # Initialize chat
                memory = ConversationBufferMemory(return_messages=True)
                llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=gemini_key
                )
                st.session_state.conversation = ConversationChain(
                    llm=llm,
                    memory=memory
                )
                st.session_state.chat_active = True
                st.success("Chat started!")
            except Exception as e:
                st.error(f"Error starting chat: {e}")
        else:
            st.error("Please enter your Gemini API key")
    
    # End chat button
    if st.button("End Chat"):
        if st.session_state.chat_active and openai_key:
            try:
                # Get conversation text
                conversation_text = ""
                for msg in st.session_state.messages:
                    sender = "User" if msg["role"] == "user" else "Bot"
                    conversation_text += f"{sender}: {msg['content']}\n"
                
                # Generate summary with OpenAI
                openai_llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    openai_api_key=openai_key
                )
                
                summary_prompt = f"Summarize this conversation in under 150 words: {conversation_text}"
                sentiment_prompt = f"Analyze the sentiment of this conversation (positive, negative, or neutral): {conversation_text}"
                
                summary = openai_llm.invoke(summary_prompt).content
                sentiment = openai_llm.invoke(sentiment_prompt).content
                
                st.session_state.summary = summary
                st.session_state.sentiment = sentiment
                st.session_state.chat_active = False
            except Exception as e:
                st.error(f"Error generating summary: {e}")
        elif not openai_key:
            st.error("Please enter your OpenAI API key")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display summary if chat has ended
if not st.session_state.chat_active and "summary" in st.session_state:
    st.header("Conversation Summary")
    st.write(st.session_state.summary)
    
    st.header("Sentiment Analysis")
    st.write(st.session_state.sentiment)

# Chat input
if st.session_state.chat_active:
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get bot response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation.predict(input=user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Enter your API keys and click 'Start Chat' to begin")
