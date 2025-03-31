import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os

# Page configuration
st.set_page_config(page_title="Langchain Chatbot", page_icon="🤖")
st.title("Langchain Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(return_messages=True)
if "conversation_active" not in st.session_state:
    st.session_state.conversation_active = False
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# Function to initialize the chatbot with Gemini
def initialize_gemini_chat():
    os.environ["GOOGLE_API_KEY"] = st.session_state.gemini_api_key
    
    # Create Gemini chat instance
    prompt_template = """
    You are a helpful and friendly assistant. The conversation history is:
    {history}
    Human: {input}
    AI:"""
    
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], 
        template=prompt_template
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    
    st.session_state.conversation = ConversationChain(
        llm=llm, 
        prompt=PROMPT,
        memory=st.session_state.conversation_memory,
        verbose=True
    )
    
    st.session_state.conversation_active = True
    st.success("Chatbot initialized! You can start chatting now.")

# Function to generate a summary and sentiment analysis using OpenAI
def get_conversation_summary():
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    
    # Extract conversation history
    history = ""
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Assistant"
        history += f"{role}: {message['content']}\n"
    
    # Create OpenAI instance
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Generate summary
    summary_prompt = f"""
    Please provide a summary of the following conversation in under 150 words:
    
    {history}
    """
    
    summary_response = llm.invoke(summary_prompt)
    
    # Generate sentiment analysis
    sentiment_prompt = f"""
    Please provide a short sentiment analysis of the following conversation. 
    Was it positive, negative, or neutral? What was the overall tone and mood?
    
    {history}
    """
    
    sentiment_response = llm.invoke(sentiment_prompt)
    
    return summary_response.content, sentiment_response.content

# API Key input
with st.sidebar:
    st.header("API Keys")
    
    # Gemini API Key
    gemini_key = st.text_input("Enter your Gemini API Key:", type="password")
    if gemini_key:
        st.session_state.gemini_api_key = gemini_key
    
    # OpenAI API Key (for summary)
    openai_key = st.text_input("Enter your OpenAI API Key (for summary):", type="password")
    if openai_key:
        st.session_state.openai_api_key = openai_key
    
    # Start button
    if st.button("Start Chat") and st.session_state.gemini_api_key:
        initialize_gemini_chat()
    
    # End button
    if st.button("End Chat") and st.session_state.conversation_active:
        if st.session_state.openai_api_key:
            with st.spinner("Generating conversation summary..."):
                summary, sentiment = get_conversation_summary()
                st.session_state.summary = summary
                st.session_state.sentiment = sentiment
                st.session_state.conversation_active = False
        else:
            st.error("Please provide an OpenAI API key for generating the summary.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display summary and sentiment after ending chat
if not st.session_state.conversation_active and "summary" in st.session_state:
    st.header("Conversation Summary")
    st.write(st.session_state.summary)
    
    st.header("Sentiment Analysis")
    st.write(st.session_state.sentiment)

# Chat input
if st.session_state.conversation_active:
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.predict(input=user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
else:
    st.info("Enter your Gemini API key and click 'Start Chat' to begin the conversation.")