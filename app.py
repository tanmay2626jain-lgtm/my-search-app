
app.py
import os
import streamlit as st
from google import genai
from google.genai import types

# 1. Initialize the Gemini Client
# The SDK automatically pulls the GEMINI_API_KEY environment variable.
try:
    client = genai.Client()
except Exception as e:
    st.error(f"Failed to initialize Gemini Client. Is your GEMINI_API_KEY set? Error: {e}")
    st.stop()

# 2. Configure Streamlit Page Setup
st.set_page_config(page_title="Gemini Conversational Search", page_icon="", layout="centered")
st.title(" Gemini Conversational Search")
st.caption("A smart chat box powered by Gemini with live web-grounded search.")

# 3. Initialize Session State for Chat History
if "chats" not in st.session_state:
    # We initialize a Gemini chat session with the Google Search tool enabled
    st.session_state.chats = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an advanced conversational search engine assistant. "
                "Always lean on your search tool to provide the most accurate, up-to-date information. "
                "Synthesize the search results cleanly and provide clear context."
            ),
            # Enabling Google Search Grounding natively
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are search metadata sources, display them nicely
        if "sources" in message and message["sources"]:
            with st.expander("Sources Cited"):
                for source in message["sources"]:
                    st.write(f"- [{source['title']}]({source['url']})")

# 5. Handle User Input
if user_query := st.chat_input("Ask me anything (e.g., 'What happened in the latest space launch?')"):
    
    # Display user message instantly
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Searching the web and thinking..."):
            try:
                # Send message through the stateful chat object
                response = st.session_state.chats.send_message(user_query)
                
                # Extract text response
                assistant_text = response.text
                response_placeholder.markdown(assistant_text)
                
                # Extract grounding metadata (search sources) if they exist
                sources_list = []
                if (response.candidates and 
                    response.candidates[0].grounding_metadata and 
                    response.candidates[0].grounding_metadata.grounding_chunks):
                    
                    chunks = response.candidates[0].grounding_metadata.grounding_chunks
                    for chunk in chunks:
                        if chunk.web:
                            sources_list.append({
                                "title": chunk.web.title,
                                "url": chunk.web.uri
                            })
                
                # Display sources if found
                if sources_list:
                    with st.expander("Sources Cited"):
                        for source in sources_list:
                            st.write(f"- [{source['title']}]({source['url']})")
                
                # Save assistant response to session state history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": assistant_text,
                    "sources": sources_list
                })
                
            except Exception as e:
                st.error(f"An error occurred while fetching the response: {e}")

