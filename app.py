import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Custom CSS styling
st.markdown("""
<style>
    /* ... (keep existing styles the same) ... */
</style>
""", unsafe_allow_html=True)

st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    huggingface_api_key = st.text_input("Hugging Face API Key", type="password")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-ai/deepseek-coder-1.3b-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Hugging Face](https://huggingface.co/) | [LangChain](https://python.langchain.com/)")

# Initialize the Hugging Face model
def get_llm():
    if not huggingface_api_key:
        st.error("🔑 Please enter your Hugging Face API key")
        return None
    
    return HuggingFaceHub(
        repo_id=selected_model,
        model_kwargs={
            "temperature": 0.3,
            "max_new_tokens": 1024
        },
        huggingfacehub_api_token=huggingface_api_key
    )

# System prompt configuration
system_prompt = """You are an expert AI coding assistant. Provide concise, correct solutions
with strategic print statements for debugging. Always respond in English."""

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt):
    llm = get_llm()
    if not llm:
        return None
    
    processing_pipeline = PromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
    return processing_pipeline.invoke({"prompt": prompt})

def build_prompt():
    prompt = f"System: {system_prompt}\n\n"
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "ai":
            prompt += f"Assistant: {msg['content']}\n"
    return prompt

if user_query:
    if not huggingface_api_key:
        st.error("🔑 Please enter your Hugging Face API key in the sidebar")
        st.stop()
    
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        full_prompt = build_prompt() + f"User: {user_query}\nAssistant:"
        ai_response = generate_ai_response(full_prompt)
    
    if ai_response:
        # Add AI response to log
        st.session_state.message_log.append({"role": "ai", "content": ai_response})
        
        # Rerun to update chat display
        st.rerun()
    else:
        st.error("Failed to generate response. Please check your API key and model selection.")