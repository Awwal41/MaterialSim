#!/usr/bin/env python3
"""
Materials AI Agent -  Frontend Interface
"""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Materials AI Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        border: 2px solid #1f77b4;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        background-color: #fafafa;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 15px;
        margin: 15px 0;
        margin-left: 20%;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .assistant-message {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 15px;
        margin: 15px 0;
        margin-right: 20%;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTextInput > div > div > input {
        font-size: 16px !important;
        padding: 12px !important;
        border: 2px solid #1f77b4 !important;
        border-radius: 10px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #28a745 !important;
        box-shadow: 0 0 0 2px rgba(40, 167, 69, 0.25) !important;
    }
    .stButton > button {
        background-color: #1f77b4 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }
    .stButton > button:hover {
        background-color: #28a745 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    .stSidebar {
        background-color: #f8f9fa !important;
    }
    .stSidebar .stMarkdown {
        color: #333 !important;
    }
    .stSuccess {
        background-color: #d4edda !important;
        border: 1px solid #c3e6cb !important;
        color: #155724 !important;
    }
    .stError {
        background-color: #f8d7da !important;
        border: 1px solid #f5c6cb !important;
        color: #721c24 !important;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False

def initialize_agent():
    """Initialize the Materials AI Agent."""
    if not st.session_state.agent_initialized:
        try:
            from materials_ai_agent import MaterialsAgent
            st.session_state.agent = MaterialsAgent()
            st.session_state.agent_initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize Materials AI Agent: {e}")
            st.error("Please check your API keys and dependencies.")
            return False
    return True

def create_header():
    """Create the main header."""
    st.markdown('<h1 class="main-header">🧬 Materials AI Agent</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Interface for Computational Materials Science")
    st.markdown("---")

def create_sidebar():
    """Create the sidebar with navigation."""
    st.sidebar.title("🧬 Materials AI Agent")
    st.sidebar.markdown("---")
    
    # Agent status
    st.sidebar.markdown("### Agent Status")
    if st.session_state.agent_initialized and st.session_state.agent:
        st.sidebar.success("✅ Agent Ready")
        
        # Show available tools
        if hasattr(st.session_state.agent, 'tools'):
            st.sidebar.markdown(f"**Available Tools:** {len(st.session_state.agent.tools)}")
            for i, tool in enumerate(st.session_state.agent.tools):
                tool_name = getattr(tool, 'name', f'Tool {i+1}')
                st.sidebar.write(f"• {tool_name}")
    else:
        st.sidebar.error("❌ Agent Not Ready")
    
    # Quick actions
    st.sidebar.markdown("### Quick Actions")
    if st.sidebar.button("🔄 Reinitialize Agent"):
        st.session_state.agent = None
        st.session_state.agent_initialized = False
        st.rerun()
    
    # Settings
    st.sidebar.markdown("### Settings")
    if st.sidebar.button("⚙️ API Keys"):
        st.session_state.show_settings = True
        st.rerun()

def display_chat_interface():
    """Display the main chat interface."""
    st.markdown("## 💬 Materials AI Assistant")
    st.markdown("Ask me anything about materials science, simulations, or analysis!")
    
    # Chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    st.markdown("### 💬 Ask me anything about materials science!")
    if prompt := st.chat_input("Type your message here... (e.g., 'Run molecular dynamics simulation for H2O')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response using the agent
        try:
            if st.session_state.agent:
                # Use the agent to process the request
                with st.spinner("AI is thinking..."):
                    response = st.session_state.agent.run_simulation(prompt)
                
                # Add assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Rerun to display the new message
                st.rerun()
            else:
                st.error("Agent not initialized. Please check your API keys.")
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun()

def display_settings():
    """Display settings page."""
    st.markdown("## ⚙️ Settings")
    
    # API Keys
    st.markdown("### API Configuration")
    
    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Your OpenAI API key for LLM functionality"
    )
    
    mp_key = st.text_input(
        "Materials Project API Key",
        value=os.getenv("MP_API_KEY", ""),
        type="password",
        help="Your Materials Project API key for database queries"
    )
    
    if st.button("💾 Save API Keys"):
        # Save to .env file
        with open('.env', 'w') as f:
            f.write(f"OPENAI_API_KEY={openai_key}\n")
            f.write(f"MP_API_KEY={mp_key}\n")
        st.success("API keys saved!")
        st.rerun()
    
    # Agent Test
    st.markdown("### Agent Test")
    if st.button("🧪 Test Agent"):
        try:
            if st.session_state.agent:
                test_response = st.session_state.agent.run_simulation("Say 'Agent is working!'")
                st.success(f"✅ Agent test successful: {test_response}")
            else:
                st.error("❌ Agent not initialized")
        except Exception as e:
            st.error(f"❌ Agent test failed: {e}")
    
    # System Info
    st.markdown("### System Information")
    st.code(f"Python Version: {sys.version}")
    
    # Dependencies
    st.markdown("**Dependencies:**")
    try:
        import streamlit
        st.success("✅ Streamlit")
    except:
        st.error("❌ Streamlit")
    
    try:
        from materials_ai_agent import MaterialsAgent
        st.success("✅ Materials AI Agent")
    except:
        st.error("❌ Materials AI Agent")

def main():
    """Main application."""
    initialize_session_state()
    
    # Create header
    create_header()
    
    # Create sidebar
    create_sidebar()
    
    # Initialize agent
    if not initialize_agent():
        st.error("""
        **Materials AI Agent could not be initialized.**
        
        Please check:
        1. Your API keys are set correctly
        2. All dependencies are installed
        3. The agent package is properly installed
        
        Go to Settings to configure your API keys.
        """)
        
        if st.button("⚙️ Go to Settings"):
            st.session_state.show_settings = True
            st.rerun()
        return
    
    # Check if settings should be shown
    if hasattr(st.session_state, 'show_settings') and st.session_state.show_settings:
        display_settings()
        if st.button("← Back to Chat"):
            st.session_state.show_settings = False
            st.rerun()
    else:
        # Main chat interface
        display_chat_interface()

if __name__ == "__main__":
    main()
