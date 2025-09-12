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
    page_title="MaterialSim AI Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Chat Container */
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        border: none;
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Message Styles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 15px 0;
        margin-left: 15%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        animation: slideInRight 0.3s ease-out;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #333;
        padding: 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 15px 0;
        margin-right: 15%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
        position: relative;
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from { transform: translateX(100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Input Styles */
    .stTextInput > div > div > input {
        font-size: 16px !important;
        padding: 15px 20px !important;
        border: 2px solid #e9ecef !important;
        border-radius: 25px !important;
        background: white !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 25px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Sidebar Styles */
    .stSidebar {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%) !important;
    }
    
    .stSidebar .stMarkdown {
        color: #495057 !important;
    }
    
    /* Status Indicators */
    .status-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
    }
    
    .status-card.error {
        border-left-color: #dc3545;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Scrollbar Styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important;
        border: 1px solid #c3e6cb !important;
        color: #155724 !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%) !important;
        border: 1px solid #f5c6cb !important;
        color: #721c24 !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    
    /* Info Messages */
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%) !important;
        border: 1px solid #bee5eb !important;
        color: #0c5460 !important;
        border-radius: 10px !important;
        padding: 15px !important;
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
    st.markdown('<h1 class="main-header">🧬 MaterialSim AI Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligent Interface for Computational Materials Science</p>', unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Pure AI</h4>
            <p>No hardcoding - AI handles everything</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🧬 MD Simulations</h4>
            <p>Molecular dynamics made simple</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Analysis</h4>
            <p>Automatic property calculation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h4>🌐 Web Interface</h4>
            <p>Beautiful, modern UI</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

def create_sidebar():
    """Create the sidebar with navigation."""
    st.sidebar.title("🧬 MaterialSim AI Agent")
    st.sidebar.markdown("---")
    
    # Agent status
    st.sidebar.markdown("### 🤖 Agent Status")
    if st.session_state.agent_initialized and st.session_state.agent:
        st.sidebar.markdown("""
        <div class="status-card">
            <h4>✅ Agent Ready</h4>
            <p>AI is ready to help with your simulations!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show available tools
        if hasattr(st.session_state.agent, 'tools'):
            st.sidebar.markdown(f"**🛠️ Available Tools:** {len(st.session_state.agent.tools)}")
            for i, tool in enumerate(st.session_state.agent.tools):
                tool_name = getattr(tool, 'name', f'Tool {i+1}')
                st.sidebar.write(f"• {tool_name}")
    else:
        st.sidebar.markdown("""
        <div class="status-card error">
            <h4>❌ Agent Not Ready</h4>
            <p>Please check your API keys and dependencies.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    if st.sidebar.button("🔄 Reinitialize Agent", use_container_width=True):
        st.session_state.agent = None
        st.session_state.agent_initialized = False
        st.rerun()
    
    # Example prompts
    st.sidebar.markdown("### 💡 Example Prompts")
    example_prompts = [
        "Run molecular dynamics simulation for H2O",
        "Analyze the RDF of silicon at 300K",
        "Predict elastic properties of aluminum",
        "Generate 3D structure of graphene"
    ]
    
    for prompt in example_prompts:
        if st.sidebar.button(f"💬 {prompt}", key=f"example_{prompt}", use_container_width=True):
            st.session_state.example_prompt = prompt
            st.rerun()
    
    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    if st.sidebar.button("🔑 API Keys", use_container_width=True):
        st.session_state.show_settings = True
        st.rerun()
    
    # Clear chat
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

def display_chat_interface():
    """Display the main chat interface."""
    st.markdown("## 💬 Materials AI Assistant")
    st.markdown("Ask me anything about materials science, simulations, or analysis!")
    
    # Handle example prompts
    if hasattr(st.session_state, 'example_prompt') and st.session_state.example_prompt:
        prompt = st.session_state.example_prompt
        st.session_state.example_prompt = None
        # Process the example prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response using the agent
        try:
            if st.session_state.agent:
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
    
    # Chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display welcome message if no messages
        if not st.session_state.messages:
            st.markdown("""
            <div class="assistant-message">
                <strong>🤖 AI Assistant:</strong> Hello! I'm your MaterialSim AI Assistant. 
                I can help you with molecular dynamics simulations, materials analysis, 
                and property predictions. Try asking me something like:
                <br><br>
                • "Run molecular dynamics simulation for H2O"<br>
                • "Analyze the RDF of silicon at 300K"<br>
                • "Predict elastic properties of aluminum"<br>
                • "Generate 3D structure of graphene"
            </div>
            """, unsafe_allow_html=True)
        
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
    st.markdown("### 🔑 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        openai_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Your OpenAI API key for LLM functionality"
        )
    
    with col2:
        mp_key = st.text_input(
            "Materials Project API Key",
            value=os.getenv("MP_API_KEY", ""),
            type="password",
            help="Your Materials Project API key for database queries"
        )
    
    if st.button("💾 Save API Keys", use_container_width=True):
        # Save to .env file
        with open('.env', 'w') as f:
            f.write(f"OPENAI_API_KEY={openai_key}\n")
            f.write(f"MP_API_KEY={mp_key}\n")
        st.success("API keys saved!")
        st.rerun()
    
    # Agent Test
    st.markdown("### 🧪 Agent Test")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Agent", use_container_width=True):
            try:
                if st.session_state.agent:
                    test_response = st.session_state.agent.run_simulation("Say 'Agent is working!'")
                    st.success(f"✅ Agent test successful: {test_response}")
                else:
                    st.error("❌ Agent not initialized")
            except Exception as e:
                st.error(f"❌ Agent test failed: {e}")
    
    with col2:
        if st.button("Reinitialize Agent", use_container_width=True):
            st.session_state.agent = None
            st.session_state.agent_initialized = False
            st.success("Agent reinitialized!")
            st.rerun()
    
    # System Info
    st.markdown("### 📊 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Python Version:**")
        st.code(f"{sys.version}")
    
    with col2:
        st.markdown("**Dependencies:**")
        try:
            import streamlit
            st.success("✅ Streamlit")
        except:
            st.error("❌ Streamlit")
        
        try:
            from materials_ai_agent import MaterialsAgent
            st.success("✅ MaterialSim AI Agent")
        except:
            st.error("❌ MaterialSim AI Agent")
    
    # Quick Start Guide
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **Set your API keys** above
    2. **Test the agent** to ensure it's working
    3. **Go back to chat** and start asking questions
    4. **Try example prompts** from the sidebar
    """)

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
        **MaterialSim AI Agent could not be initialized.**
        
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
