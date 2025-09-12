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

# Set theme to dark mode explicitly
st.markdown("""
<style>
    /* Force dark theme */
    .stApp {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    
    /* Override Streamlit's default text colors */
    .main .block-container {
        color: #ffffff !important;
        background-color: #0e1117 !important;
    }
    
    /* Ensure all text is visible */
    p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Override Streamlit's markdown text color */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Override Streamlit's main text color */
    .main .block-container p {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    
    /* Force all text to be white */
    .main * {
        color: #ffffff !important;
    }
    
    /* Override Streamlit's text color inheritance */
    .main .block-container * {
        color: #ffffff !important;
    }
    
    /* Ensure markdown content is visible */
    .main .block-container .stMarkdown {
        color: #ffffff !important;
    }
    
    .main .block-container .stMarkdown p {
        color: #ffffff !important;
    }
    
    .main .block-container .stMarkdown h1,
    .main .block-container .stMarkdown h2,
    .main .block-container .stMarkdown h3,
    .main .block-container .stMarkdown h4,
    .main .block-container .stMarkdown h5,
    .main .block-container .stMarkdown h6 {
        color: #ffffff !important;
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
        color: #b0b0b0;
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
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
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
        background: linear-gradient(135deg, #2d2d44 0%, #3a3a5c 100%);
        color: #ffffff;
        padding: 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 15px 0;
        margin-right: 15%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
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
        border: 2px solid #3a3a5c !important;
        border-radius: 25px !important;
        background: #2d2d44 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #b0b0b0 !important;
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
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%) !important;
    }
    
    .stSidebar .stMarkdown {
        color: #ffffff !important;
    }
    
    .stSidebar .stMarkdown p {
        color: #ffffff !important;
    }
    
    .stSidebar .stMarkdown h1,
    .stSidebar .stMarkdown h2,
    .stSidebar .stMarkdown h3,
    .stSidebar .stMarkdown h4,
    .stSidebar .stMarkdown h5,
    .stSidebar .stMarkdown h6 {
        color: #ffffff !important;
    }
    
    /* Override Streamlit's sidebar text colors */
    .stSidebar * {
        color: #ffffff !important;
    }
    
    /* Status Indicators */
    .status-card {
        background: #2d2d44;
        color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border-left: 4px solid #28a745;
    }
    
    .status-card.error {
        border-left-color: #dc3545;
    }
    
    /* Feature Cards */
    .feature-card {
        background: #2d2d44;
        color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #3a3a5c;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    
    .feature-card h4 {
        color: #ffffff !important;
    }
    
    .feature-card p {
        color: #b0b0b0 !important;
    }
    
    /* Scrollbar Styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #2d2d44;
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
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6b 100%) !important;
        border: 1px solid #2d4a6b !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    
    /* Additional dark theme overrides */
    .stSelectbox > div > div {
        background-color: #2d2d44 !important;
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div > div {
        color: #ffffff !important;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #2d2d44 !important;
        color: #ffffff !important;
        border: 2px solid #3a3a5c !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
    }
    
    /* Override Streamlit's default text colors in various components */
    .stAlert {
        color: #ffffff !important;
    }
    
    .stAlert p {
        color: #ffffff !important;
    }
    
    /* Code blocks */
    .stCode {
        background-color: #1e1e2e !important;
        color: #ffffff !important;
    }
    
    /* Tables */
    .stDataFrame {
        background-color: #2d2d44 !important;
        color: #ffffff !important;
    }
    
    /* Simulation Status */
    .simulation-status {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6b 100%);
        border: 1px solid #2d4a6b;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .simulation-status h3 {
        color: #ffffff !important;
        margin-bottom: 10px;
    }
    
    .simulation-status p {
        color: #b0b0b0 !important;
        margin: 5px 0;
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Success Message */
    .stSuccess {
        background: linear-gradient(135deg, #1e5f3f 0%, #2d6b4a 100%) !important;
        border: 1px solid #2d6b4a !important;
        color: #ffffff !important;
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
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'simulation_params' not in st.session_state:
        st.session_state.simulation_params = {}

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
    
    # Clean professional description
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0; padding: 1.5rem; background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%); border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);">
        <p style="font-size: 1.1rem; color: #b0b0b0; margin: 0; line-height: 1.6;">
            Advanced molecular dynamics simulations, materials analysis, and property predictions powered by AI. 
            Simply describe what you want to simulate or analyze, and let the AI handle the rest.
        </p>
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
    st.sidebar.markdown("### ⚡ Actions")
    if st.sidebar.button("🔄 Reinitialize Agent", use_container_width=True):
        st.session_state.agent = None
        st.session_state.agent_initialized = False
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Example prompts
    st.sidebar.markdown("### 💡 Quick Examples")
    example_prompts = [
        "Simulate H2O molecular dynamics",
        "Analyze silicon RDF at 300K",
        "Predict aluminum elastic properties",
        "Generate graphene structure"
    ]
    
    for prompt in example_prompts:
        if st.sidebar.button(f"💬 {prompt}", key=f"example_{prompt}", use_container_width=True):
            st.session_state.example_prompt = prompt
            st.rerun()
    
    # Settings
    st.sidebar.markdown("### ⚙️ Configuration")
    if st.sidebar.button("🔑 API Settings", use_container_width=True):
        st.session_state.show_settings = True
        st.rerun()

def is_simulation_request(prompt: str) -> bool:
    """Check if the prompt is a simulation request."""
    simulation_keywords = [
        "simulate", "simulation", "molecular dynamics", "md", "lammps",
        "run simulation", "run md", "molecular dynamics simulation"
    ]
    return any(keyword in prompt.lower() for keyword in simulation_keywords)

def parse_simulation_parameters(prompt: str) -> dict:
    """Parse simulation parameters from natural language."""
    import re
    
    # Extract material
    material = "Si"  # Default
    if "h2o" in prompt.lower() or "water" in prompt.lower():
        material = "H2O"
    elif "silicon" in prompt.lower() or "si" in prompt.lower():
        material = "Si"
    elif "aluminum" in prompt.lower() or "al" in prompt.lower():
        material = "Al"
    elif "copper" in prompt.lower() or "cu" in prompt.lower():
        material = "Cu"
    elif "iron" in prompt.lower() or "fe" in prompt.lower():
        material = "Fe"
    
    # Extract temperature
    temperature = 300.0  # Default
    temp_match = re.search(r'(\d+)\s*k', prompt.lower())
    if temp_match:
        temperature = float(temp_match.group(1))
    
    # Extract force field
    force_field = "tersoff"  # Default
    if "tersoff" in prompt.lower():
        force_field = "tersoff"
    elif "lennard" in prompt.lower() or "lj" in prompt.lower():
        force_field = "lj"
    elif "eam" in prompt.lower():
        force_field = "eam"
    
    # Extract steps
    n_steps = 10000  # Default
    steps_match = re.search(r'(\d+)\s*steps?', prompt.lower())
    if steps_match:
        n_steps = int(steps_match.group(1))
    
    return {
        "material": material,
        "temperature": temperature,
        "force_field": force_field,
        "n_steps": n_steps
    }

def show_simulation_confirmation(prompt: str):
    """Show simulation parameter confirmation dialog."""
    params = parse_simulation_parameters(prompt)
    
    st.markdown("### 🔬 Simulation Parameters")
    st.markdown("Please review and confirm the simulation parameters:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        material = st.text_input("Material", value=params["material"], key="sim_material")
        temperature = st.number_input("Temperature (K)", value=params["temperature"], min_value=1.0, max_value=5000.0, key="sim_temp")
    
    with col2:
        force_field = st.selectbox("Force Field", ["tersoff", "lj", "eam"], 
                                 index=["tersoff", "lj", "eam"].index(params["force_field"]), key="sim_ff")
        n_steps = st.number_input("Number of Steps", value=params["n_steps"], min_value=1000, max_value=1000000, key="sim_steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Run Simulation", use_container_width=True):
            run_simulation_with_progress(material, temperature, force_field, n_steps)
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.messages.append({"role": "assistant", "content": "Simulation cancelled."})
            st.rerun()
    
    with col3:
        if st.button("✏️ Modify Parameters", use_container_width=True):
            st.rerun()

def run_simulation_with_progress(material: str, temperature: float, force_field: str, n_steps: int):
    """Run simulation with progress bar and detailed feedback."""
    try:
        if not st.session_state.agent:
            st.error("Agent not initialized. Please check your API keys.")
            return
        
        # Store simulation parameters in session state
        st.session_state.simulation_running = True
        st.session_state.simulation_params = {
            "material": material,
            "temperature": temperature,
            "force_field": force_field,
            "n_steps": n_steps
        }
        
        # Create progress container
        st.markdown("### 🚀 Running Simulation...")
        
        # Step 1: Parsing parameters
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📋 Parsing simulation parameters...")
        progress_bar.progress(10)
        
        # Step 2: Creating structure
        status_text.text("🏗️ Creating atomic structure...")
        progress_bar.progress(25)
        
        # Step 3: Setting up simulation
        status_text.text("⚙️ Setting up LAMMPS input files...")
        progress_bar.progress(50)
        
        # Step 4: Running simulation
        status_text.text("🔄 Running molecular dynamics simulation...")
        progress_bar.progress(75)
        
        # Actually run the simulation
        with st.spinner("Simulation in progress..."):
            response = st.session_state.agent.run_simulation(
                f"Simulate {material} at {temperature}K using {force_field} potential for {n_steps} steps"
            )
        
        # Step 5: Complete
        status_text.text("✅ Simulation completed!")
        progress_bar.progress(100)
        
        # Add response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Show download options
        show_download_options(response)
        
        # Mark simulation as completed
        st.session_state.simulation_running = False
        
        # Show success message
        st.success(f"✅ Simulation completed successfully! {material} at {temperature}K with {n_steps} steps")
        
        # Auto-rerun to show results in chat
        st.rerun()
        
    except Exception as e:
        error_msg = f"❌ Simulation failed: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.session_state.simulation_running = False
        st.rerun()

def show_download_options(response: str):
    """Show download options for simulation files."""
    if "Directory:" in response:
        # Extract directory from response
        import re
        dir_match = re.search(r'Directory: (simulations[^\\n]*)', response)
        if dir_match:
            sim_dir = Path(dir_match.group(1))
            if sim_dir.exists():
                st.markdown("### 📁 Download Simulation Files")
                
                files = list(sim_dir.glob("*"))
                for file_path in files:
                    if file_path.is_file():
                        with open(file_path, "rb") as file:
                            st.download_button(
                                label=f"📄 {file_path.name}",
                                data=file.read(),
                                file_name=file_path.name,
                                mime="application/octet-stream"
                            )

def get_ai_response(prompt: str):
    """Get AI response for non-simulation requests."""
    try:
        if st.session_state.agent:
            with st.spinner("AI is thinking..."):
                response = st.session_state.agent.run_simulation(prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Check if this is an analysis request and show analysis options
            if is_analysis_request(prompt):
                show_analysis_options(prompt, response)
            
            st.rerun()
        else:
            st.error("Agent not initialized. Please check your API keys.")
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.rerun()

def is_analysis_request(prompt: str) -> bool:
    """Check if the prompt is an analysis request."""
    analysis_keywords = [
        "analyze", "analysis", "plot", "graph", "visualize", "rdf", "msd",
        "radial distribution", "mean squared displacement", "properties",
        "download", "result", "output", "file"
    ]
    return any(keyword in prompt.lower() for keyword in analysis_keywords)

def show_analysis_options(prompt: str, response: str):
    """Show analysis options for simulation results."""
    st.markdown("### 📊 Analysis Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Plot RDF", use_container_width=True):
            st.info("RDF analysis would be performed here")
    
    with col2:
        if st.button("📊 Plot MSD", use_container_width=True):
            st.info("MSD analysis would be performed here")
    
    with col3:
        if st.button("🌡️ Plot Temperature", use_container_width=True):
            st.info("Temperature analysis would be performed here")
    
    # Show additional analysis options
    st.markdown("#### 🔬 Advanced Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚡ Compute Properties", use_container_width=True):
            st.info("Property computation would be performed here")
    
    with col2:
        if st.button("📋 Generate Report", use_container_width=True):
            st.info("Report generation would be performed here")

def display_chat_interface():
    """Display the main chat interface."""
    
    # Check if simulation is running
    if hasattr(st.session_state, 'simulation_running') and st.session_state.simulation_running:
        params = st.session_state.simulation_params
        st.markdown(f"""
        <div class="simulation-status">
            <h3>🔄 Simulation in Progress...</h3>
            <p><strong>Material:</strong> {params['material']}</p>
            <p><strong>Temperature:</strong> {params['temperature']}K</p>
            <p><strong>Steps:</strong> {params['n_steps']}</p>
            <p><strong>Force Field:</strong> {params['force_field']}</p>
            <p><em>Please wait while the simulation runs...</em></p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Chat input - moved to top
    st.markdown("### 💬 Describe your materials science task")
    if prompt := st.chat_input("Type your message here... (e.g., 'Simulate H2O molecular dynamics at 300K')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Check if this is a simulation request
        if is_simulation_request(prompt):
            # Show parameter confirmation
            show_simulation_confirmation(prompt)
        else:
            # Regular chat response
            get_ai_response(prompt)
    
    # Handle example prompts
    if hasattr(st.session_state, 'example_prompt') and st.session_state.example_prompt:
        prompt = st.session_state.example_prompt
        st.session_state.example_prompt = None
        if is_simulation_request(prompt):
            show_simulation_confirmation(prompt)
        else:
            get_ai_response(prompt)
    
    # Chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display welcome message if no messages
        if not st.session_state.messages:
            st.markdown("""
            <div class="assistant-message">
                <strong>🤖 Welcome to MaterialSim AI Agent</strong><br><br>
                I'm your intelligent assistant for computational materials science. I can help you with:
                <br><br>
                <strong>🧬 Molecular Dynamics Simulations:</strong><br>
                • Set up and run MD simulations with parameter confirmation<br>
                • Monitor simulation progress with real-time feedback<br>
                • Download simulation results and output files<br><br>
                
                <strong>📊 Analysis & Visualization:</strong><br>
                • Analyze simulation results (RDF, MSD, properties)<br>
                • Generate plots and visualizations<br>
                • Compute materials properties<br><br>
                
                <strong>💬 Interactive Workflow:</strong><br>
                • Natural language interface for all operations<br>
                • Step-by-step parameter confirmation<br>
                • Professional progress tracking<br><br>
                
                <em>Try: "Simulate H2O molecular dynamics at 300K" or "Analyze the results"</em>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

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
