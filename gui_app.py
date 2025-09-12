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
    if 'simulation_workflow' not in st.session_state:
        st.session_state.simulation_workflow = {
            'step': 0,  # 0: material, 1: temperature, 2: ensemble, 3: thermostat, 4: timestep, 5: structure, 6: confirm, 7: running, 8: complete
            'material': '',
            'temperature': 300.0,
            'ensemble': 'NVT',
            'thermostat': 'Nose-Hoover',
            'timestep': 0.001,
            'n_steps': 10000,
            'force_field': 'tersoff',
            'structure_source': 'generate',  # 'generate', 'upload', 'material_project'
            'structure_file': None,
            'explanations_shown': set(),
            'user_confirmations': {}
        }

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

def parse_initial_simulation_params(prompt: str) -> dict:
    """Parse initial simulation parameters from natural language."""
    import re
    
    # Extract material
    material = ""  # Will be set by user
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
    
    return {
        "material": material,
        "temperature": temperature
    }

def start_interactive_simulation_workflow(prompt: str):
    """Start the interactive simulation workflow."""
    # Parse initial parameters from prompt
    initial_params = parse_initial_simulation_params(prompt)
    
    # Initialize workflow with parsed parameters
    if initial_params["material"]:
        st.session_state.simulation_workflow["material"] = initial_params["material"]
    if initial_params["temperature"]:
        st.session_state.simulation_workflow["temperature"] = initial_params["temperature"]
    
    # Start conversational workflow
    st.session_state.simulation_workflow["step"] = 1
    st.session_state.simulation_workflow["mode"] = "conversational"
    
    # Add assistant response to chat
    material = initial_params["material"] or "your material"
    temperature = initial_params["temperature"]
    
    response = f"""Great! I'll help you set up a simulation for {material} at {temperature}K. Let me ask you a few questions to configure the simulation properly.

**Step 1: Material Confirmation**
I detected you want to simulate {material}. Is this correct, or would you like to change the material? You can say:
- "Yes, that's correct" 
- "Change it to [material name]"
- "I want to simulate [different material]"

What would you like to do?"""
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response
    })
    st.rerun()

def handle_simulation_conversation(prompt: str):
    """Handle conversational simulation workflow."""
    workflow = st.session_state.simulation_workflow
    step = workflow["step"]
    prompt_lower = prompt.lower()
    
    if step == 1:  # Material confirmation
        if any(word in prompt_lower for word in ["yes", "correct", "right", "that's correct", "that is correct"]):
            # Move to temperature confirmation
            workflow["step"] = 2
            response = f"""Perfect! We'll simulate {workflow['material']}.

**Step 2: Temperature Confirmation**
I detected you want to simulate at {workflow['temperature']}K. Is this the temperature you want, or would you like to change it? You can say:
- "Yes, that's correct"
- "Change it to [temperature]K" 
- "I want [temperature]K"

What would you like to do?"""
        elif any(word in prompt_lower for word in ["change", "different", "want to simulate"]):
            # Extract new material
            new_material = extract_material_from_prompt(prompt)
            if new_material:
                workflow["material"] = new_material
                workflow["step"] = 2
                response = f"""Great! I'll update the material to {new_material}.

**Step 2: Temperature Confirmation**
I detected you want to simulate at {workflow['temperature']}K. Is this the temperature you want, or would you like to change it? You can say:
- "Yes, that's correct"
- "Change it to [temperature]K"
- "I want [temperature]K"

What would you like to do?"""
            else:
                response = "I didn't catch the material name. Could you please specify what material you'd like to simulate? For example: 'Change it to aluminum' or 'I want to simulate copper'."
        else:
            response = "I'm not sure what you mean. Please say 'Yes, that's correct' to confirm the material, or tell me what material you'd like to simulate instead."
    
    elif step == 2:  # Temperature confirmation
        if any(word in prompt_lower for word in ["yes", "correct", "right", "that's correct", "that is correct"]):
            # Move to ensemble selection
            workflow["step"] = 3
            response = f"""Excellent! We'll simulate {workflow['material']} at {workflow['temperature']}K.

**Step 3: Thermodynamic Ensemble**
Which thermodynamic ensemble would you like to use? The options are:
- **NVT** (Canonical): Constant number of particles, volume, and temperature. Good for studying properties at constant temperature.
- **NPT** (Isothermal-Isobaric): Constant number of particles, pressure, and temperature. Good for studying properties at constant pressure.
- **NVE** (Microcanonical): Constant number of particles, volume, and energy. Good for studying energy conservation.

Which ensemble would you prefer? Just say 'NVT', 'NPT', or 'NVE'."""
        elif any(word in prompt_lower for word in ["change", "different", "want"]):
            # Extract new temperature
            new_temp = extract_temperature_from_prompt(prompt)
            if new_temp:
                workflow["temperature"] = new_temp
                workflow["step"] = 3
                response = f"""Perfect! I'll update the temperature to {new_temp}K.

**Step 3: Thermodynamic Ensemble**
Which thermodynamic ensemble would you like to use? The options are:
- **NVT** (Canonical): Constant number of particles, volume, and temperature. Good for studying properties at constant temperature.
- **NPT** (Isothermal-Isobaric): Constant number of particles, pressure, and temperature. Good for studying properties at constant pressure.
- **NVE** (Microcanonical): Constant number of particles, volume, and energy. Good for studying energy conservation.

Which ensemble would you prefer? Just say 'NVT', 'NPT', or 'NVE'."""
            else:
                response = "I didn't catch the temperature. Could you please specify the temperature? For example: 'Change it to 500K' or 'I want 1000K'."
        else:
            response = "I'm not sure what you mean. Please say 'Yes, that's correct' to confirm the temperature, or tell me what temperature you'd like to use instead."
    
    elif step == 3:  # Ensemble selection
        if "nvt" in prompt_lower:
            workflow["ensemble"] = "NVT"
            workflow["step"] = 4
            response = f"""Great! We'll use the NVT ensemble.

**Step 4: Thermostat Selection**
For temperature control, which thermostat would you like to use?
- **Nose-Hoover**: Most accurate, recommended for most simulations
- **Berendsen**: Simple and fast, good for quick equilibration
- **Langevin**: Good for liquid simulations
- **None**: No thermostat (only for NVE ensemble)

Which thermostat would you prefer?"""
        elif "npt" in prompt_lower:
            workflow["ensemble"] = "NPT"
            workflow["step"] = 4
            response = f"""Excellent! We'll use the NPT ensemble.

**Step 4: Thermostat Selection**
For temperature control, which thermostat would you like to use?
- **Nose-Hoover**: Most accurate, recommended for most simulations
- **Berendsen**: Simple and fast, good for quick equilibration
- **Langevin**: Good for liquid simulations

Which thermostat would you prefer?"""
        elif "nve" in prompt_lower:
            workflow["ensemble"] = "NVE"
            workflow["thermostat"] = "None"
            workflow["step"] = 5
            response = f"""Perfect! We'll use the NVE ensemble (no thermostat needed).

**Step 5: Timestep and Simulation Length**
What timestep would you like to use? For {workflow['material']}, I recommend:
- **0.001 ps** for most simulations
- **0.0005 ps** for more accuracy
- **0.002 ps** for faster simulation

And how many steps would you like to run? For example:
- **10,000 steps** (~10 ps) for quick tests
- **100,000 steps** (~100 ps) for property calculations
- **1,000,000 steps** (~1 ns) for long simulations

What timestep and number of steps would you like?"""
        else:
            response = "Please choose one of the ensembles: NVT, NPT, or NVE. Just say the name of the ensemble you prefer."
    
    elif step == 4:  # Thermostat selection
        if "nose" in prompt_lower or "hoover" in prompt_lower:
            workflow["thermostat"] = "Nose-Hoover"
            workflow["step"] = 5
            response = f"""Excellent! We'll use the Nose-Hoover thermostat.

**Step 5: Timestep and Simulation Length**
What timestep would you like to use? For {workflow['material']}, I recommend:
- **0.001 ps** for most simulations
- **0.0005 ps** for more accuracy
- **0.002 ps** for faster simulation

And how many steps would you like to run? For example:
- **10,000 steps** (~10 ps) for quick tests
- **100,000 steps** (~100 ps) for property calculations
- **1,000,000 steps** (~1 ns) for long simulations

What timestep and number of steps would you like?"""
        elif "berendsen" in prompt_lower:
            workflow["thermostat"] = "Berendsen"
            workflow["step"] = 5
            response = f"""Great! We'll use the Berendsen thermostat.

**Step 5: Timestep and Simulation Length**
What timestep would you like to use? For {workflow['material']}, I recommend:
- **0.001 ps** for most simulations
- **0.0005 ps** for more accuracy
- **0.002 ps** for faster simulation

And how many steps would you like to run? For example:
- **10,000 steps** (~10 ps) for quick tests
- **100,000 steps** (~100 ps) for property calculations
- **1,000,000 steps** (~1 ns) for long simulations

What timestep and number of steps would you like?"""
        elif "langevin" in prompt_lower:
            workflow["thermostat"] = "Langevin"
            workflow["step"] = 5
            response = f"""Perfect! We'll use the Langevin thermostat.

**Step 5: Timestep and Simulation Length**
What timestep would you like to use? For {workflow['material']}, I recommend:
- **0.001 ps** for most simulations
- **0.0005 ps** for more accuracy
- **0.002 ps** for faster simulation

And how many steps would you like to run? For example:
- **10,000 steps** (~10 ps) for quick tests
- **100,000 steps** (~100 ps) for property calculations
- **1,000,000 steps** (~1 ns) for long simulations

What timestep and number of steps would you like?"""
        else:
            response = "Please choose one of the thermostats: Nose-Hoover, Berendsen, or Langevin. Just say the name of the thermostat you prefer."
    
    elif step == 5:  # Timestep and steps
        timestep = extract_timestep_from_prompt(prompt)
        n_steps = extract_steps_from_prompt(prompt)
        
        if timestep and n_steps:
            workflow["timestep"] = timestep
            workflow["n_steps"] = n_steps
            workflow["step"] = 6
            total_time = timestep * n_steps
            response = f"""Perfect! We'll use a timestep of {timestep} ps and run {n_steps:,} steps (total time: {total_time:.2f} ps).

**Step 6: Structure Source**
How would you like to provide the atomic structure?
- **Generate**: I'll create a standard crystal structure for {workflow['material']}
- **Upload**: You can upload your own structure file (XYZ, POSCAR, CIF, PDB)
- **Materials Project**: I can search and download from the Materials Project database

Which option would you prefer?"""
        else:
            response = "I need both a timestep and number of steps. Please specify both, for example: '0.001 ps and 10000 steps' or 'timestep 0.001 and 50000 steps'."
    
    elif step == 6:  # Structure source
        if any(word in prompt_lower for word in ["generate", "create", "standard"]):
            workflow["structure_source"] = "generate"
            workflow["step"] = 7
            response = f"""Excellent! I'll generate a standard crystal structure for {workflow['material']}.

**Step 7: Force Field Selection**
Which force field would you like to use?
- **Tersoff**: Good for covalent materials like Si, C, Ge
- **EAM**: Good for metals like Al, Cu, Fe, Ni
- **Lennard-Jones**: Good for noble gases and simple systems
- **ReaxFF**: Good for reactive systems with C, H, O, N

For {workflow['material']}, I recommend **Tersoff**. Which force field would you like to use?"""
        elif any(word in prompt_lower for word in ["upload", "file", "own"]):
            workflow["structure_source"] = "upload"
            workflow["step"] = 7
            response = f"""Great! You'll upload your own structure file.

**Step 7: Force Field Selection**
Which force field would you like to use?
- **Tersoff**: Good for covalent materials like Si, C, Ge
- **EAM**: Good for metals like Al, Cu, Fe, Ni
- **Lennard-Jones**: Good for noble gases and simple systems
- **ReaxFF**: Good for reactive systems with C, H, O, N

For {workflow['material']}, I recommend **Tersoff**. Which force field would you like to use?"""
        elif any(word in prompt_lower for word in ["materials project", "mp", "database"]):
            workflow["structure_source"] = "material_project"
            workflow["step"] = 7
            response = f"""Perfect! I'll search the Materials Project database for {workflow['material']}.

**Step 7: Force Field Selection**
Which force field would you like to use?
- **Tersoff**: Good for covalent materials like Si, C, Ge
- **EAM**: Good for metals like Al, Cu, Fe, Ni
- **Lennard-Jones**: Good for noble gases and simple systems
- **ReaxFF**: Good for reactive systems with C, H, O, N

For {workflow['material']}, I recommend **Tersoff**. Which force field would you like to use?"""
        else:
            response = "Please choose one of the options: Generate, Upload, or Materials Project. Just say which one you prefer."
    
    elif step == 7:  # Force field selection
        if "tersoff" in prompt_lower:
            workflow["force_field"] = "Tersoff"
        elif "eam" in prompt_lower:
            workflow["force_field"] = "EAM"
        elif "lennard" in prompt_lower or "lj" in prompt_lower:
            workflow["force_field"] = "Lennard-Jones"
        elif "reaxff" in prompt_lower or "reax" in prompt_lower:
            workflow["force_field"] = "ReaxFF"
        else:
            response = "Please choose one of the force fields: Tersoff, EAM, Lennard-Jones, or ReaxFF. Just say the name of the force field you prefer."
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
            return
        
        # Move to confirmation
        workflow["step"] = 8
        response = f"""Perfect! We'll use the {workflow['force_field']} force field.

**Simulation Summary:**
- **Material**: {workflow['material']}
- **Temperature**: {workflow['temperature']}K
- **Ensemble**: {workflow['ensemble']}
- **Thermostat**: {workflow['thermostat']}
- **Timestep**: {workflow['timestep']} ps
- **Steps**: {workflow['n_steps']:,}
- **Total Time**: {workflow['timestep'] * workflow['n_steps']:.2f} ps
- **Structure**: {workflow['structure_source']}
- **Force Field**: {workflow['force_field']}

Does everything look correct? Say 'Yes, run the simulation' to start, or tell me what you'd like to change."""
    
    elif step == 8:  # Final confirmation
        if any(word in prompt_lower for word in ["yes", "correct", "run", "start", "go"]):
            # Start simulation
            workflow["step"] = 9
            response = f"""Excellent! Starting the simulation now...

🚀 **Running Simulation...**
- Setting up atomic structure...
- Preparing LAMMPS input files...
- Running molecular dynamics simulation...
- Processing results...

This may take a few minutes. I'll let you know when it's complete!"""
            
            # Actually run the simulation here
            run_simulation_with_progress()
        else:
            response = "Please say 'Yes, run the simulation' to start, or tell me what parameter you'd like to change."
    
    elif step == 9:  # Post-simulation analysis
        if any(word in prompt_lower for word in ["analyze", "analysis", "rdf", "msd", "plot", "graph", "result", "results"]):
            # Handle analysis requests
            response = f"""Great! I can help you analyze the simulation results. The simulation has completed and I have the following output files:

📁 **Available Files:**
- `in.lammps` - LAMMPS input file
- `output.log` - Simulation log with thermodynamic data
- `structure.xyz` - Atomic trajectory file

🔬 **Analysis Options:**
- **RDF (Radial Distribution Function)**: Shows atomic structure and coordination
- **MSD (Mean Squared Displacement)**: Shows diffusion behavior
- **Temperature/Energy plots**: Shows thermodynamic properties
- **Structure visualization**: 3D atomic structure display

What would you like to analyze? Just say "RDF", "MSD", "temperature plot", or describe what you want to see."""
        elif any(word in prompt_lower for word in ["download", "files", "output"]):
            response = f"""I can help you download the simulation files. The following files are available:

📁 **Simulation Files:**
- `in.lammps` - LAMMPS input file
- `output.log` - Simulation log with thermodynamic data  
- `structure.xyz` - Atomic trajectory file

Would you like me to prepare these files for download, or would you prefer to analyze the results first?"""
        elif any(word in prompt_lower for word in ["new", "another", "different", "restart"]):
            # Reset workflow for new simulation
            workflow["step"] = 0
            response = f"""Sure! Let's start a new simulation. What material would you like to simulate this time?

Just tell me what you want to simulate, for example:
- "Simulate aluminum at 500K"
- "I want to run a simulation of water"
- "Simulate copper at room temperature"

What would you like to simulate?"""
        else:
            response = f"""The simulation is complete! You can now:

🔬 **Analyze results**: Say "analyze results", "RDF", "MSD", or "plot temperature"
📁 **Download files**: Say "download files" to get the output files
🔄 **New simulation**: Say "new simulation" to start over

What would you like to do?"""
    
    else:
        response = "I'm not sure what you mean. Please respond to the current question or say 'cancel' to start over."
    
    # Add response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

def extract_material_from_prompt(prompt: str) -> str:
    """Extract material from user prompt."""
    prompt_lower = prompt.lower()
    
    # Common materials
    if "silicon" in prompt_lower or "si" in prompt_lower:
        return "Si"
    elif "aluminum" in prompt_lower or "al" in prompt_lower:
        return "Al"
    elif "copper" in prompt_lower or "cu" in prompt_lower:
        return "Cu"
    elif "iron" in prompt_lower or "fe" in prompt_lower:
        return "Fe"
    elif "water" in prompt_lower or "h2o" in prompt_lower:
        return "H2O"
    elif "carbon" in prompt_lower or "c" in prompt_lower:
        return "C"
    
    return None

def extract_temperature_from_prompt(prompt: str) -> float:
    """Extract temperature from user prompt."""
    import re
    temp_match = re.search(r'(\d+)\s*k', prompt.lower())
    if temp_match:
        return float(temp_match.group(1))
    return None

def extract_timestep_from_prompt(prompt: str) -> float:
    """Extract timestep from user prompt."""
    import re
    timestep_match = re.search(r'(\d+\.?\d*)\s*ps', prompt.lower())
    if timestep_match:
        return float(timestep_match.group(1))
    return None

def extract_steps_from_prompt(prompt: str) -> int:
    """Extract number of steps from user prompt."""
    import re
    steps_match = re.search(r'(\d+)\s*steps?', prompt.lower())
    if steps_match:
        return int(steps_match.group(1))
    return None

def show_interactive_simulation_workflow():
    """Show the interactive simulation workflow."""
    # This function is no longer needed - everything is handled in the chat interface
    pass

def show_material_selection():
    """Step 1: Material selection."""
    st.markdown("### 🧬 Step 1: Select Material")
    st.markdown("What material would you like to simulate?")
    
    # Material input
    material = st.text_input(
        "Material Formula (e.g., Si, Al2O3, H2O, Fe)",
        value=st.session_state.simulation_workflow["material"],
        placeholder="Enter material formula..."
    )
    
    # Show help
    if st.button("❓ What materials can I simulate?"):
        show_material_help()
    
    # Common materials quick select
    st.markdown("**Quick Select:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Si (Silicon)", use_container_width=True):
            st.session_state.simulation_workflow["material"] = "Si"
            st.rerun()
    with col2:
        if st.button("Al (Aluminum)", use_container_width=True):
            st.session_state.simulation_workflow["material"] = "Al"
            st.rerun()
    with col3:
        if st.button("H2O (Water)", use_container_width=True):
            st.session_state.simulation_workflow["material"] = "H2O"
            st.rerun()
    with col4:
        if st.button("Fe (Iron)", use_container_width=True):
            st.session_state.simulation_workflow["material"] = "Fe"
            st.rerun()
    
    # Navigation
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 0
            st.session_state.messages.append({"role": "assistant", "content": "Simulation cancelled."})
            st.rerun()
    with col2:
        if st.button("➡️ Next: Temperature", use_container_width=True):
            if material:
                st.session_state.simulation_workflow["material"] = material
                st.session_state.simulation_workflow["step"] = 2
                st.rerun()
            else:
                st.error("Please enter a material formula.")

def show_temperature_selection():
    """Step 2: Temperature selection."""
    st.markdown("### 🌡️ Step 2: Set Temperature")
    st.markdown("What temperature would you like to simulate at?")
    
    # Temperature input
    temperature = st.number_input(
        "Temperature (K)",
        value=st.session_state.simulation_workflow["temperature"],
        min_value=1.0,
        max_value=5000.0,
        step=10.0,
        help="Temperature in Kelvin"
    )
    
    # Show help
    if st.button("❓ What temperature should I use?"):
        show_temperature_help()
    
    # Common temperatures
    st.markdown("**Common Temperatures:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("77K (Liquid N₂)", use_container_width=True):
            st.session_state.simulation_workflow["temperature"] = 77.0
            st.rerun()
    with col2:
        if st.button("300K (Room Temp)", use_container_width=True):
            st.session_state.simulation_workflow["temperature"] = 300.0
            st.rerun()
    with col3:
        if st.button("1000K (High Temp)", use_container_width=True):
            st.session_state.simulation_workflow["temperature"] = 1000.0
            st.rerun()
    with col4:
        if st.button("2000K (Very High)", use_container_width=True):
            st.session_state.simulation_workflow["temperature"] = 2000.0
            st.rerun()
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 1
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 0
            st.session_state.messages.append({"role": "assistant", "content": "Simulation cancelled."})
            st.rerun()
    with col3:
        if st.button("➡️ Next: Ensemble", use_container_width=True):
            st.session_state.simulation_workflow["temperature"] = temperature
            st.session_state.simulation_workflow["step"] = 3
            st.rerun()

def show_ensemble_selection():
    """Step 3: Ensemble selection."""
    st.markdown("### ⚖️ Step 3: Choose Thermodynamic Ensemble")
    st.markdown("Which thermodynamic ensemble would you like to use?")
    
    # Ensemble selection
    ensemble = st.selectbox(
        "Thermodynamic Ensemble",
        ["NVT", "NPT", "NVE"],
        index=["NVT", "NPT", "NVE"].index(st.session_state.simulation_workflow["ensemble"]),
        help="Choose the thermodynamic ensemble for your simulation"
    )
    
    # Show help
    if st.button("❓ What do NVT, NPT, and NVE mean?"):
        show_ensemble_help()
    
    # Show ensemble description
    if ensemble == "NVT":
        st.info("**NVT (Canonical Ensemble)**: Constant number of particles (N), volume (V), and temperature (T). Good for studying properties at constant temperature.")
    elif ensemble == "NPT":
        st.info("**NPT (Isothermal-Isobaric Ensemble)**: Constant number of particles (N), pressure (P), and temperature (T). Good for studying properties at constant pressure and temperature.")
    elif ensemble == "NVE":
        st.info("**NVE (Microcanonical Ensemble)**: Constant number of particles (N), volume (V), and energy (E). Good for studying energy conservation and dynamics.")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 1
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 0
            st.session_state.messages.append({"role": "assistant", "content": "Simulation cancelled."})
            st.rerun()
    with col3:
        if st.button("➡️ Next: Thermostat", use_container_width=True):
            st.session_state.simulation_workflow["ensemble"] = ensemble
            st.session_state.simulation_workflow["step"] = 3
            st.rerun()

def show_thermostat_selection():
    """Step 4: Thermostat selection."""
    st.markdown("### 🌡️ Step 4: Choose Thermostat")
    st.markdown("Which thermostat would you like to use for temperature control?")
    
    # Thermostat selection
    thermostat = st.selectbox(
        "Thermostat",
        ["Nose-Hoover", "Berendsen", "Langevin", "None"],
        index=["Nose-Hoover", "Berendsen", "Langevin", "None"].index(st.session_state.simulation_workflow["thermostat"]),
        help="Choose the thermostat for temperature control"
    )
    
    # Show help
    if st.button("❓ What is a thermostat?"):
        show_thermostat_help()
    
    # Show thermostat description
    if thermostat == "Nose-Hoover":
        st.info("**Nose-Hoover**: Extended system thermostat that provides proper canonical sampling. Recommended for most simulations.")
    elif thermostat == "Berendsen":
        st.info("**Berendsen**: Simple velocity rescaling thermostat. Fast but not strictly canonical.")
    elif thermostat == "Langevin":
        st.info("**Langevin**: Stochastic thermostat that adds random forces. Good for liquid simulations.")
    elif thermostat == "None":
        st.info("**None**: No thermostat. Use only for NVE ensemble or when temperature control is not needed.")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 2
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 0
            st.session_state.messages.append({"role": "assistant", "content": "Simulation cancelled."})
            st.rerun()
    with col3:
        if st.button("➡️ Next: Timestep", use_container_width=True):
            st.session_state.simulation_workflow["thermostat"] = thermostat
            st.session_state.simulation_workflow["step"] = 4
            st.rerun()

def show_timestep_selection():
    """Step 5: Timestep selection."""
    st.markdown("### ⏱️ Step 5: Set Timestep and Simulation Length")
    st.markdown("What timestep and simulation length would you like to use?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        timestep = st.number_input(
            "Timestep (ps)",
            value=st.session_state.simulation_workflow["timestep"],
            min_value=0.0001,
            max_value=0.01,
            step=0.0001,
            format="%.4f",
            help="Timestep in picoseconds"
        )
    
    with col2:
        n_steps = st.number_input(
            "Number of Steps",
            value=st.session_state.simulation_workflow["n_steps"],
            min_value=1000,
            max_value=1000000,
            step=1000,
            help="Total number of simulation steps"
        )
    
    # Show help
    if st.button("❓ What timestep should I use?"):
        show_timestep_help()
    
    # Show simulation time
    total_time = timestep * n_steps
    st.info(f"**Total simulation time**: {total_time:.2f} ps ({total_time/1000:.3f} ns)")
    
    # Common timestep recommendations
    st.markdown("**Timestep Recommendations:**")
    st.markdown("- **Metals**: 0.001-0.002 ps")
    st.markdown("- **Covalent materials**: 0.0005-0.001 ps")
    st.markdown("- **Liquids**: 0.0005-0.001 ps")
    st.markdown("- **Gases**: 0.001-0.005 ps")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 3
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 0
            st.session_state.messages.append({"role": "assistant", "content": "Simulation cancelled."})
            st.rerun()
    with col3:
        if st.button("➡️ Next: Structure", use_container_width=True):
            st.session_state.simulation_workflow["timestep"] = timestep
            st.session_state.simulation_workflow["n_steps"] = n_steps
            st.session_state.simulation_workflow["step"] = 5
            st.rerun()

def show_structure_selection():
    """Step 6: Structure selection."""
    st.markdown("### 🏗️ Step 6: Choose Structure Source")
    st.markdown("How would you like to provide the atomic structure?")
    
    # Structure source selection
    structure_source = st.radio(
        "Structure Source",
        ["Generate", "Upload File", "Materials Project Database"],
        index=["generate", "upload", "material_project"].index(st.session_state.simulation_workflow["structure_source"]),
        help="Choose how to provide the atomic structure"
    )
    
    # Show help
    if st.button("❓ What are these options?"):
        show_structure_help()
    
    if structure_source == "Generate":
        st.info("**Generate**: I'll create a standard crystal structure for your material.")
        st.markdown("**Available structures:**")
        st.markdown("- Si: Diamond cubic")
        st.markdown("- Al: Face-centered cubic")
        st.markdown("- Fe: Body-centered cubic")
        st.markdown("- H2O: Water molecule")
        
    elif structure_source == "Upload File":
        st.info("**Upload File**: Upload your own structure file (POSCAR, XYZ, etc.)")
        uploaded_file = st.file_uploader(
            "Choose structure file",
            type=['xyz', 'poscar', 'cif', 'pdb'],
            help="Supported formats: XYZ, POSCAR, CIF, PDB"
        )
        if uploaded_file:
            st.session_state.simulation_workflow["structure_file"] = uploaded_file
            st.success(f"Uploaded: {uploaded_file.name}")
    
    elif structure_source == "Materials Project Database":
        st.info("**Materials Project**: Search and download structures from the Materials Project database.")
        mp_query = st.text_input(
            "Search Materials Project",
            placeholder="e.g., mp-149 (Si), mp-1143 (Al2O3)",
            help="Enter Materials Project ID or search term"
        )
        if mp_query:
            st.info(f"Would search for: {mp_query}")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 4
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 0
            st.session_state.messages.append({"role": "assistant", "content": "Simulation cancelled."})
            st.rerun()
    with col3:
        if st.button("➡️ Next: Confirm", use_container_width=True):
            st.session_state.simulation_workflow["structure_source"] = structure_source.lower().replace(" ", "_")
            st.session_state.simulation_workflow["step"] = 6
            st.rerun()

def show_simulation_confirmation():
    """Step 7: Simulation confirmation."""
    st.markdown("### ✅ Step 7: Confirm Simulation Parameters")
    st.markdown("Please review your simulation parameters before running:")
    
    workflow = st.session_state.simulation_workflow
    
    # Display parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Parameters:**")
        st.markdown(f"- **Material**: {workflow['material']}")
        st.markdown(f"- **Temperature**: {workflow['temperature']} K")
        st.markdown(f"- **Ensemble**: {workflow['ensemble']}")
        st.markdown(f"- **Thermostat**: {workflow['thermostat']}")
    
    with col2:
        st.markdown("**Simulation Details:**")
        st.markdown(f"- **Timestep**: {workflow['timestep']} ps")
        st.markdown(f"- **Steps**: {workflow['n_steps']:,}")
        st.markdown(f"- **Total Time**: {workflow['timestep'] * workflow['n_steps']:.2f} ps")
        st.markdown(f"- **Structure**: {workflow['structure_source']}")
    
    # Force field selection
    st.markdown("**Force Field:**")
    force_field = st.selectbox(
        "Select Force Field",
        ["tersoff", "eam", "lj", "reaxff"],
        index=["tersoff", "eam", "lj", "reaxff"].index(workflow["force_field"]),
        help="Choose the appropriate force field for your material"
    )
    
    # Show force field info
    if force_field == "tersoff":
        st.info("**Tersoff**: Good for covalent materials like Si, C, Ge")
    elif force_field == "eam":
        st.info("**EAM**: Good for metals like Al, Cu, Fe, Ni")
    elif force_field == "lj":
        st.info("**Lennard-Jones**: Good for noble gases and simple systems")
    elif force_field == "reaxff":
        st.info("**ReaxFF**: Good for reactive systems with C, H, O, N")
    
    # Navigation
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 5
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 0
            st.session_state.messages.append({"role": "assistant", "content": "Simulation cancelled."})
            st.rerun()
    with col3:
        if st.button("✏️ Edit", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 0
            st.rerun()
    with col4:
        if st.button("🚀 Run Simulation", use_container_width=True):
            st.session_state.simulation_workflow["force_field"] = force_field
            st.session_state.simulation_workflow["step"] = 7
            run_simulation_with_progress()
            st.rerun()

def show_simulation_progress():
    """Step 8: Simulation progress."""
    st.markdown("### 🚀 Running Simulation...")
    
    workflow = st.session_state.simulation_workflow
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulation steps
    steps = [
        "Initializing simulation...",
        "Setting up atomic structure...",
        "Preparing LAMMPS input files...",
        "Running molecular dynamics...",
        "Processing results...",
        "Simulation complete!"
    ]
    
    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) / len(steps))
        
        # Simulate some processing time
        import time
        time.sleep(1)
    
    # Mark as complete
    st.session_state.simulation_workflow["step"] = 8
    st.rerun()

def show_simulation_complete():
    """Step 9: Simulation complete."""
    st.markdown("### ✅ Simulation Complete!")
    
    workflow = st.session_state.simulation_workflow
    
    # Show results summary
    st.success(f"Successfully completed {workflow['n_steps']:,} step MD simulation of {workflow['material']} at {workflow['temperature']}K")
    
    # Show simulation details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Simulation Summary:**")
        st.markdown(f"- Material: {workflow['material']}")
        st.markdown(f"- Temperature: {workflow['temperature']} K")
        st.markdown(f"- Ensemble: {workflow['ensemble']}")
        st.markdown(f"- Steps: {workflow['n_steps']:,}")
        st.markdown(f"- Total Time: {workflow['timestep'] * workflow['n_steps']:.2f} ps")
    
    with col2:
        st.markdown("**Output Files:**")
        st.markdown("- trajectory.xyz")
        st.markdown("- log.lammps")
        st.markdown("- data.lammps")
        st.markdown("- in.lammps")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📁 Download Files", use_container_width=True):
            st.info("Download functionality would be implemented here")
    
    with col2:
        if st.button("📊 Analyze Results", use_container_width=True):
            st.info("Analysis options would be shown here")
    
    with col3:
        if st.button("🔄 New Simulation", use_container_width=True):
            st.session_state.simulation_workflow["step"] = 0
            st.rerun()
    
    # Reset workflow
    if st.button("🏠 Back to Chat"):
        st.session_state.simulation_workflow["step"] = 0
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"Simulation completed! {workflow['material']} at {workflow['temperature']}K with {workflow['n_steps']:,} steps."
        })
        st.rerun()

def show_material_help():
    """Show help for material selection."""
    st.markdown("""
    **Materials You Can Simulate:**
    
    **Elements:**
    - **Si (Silicon)**: Diamond cubic structure, good for semiconductor studies
    - **Al (Aluminum)**: Face-centered cubic, good for metal studies
    - **Fe (Iron)**: Body-centered cubic, good for magnetic materials
    - **Cu (Copper)**: Face-centered cubic, good for electrical conductivity
    - **C (Carbon)**: Diamond, graphite, or graphene structures
    
    **Compounds:**
    - **H2O (Water)**: Molecular dynamics of water
    - **Al2O3 (Alumina)**: Ceramic material
    - **SiO2 (Silica)**: Glass and ceramic applications
    
    **Custom Materials:**
    - You can also enter any chemical formula
    - The system will try to generate an appropriate structure
    - For complex materials, consider uploading a structure file
    """)

def show_temperature_help():
    """Show help for temperature selection."""
    st.markdown("""
    **Temperature Guidelines:**
    
    **Low Temperatures (1-100K):**
    - Study quantum effects and low-temperature properties
    - Good for superconductors and quantum materials
    
    **Room Temperature (300K):**
    - Standard for most materials studies
    - Good for comparing with experimental data
    
    **High Temperatures (1000-3000K):**
    - Study thermal properties and phase transitions
    - Good for understanding melting behavior
    
    **Very High Temperatures (3000K+):**
    - Study extreme conditions
    - Good for plasma and high-energy physics
    
    **Material-Specific Recommendations:**
    - **Metals**: 300-1000K for most studies
    - **Semiconductors**: 77-500K for electronic properties
    - **Ceramics**: 300-2000K for thermal properties
    - **Polymers**: 200-500K for mechanical properties
    """)

def show_ensemble_help():
    """Show help for ensemble selection."""
    st.markdown("""
    **Thermodynamic Ensembles Explained:**
    
    **NVT (Canonical Ensemble):**
    - **What it means**: Constant Number of particles, Volume, and Temperature
    - **When to use**: Most common choice for studying properties at constant temperature
    - **Good for**: Structural properties, diffusion, phase transitions
    - **Example**: Studying how a material behaves at 300K
    
    **NPT (Isothermal-Isobaric Ensemble):**
    - **What it means**: Constant Number of particles, Pressure, and Temperature
    - **When to use**: When you want to study properties at constant pressure
    - **Good for**: Density changes, volume expansion, pressure effects
    - **Example**: Studying how a material expands when heated
    
    **NVE (Microcanonical Ensemble):**
    - **What it means**: Constant Number of particles, Volume, and Energy
    - **When to use**: When you want to study energy conservation
    - **Good for**: Dynamics, energy flow, isolated systems
    - **Example**: Studying how energy moves through a system
    """)

def show_thermostat_help():
    """Show help for thermostat selection."""
    st.markdown("""
    **Thermostats Explained:**
    
    **Nose-Hoover Thermostat:**
    - **How it works**: Adds an extra degree of freedom to control temperature
    - **Pros**: Provides proper canonical sampling, very accurate
    - **Cons**: Slightly more complex, can have oscillations
    - **Best for**: Most simulations, especially when you need accurate thermodynamics
    
    **Berendsen Thermostat:**
    - **How it works**: Rescales velocities to reach target temperature
    - **Pros**: Simple, fast, stable
    - **Cons**: Not strictly canonical, can suppress fluctuations
    - **Best for**: Quick equilibration, when you don't need perfect thermodynamics
    
    **Langevin Thermostat:**
    - **How it works**: Adds random forces to control temperature
    - **Pros**: Good for liquid simulations, handles complex systems well
    - **Cons**: Can be noisy, not always physically realistic
    - **Best for**: Liquid simulations, complex molecular systems
    
    **No Thermostat:**
    - **When to use**: Only for NVE ensemble or when temperature control isn't needed
    - **Pros**: No artificial temperature control
    - **Cons**: Temperature will drift, not suitable for most simulations
    """)

def show_timestep_help():
    """Show help for timestep selection."""
    st.markdown("""
    **Timestep Guidelines:**
    
    **What is a timestep?**
    - The time interval between each simulation step
    - Smaller timesteps = more accurate but slower
    - Larger timesteps = faster but less accurate
    
    **General Rule:**
    - Timestep should be 1/10th of the fastest vibration period
    - For most materials, this is around 0.001 ps
    
    **Material-Specific Recommendations:**
    - **Metals (Al, Cu, Fe)**: 0.001-0.002 ps
    - **Covalent materials (Si, C)**: 0.0005-0.001 ps
    - **Liquids (H2O)**: 0.0005-0.001 ps
    - **Gases**: 0.001-0.005 ps
    - **Light atoms (H)**: 0.0001-0.0005 ps
    
    **Simulation Length:**
    - **Short simulations (1-10 ps)**: Quick tests, equilibration
    - **Medium simulations (10-100 ps)**: Property calculations
    - **Long simulations (100+ ps)**: Diffusion, rare events
    
    **Total Time = Timestep × Number of Steps**
    """)

def show_structure_help():
    """Show help for structure selection."""
    st.markdown("""
    **Structure Options Explained:**
    
    **Generate Structure:**
    - **What it does**: Creates a standard crystal structure for your material
    - **Pros**: Quick, no files needed, good for common materials
    - **Cons**: Limited to standard structures
    - **Best for**: Simple materials, quick tests
    
    **Upload File:**
    - **What it does**: Uses your own atomic structure file
    - **Supported formats**: XYZ, POSCAR, CIF, PDB
    - **Pros**: Complete control over structure, can use any material
    - **Cons**: Need to prepare structure file
    - **Best for**: Complex materials, specific structures
    
    **Materials Project Database:**
    - **What it does**: Downloads structures from the Materials Project
    - **Pros**: Real experimental structures, extensive database
    - **Cons**: Requires internet connection, may not have all materials
    - **Best for**: When you want experimental structures
    
    **Structure File Formats:**
    - **XYZ**: Simple format with atom types and coordinates
    - **POSCAR**: VASP format, includes cell parameters
    - **CIF**: Crystallographic Information File, very detailed
    - **PDB**: Protein Data Bank format, good for biomolecules
    """)

def run_simulation_with_progress():
    """Run simulation with progress bar and detailed feedback."""
    try:
        if not st.session_state.agent:
            st.error("Agent not initialized. Please check your API keys.")
            return
        
        workflow = st.session_state.simulation_workflow
        
        # Store simulation parameters in session state
        st.session_state.simulation_running = True
        st.session_state.simulation_params = {
            "material": workflow["material"],
            "temperature": workflow["temperature"],
            "force_field": workflow["force_field"],
            "n_steps": workflow["n_steps"]
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
                f"Simulate {workflow['material']} at {workflow['temperature']}K using {workflow['force_field']} potential for {workflow['n_steps']} steps"
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
        st.success(f"✅ Simulation completed successfully! {workflow['material']} at {workflow['temperature']}K with {workflow['n_steps']} steps")
        
        # Move to post-simulation analysis step
        st.session_state.simulation_workflow["step"] = 9
        
        # Auto-rerun to show results in chat
        st.rerun()
        
    except Exception as e:
        error_msg = f"❌ Simulation failed: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.session_state.simulation_running = False
        st.session_state.simulation_workflow["step"] = 0
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
                response = st.session_state.agent.chat(prompt)
            
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
    
    # Chat input - moved to top
    if prompt := st.chat_input(placeholder="💬 Describe your materials science task"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Check if we're in simulation workflow
        if st.session_state.simulation_workflow["step"] > 0:
            # Handle conversational simulation workflow
            handle_simulation_conversation(prompt)
        else:
            # Check if this is a new simulation request
            if is_simulation_request(prompt):
                # Start interactive simulation workflow
                start_interactive_simulation_workflow(prompt)
            else:
                # Regular chat response
                get_ai_response(prompt)
    
    
    # Chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display welcome message if no messages
        if not st.session_state.messages:
            st.markdown("""
            <div class="assistant-message">
                <strong>🤖 Welcome to MaterialSim AI Agent</strong><br><br>
                I'm your intelligent assistant for computational materials science. I can help you with molecular dynamics simulations and materials analysis through natural conversation.
                <br><br>
                <strong>🧬 Conversational Molecular Dynamics Simulations:</strong><br>
                • Natural language parameter collection with explanations<br>
                • Educational guidance for simulation parameters<br>
                • Structure input options (generate, upload, Materials Project)<br>
                • Real-time progress tracking and monitoring<br>
                • Post-simulation analysis and download options<br><br>
                
                <em>Just tell me what you'd like to do! For example, say "I want to simulate silicon at 300K" and I'll guide you through the entire process.</em>
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
