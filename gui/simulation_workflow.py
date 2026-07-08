"""Conversational simulation workflow for the Streamlit GUI."""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from gui.analysis import perform_msd_analysis, perform_rdf_analysis, perform_thermodynamic_analysis

_FORCE_FIELD_LABELS = {
    "tersoff": "tersoff", "eam": "eam", "reaxff": "reaxff", "meam": "meam",
    "lennard-jones": "lj", "lennard jones": "lj", "lj": "lj", "emt": "emt",
    "mace": "mace", "chgnet": "chgnet", "m3gnet": "m3gnet",
    "opls": "opls", "gaff": "gaff", "openff": "openff", "auto": "auto",
}


def _force_field_kind(label) -> str:
    """Map a human force-field label to a registered potential kind."""
    if not label:
        return "auto"
    return _FORCE_FIELD_LABELS.get(str(label).strip().lower(), "auto")


def is_simulation_request(prompt: str) -> bool:
    """Check if the prompt is a simulation request."""
    simulation_keywords = [
        "simulate", "simulation", "molecular dynamics", "md", "lammps",
        "run simulation", "run md", "molecular dynamics simulation"
    ]
    return any(keyword in prompt.lower() for keyword in simulation_keywords)

def parse_initial_simulation_params(prompt: str) -> dict:
    """Parse initial simulation parameters via the shared, non-hardcoded extractor.

    Uses the deterministic spec extractor rather than brittle substring matching
    (the old code matched 'si' inside many words and silently guessed). If no
    material can be resolved, ``material`` is returned empty so the workflow asks
    the user instead of guessing.
    """
    from materials_ai_agent.core.config import Config
    from materials_ai_agent.spec.extractor import extract_spec

    config = Config.from_env()
    try:
        spec = extract_spec(prompt, config)
    except Exception:
        return {"material": "", "temperature": config.default_temperature}

    material = spec.system.material or spec.system.smiles or spec.system.mp_material_id or ""
    if material in {"unresolved", "custom", "user", "uploaded"}:
        material = ""
    temperature = max(
        config.min_temperature, min(spec.ensemble.temperature, config.max_temperature)
    )
    return {
        "material": material,
        "temperature": temperature,
        "ensemble": spec.ensemble.name,
        "protocol": spec.protocol.name,
        "force_field": spec.potential.kind,
        "engine": spec.engine,
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
            workflow["step"] = 65
            response = """Great! Use the file uploader below to provide your structure (XYZ, CIF, POSCAR, PDB).

Once the file is uploaded, we'll continue to force-field selection."""
        elif any(word in prompt_lower for word in ["materials project", "mp", "database"]):
            workflow["structure_source"] = "material_project"
            workflow["step"] = 66
            response = f"""Perfect! I'll fetch a structure for **{workflow['material']}** from the Materials Project database when you continue.

You can also type a specific Materials Project id (e.g. `mp-1234`) in the box below, then press Enter or click **Use MP structure**."""
        else:
            response = "Please choose one of the options: Generate, Upload, or Materials Project. Just say which one you prefer."
    elif step == 65:
        response = "Upload your structure file using the uploader below, then we'll proceed to force-field selection."
    elif step == 66:
        mp_id = extract_mp_id_from_prompt(prompt)
        if mp_id:
            workflow["mp_material_id"] = mp_id
            try:
                path = fetch_mp_structure_for_workflow(workflow)
                workflow["structure_file"] = path
                workflow["structure_source"] = "material_project"
                workflow["step"] = 7
                response = f"""Loaded **{workflow['mp_material_id']}** from Materials Project.

**Step 7: Force Field Selection**
Which force field would you like to use?
- **Tersoff**: Good for covalent materials like Si, C, Ge
- **EAM**: Good for metals like Al, Cu, Fe, Ni
- **Lennard-Jones**: Good for noble gases and simple systems
- **ReaxFF**: Good for reactive systems with C, H, O, N

Which force field would you like to use?"""
            except Exception as exc:
                response = f"Could not fetch {mp_id} from Materials Project: {exc}"
        elif workflow.get("structure_file"):
            workflow["step"] = 7
            src = workflow.get("mp_material_id") or workflow["material"]
            response = f"""Structure ready from Materials Project ({src}).

**Step 7: Force Field Selection**
Which force field would you like to use?
- **Tersoff**: Good for covalent materials like Si, C, Ge
- **EAM**: Good for metals like Al, Cu, Fe, Ni
- **Lennard-Jones**: Good for noble gases and simple systems
- **ReaxFF**: Good for reactive systems with C, H, O, N

Which force field would you like to use?"""
        else:
            response = (
                "Enter a Materials Project id (e.g. mp-1234) or click **Use MP structure** "
                f"to fetch the lowest-energy structure for {workflow['material']}."
            )

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

        workflow["step"] = 8
        structure_note = workflow["structure_source"]
        if workflow.get("structure_file"):
            structure_note += f" ({Path(workflow['structure_file']).name})"
        if workflow.get("mp_material_id"):
            structure_note += f" ({workflow['mp_material_id']})"
        response = f"""Perfect! We'll use the {workflow['force_field']} force field.

**Simulation Summary:**
- **Material**: {workflow['material']}
- **Temperature**: {workflow['temperature']}K
- **Ensemble**: {workflow['ensemble']}
- **Thermostat**: {workflow['thermostat']}
- **Timestep**: {workflow['timestep']} ps
- **Steps**: {workflow['n_steps']:,}
- **Total Time**: {workflow['timestep'] * workflow['n_steps']:.2f} ps
- **Structure**: {structure_note}
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
            # Check if user specified a specific analysis
            if "rdf" in prompt_lower or "radial distribution" in prompt_lower:
                # Perform RDF analysis
                response = perform_rdf_analysis()
            elif "msd" in prompt_lower or "mean squared displacement" in prompt_lower:
                # Perform MSD analysis
                response = perform_msd_analysis()
            elif "temperature" in prompt_lower or "energy" in prompt_lower:
                # Perform temperature/energy analysis
                response = perform_thermodynamic_analysis()
            else:
                # Show analysis options
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


def extract_mp_id_from_prompt(prompt: str) -> str | None:
    """Extract a Materials Project id from user text."""
    m = re.search(r"\b(mp-\d+)\b", prompt, re.I)
    return m.group(1) if m else None


def _structure_upload_dir() -> Path:
    root = Path("simulations") / "uploaded_structures"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_uploaded_structure(uploaded_file) -> str:
    """Persist a Streamlit uploaded file and return its path."""
    dest = _structure_upload_dir() / uploaded_file.name
    dest.write_bytes(uploaded_file.getvalue())
    return str(dest.resolve())


def fetch_mp_structure_for_workflow(workflow: dict) -> str:
    """Fetch an MP structure, save it locally, and return the file path."""
    from materials_ai_agent.core.config import Config
    from materials_ai_agent.mp_structure import fetch_mp_structure
    from ase.io import write

    config = Config.from_env()
    identifier = workflow.get("mp_material_id") or workflow["material"]
    atoms = fetch_mp_structure(identifier, api_key=config.mp_api_key)
    mp_id = atoms.info.get("mp_material_id", identifier)
    workflow["mp_material_id"] = mp_id
    dest = _structure_upload_dir() / f"{mp_id}.cif"
    write(str(dest), atoms, format="cif")
    return str(dest.resolve())


def render_structure_capture() -> None:
    """Show upload / Materials Project controls during structure setup."""
    workflow = st.session_state.simulation_workflow
    step = workflow.get("step", 0)

    if step == 65:
        st.markdown("#### Upload structure")
        uploaded = st.file_uploader(
            "Structure file (XYZ, CIF, POSCAR, PDB)",
            type=["xyz", "cif", "poscar", "vasp", "pdb"],
            key="sim_structure_upload",
        )
        if uploaded is not None:
            path = save_uploaded_structure(uploaded)
            workflow["structure_file"] = path
            workflow["structure_source"] = "upload"
            workflow["step"] = 7
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"Structure **{uploaded.name}** loaded ({path}).\n\n"
                    "**Step 7: Force Field Selection**\n"
                    "Which force field would you like? Tersoff, EAM, Lennard-Jones, or ReaxFF."
                ),
            })
            st.rerun()

    elif step == 66:
        st.markdown("#### Materials Project structure")
        mp_input = st.text_input(
            "Materials Project id (optional)",
            value=workflow.get("mp_material_id") or "",
            placeholder="mp-1234",
            key="sim_mp_id_input",
        )
        if mp_input:
            workflow["mp_material_id"] = mp_input.strip()

        if st.button("Use MP structure", key="sim_fetch_mp"):
            try:
                path = fetch_mp_structure_for_workflow(workflow)
                workflow["structure_file"] = path
                workflow["structure_source"] = "material_project"
                workflow["step"] = 7
                mp_id = workflow.get("mp_material_id", workflow["material"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"Loaded **{mp_id}** from Materials Project.\n\n"
                        "**Step 7: Force Field Selection**\n"
                        "Which force field would you like? Tersoff, EAM, Lennard-Jones, or ReaxFF."
                    ),
                })
                st.rerun()
            except Exception as exc:
                st.error(f"Could not fetch from Materials Project: {exc}")


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
        
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run the real molecular dynamics simulation.
        with st.spinner("Running molecular dynamics simulation..."):
            status_text.text("🔄 Running molecular dynamics simulation...")
            progress_bar.progress(25)

            from materials_ai_agent.simple_simulation import run_simple_simulation
            from materials_ai_agent.structure_builder import normalize_structure_source

            run_kwargs = {
                "material": workflow["material"],
                "temperature": workflow["temperature"],
                "n_steps": workflow["n_steps"],
                "force_field": _force_field_kind(workflow.get("force_field")),
                "ensemble": workflow.get("ensemble"),
                "thermostat": workflow.get("thermostat"),
                "timestep": workflow.get("timestep"),
                "engine": workflow.get("engine"),
                "protocol": workflow.get("protocol"),
                "output_frequency": 100,
                "structure_source": normalize_structure_source(
                    workflow.get("structure_source", "generate")
                ),
            }
            if workflow.get("structure_file"):
                run_kwargs["structure_file"] = workflow["structure_file"]
            if workflow.get("mp_material_id"):
                run_kwargs["mp_material_id"] = workflow["mp_material_id"]

            result = run_simple_simulation(**run_kwargs)

            if result["success"]:
                st.session_state['last_sim_dir'] = result['simulation_directory']
                response = (
                    f"✅ {result['message']}\n\n"
                    f"Directory: {result['simulation_directory']}\n"
                    f"Frames written: {result.get('n_frames')}\n"
                    f"Output files: {result['output_files']}"
                )
            elif result.get("needs_clarification"):
                response = f"ℹ️ I need more information before running: {result.get('error')}"
            else:
                response = f"❌ Simulation failed: {result.get('error')}"

            status_text.text("✅ Simulation completed!")
            progress_bar.progress(100)
        
        # Add response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Show download options
        show_download_options(response)
        
        # Mark simulation as completed
        st.session_state.simulation_running = False
        
        # Update workflow with actual simulation results
        if result["success"]:
            # Extract actual temperature and n_steps from the result
            actual_temp = result.get("temperature", workflow["temperature"])
            actual_n_steps = result.get("n_steps", workflow["n_steps"])
            
            # Update workflow state with actual values
            st.session_state.simulation_workflow["temperature"] = actual_temp
            st.session_state.simulation_workflow["n_steps"] = actual_n_steps
            
            # Update success message with actual values
            st.success(f"✅ Simulation completed successfully! {workflow['material']} at {actual_temp}K with {actual_n_steps} steps")
        else:
            st.error(f"❌ Simulation failed: {result.get('error', 'Unknown error')}")
        
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