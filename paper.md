---
title: 'MaterialSim: An AI Agent for Automating Computational Materials Simulations'
tags:
  - Python
  - materials science
  - molecular dynamics
  - computational materials
  - large language models
  - agent
  - LAMMPS
  - ASE
  - voice interface
  - speech recognition
authors:
  - name: Awwal Oladipupo
    corresponding: true
    email: awwalola@umich.edu
    orcid: 0009-0006-3979-2078
    affiliation: "1"
  - name: Vallabh Vasudevan
    orcid: 0000-0001-7933-4924
    affiliation: "1"
  - name: Akhila Ponugoti
    orcid: 0009-0007-8947-7794
    affiliation: "2"
  - name: Toheeb Balogun
    orcid: 0009-0006-8505-3196
    affiliation: "3"
  - name: Jodie A. Yuwono
    orcid: 0000-0002-0915-0756
    affiliation: "4"
affiliations:
  - index: 1
    name: Department of Chemical Engineering, University of Michigan, Ann Arbor, MI, United States
    ror: 00jmfr291
  - index: 2
    name: Johns Hopkins University, Baltimore, MD, United States
    ror: 00za53h95
  - index: 3
    name: Department of Chemical Engineering, Louisiana State University, Baton Rouge, LA, United States
    ror: 05ect4e57
  - index: 4
    name: School of Chemical Engineering and Materials Science, The University of Adelaide, Adelaide, South Australia, Australia
    ror: 00892tw58
date: 8 April 2026
bibliography: paper.bib
---

# Summary

Discovering and understanding materials often relies on computer simulations that track how atoms move and interact over time. Setting up these simulations has traditionally required specialized expertise: choosing simulation software, interatomic models, numerical settings, and analysis steps, then wiring them together with scripts. These manual steps slow down discovery and limit the exploration of chemical space.

**MaterialSim** helps researchers drive such workflows through natural language—typed or spoken: an agent plans and calls tools for molecular dynamics, property analysis, literature-informed model selection, and optional machine learning, producing quantitative outputs and plots. A **JARVIS-style voice interface** in the Streamlit GUI [@streamlit2023] lets users speak commands (for example, “simulate copper at 300 K” or “analyze the latest results”); speech is transcribed, routed through the same agent orchestration layer as text, and responses are read aloud via text-to-speech while an animated heads-up display reflects listening, reasoning, and speaking states.

The platform runs **real molecular dynamics** through ASE [@larsen2017atomic] integrators and interatomic calculators (EMT for supported metals, Lennard-Jones fallback otherwise), writing genuine trajectory and thermodynamic logs—no synthetic mock outputs. An optional LAMMPS [@plimpton1995fast] subprocess path remains available for users with a local installation. Simulations can be launched from the **command-line interface** (`materials-agent run …`), Python API (`MaterialsAgent.run_simulation`), or voice/GUI modes without changing the underlying engine.

An **intelligent model router** classifies each request (simulation planning, analysis interpretation, literature review, property prediction, voice dialogue) and, when enabled, searches **arXiv** [@arxiv2007arxiv] and the open web for benchmarks and papers before selecting an appropriate hosted large language model (for example, `gpt-4o` vs `gpt-4o-mini` for latency-sensitive voice replies). Post-processing computes radial distribution functions, mean-squared displacement, and thermodynamic summaries from saved trajectories, with Matplotlib plots [@hunter2007matplotlib].

The software is developed in public as the GitHub repository [Awwal41/MaterialSim](https://github.com/Awwal41/MaterialSim). The Python import path is `materials_ai_agent`, the main class is `MaterialsAgent`, and the console entry point is `materials-agent` (distribution metadata in `setup.py` uses the name `materialsim-ai-agent`). Core dependencies include ASE, LangChain/LangGraph [@langchain2023; @langgraph2024], hosted LLMs via the OpenAI API [@openai2024api] (users supply keys; simulations and analysis work without them), Pymatgen [@ong2013python] for optional database tooling, and scikit-learn/PyTorch [@pedregosa2011scikit; @paszke2019pytorch] as optional ML extras. Voice dependencies (`edge-tts`, SpeechRecognition) are listed in `requirements_gui.txt`. **Materials Project** [@jain2013commentary] and **NOMAD** [@draxl2018nomad] queries are implemented when optional packages and API keys are present; **Open Catalyst** [@chanussot2021open] is recorded as not yet implemented. The Streamlit application (`gui_app.py`, launched via `launch_gui.py`) provides conversational and JARVIS voice modes; a CLI supports `run`, `analyze`, `predict`, `query`, and `interactive` subcommands. The project targets Python 3.8+.

# Statement of need

Computational materials research depends on reproducible, composable pipelines from structure retrieval to simulation and post-processing. High-throughput frameworks such as FireWorks within the Atomate ecosystem [@jain2015fireworks] and analysis toolkits such as MatMiner [@ward2018matminer] have lowered barriers for expert users, but they still assume substantial familiarity with workflow configuration and Python glue code. Recent work has shown that large language models can act as planners that call external tools in scientific settings, for example, ChemCrow augments models with chemistry tools [@bran2023chemcrow]—but comparable, materials, focused orchestration that is packaged for simulation-scale tasks (MD engines, materials databases, cluster-capable execution of those engines) remains comparatively scarce.

MaterialSim targets researchers, educators, and interdisciplinary teams who want to iterate quickly on simulation ideas without rewriting boilerplate for every study. It is designed for exploratory studies, teaching demonstrations, and rapid prototyping where the cost of manual pipeline construction is high. The **voice interface** lowers the interaction barrier further: hands-busy or accessibility-minded users can run full simulation-to-analysis workflows by speaking, while the CLI and Python API serve scripted and cluster-oriented use cases. The package emphasizes transparent composition of third-party tools rather than replacing domain codes: ASE and, when installed, LAMMPS remain the sources of truth for MD numerics; Pymatgen structures interoperate with database tooling.

# State of the field

Workflow automation in materials informatics is mature in high-throughput and database-centric settings. Atomate/FireWorks [@jain2015fireworks] provides powerful, production-oriented workflow management for density-functional-theory-centric pipelines; MatMiner [@ward2018matminer] focuses on featurization and machine-learning datasets from structures and calculations. These systems excel when users already know which codes and presets to chain. General scientific tool-augmented language agents [@bran2023chemcrow] demonstrate natural-language steering of tool ecosystems but are not specialized for materials simulation stacks, HPC submission, or typical MD observables.

MaterialSim occupies a complementary niche: **natural-language and voice orchestration of MD-centric materials simulations** with explicit bindings to ASE (primary engine) and optional LAMMPS, database-backed access where implemented (Materials Project and NOMAD today), literature-aware LLM routing, optional ML layers (scikit-learn [@pedregosa2011scikit], PyTorch [@paszke2019pytorch], and other optional dependencies), and plotting (Matplotlib [@hunter2007matplotlib], Plotly [@plotly2015]). A separate contribution to FireWorks or MatMiner would not, on its own, deliver the same end-user experience; conversely, MaterialSim intentionally builds *on* those ecosystems' components where possible (for example, Pymatgen interoperability) instead of reimplementing solvers or potentials. The scholarly contribution is the **integration architecture**: a maintainable agent layer (LangChain/LangGraph [@langchain2023; @langgraph2024]) that maps research intent—spoken or typed—to validated tool calls, intelligent model selection informed by arXiv and web search, logging, and configuration, while keeping the scientific heavy lifting in established open-source engines.

# Software design

MaterialSim follows a modular layout so that new tools (potentials, analysis routines, schedulers) can be added without rewriting the core agent loop.

**Agent core.** A LangGraph-style ReAct agent [@langgraph2024] parses user intent (from text or transcribed speech), selects tools, and sequences steps such as database query, structure preparation, simulation, and post-processing. A `ModelRouter` classifies tasks and optionally queries arXiv [@arxiv2007arxiv] and web benchmarks before instantiating the appropriate hosted LLM. This separates *policy* (what to run next, which model to use) from *mechanism* (how each tool runs), which aids testing and extension.

**Simulation engine.** `simple_simulation.py` drives ASE molecular dynamics with configurable ensemble, thermostat, timestep, and output frequency, persisting `trajectory.xyz`, `output.log`, and `final_structure.xyz`. `LAMMPSInterface` provides an alternate subprocess path when `LAMMPS_EXECUTABLE` is available. Structure generation uses ASE `bulk`/`molecule` builders informed by an internal materials database.

**Voice orchestration.** The `materials_ai_agent.voice` package implements speech-to-text (SpeechRecognition [@speechrecognition2024]), text-to-speech (`edge-tts`), and a `VoiceOrchestrator` that maps utterances to simulation, analysis, research, database, and chat actions via `MaterialsAgent.process_command`. The Streamlit JARVIS panel (`gui/jarvis_panel.py`, `gui/jarvis_ui.py`) renders an animated HUD and auto-plays spoken responses.

**Analysis.** `analysis_engine.py` computes RDF, MSD, and thermodynamic statistics from trajectory and log files, saving publication-style plots. Results are available through the CLI, Python API, GUI, and voice replies.

**Tool suite.** LangChain-callable tools wrap MD execution, analysis, materials listing, simulation discovery, and literature/model research (`agent_tools.py`). Optional wrappers expose LAMMPS setup, ML training (`MLTool`), and database queries (`DatabaseTool`) when dependencies and API keys are configured.

**Data access.** The `DatabaseTool` implementation queries the Materials Project (via `mp-api` / `MPRester`) and NOMAD [@jain2013commentary; @draxl2018nomad] when optional packages are installed. Database metadata in the same module lists Open Catalyst [@chanussot2021open] as **not yet implemented**; the paper describes current behavior rather than roadmap items as finished features.

**Interfaces.** Users may interact through: (1) **CLI** — `materials-agent run|analyze|predict|query|interactive`; (2) **Python API** — `MaterialsAgent` methods; (3) **Streamlit GUI** — conversational workflow and JARVIS voice mode; (4) **voice commands** — microphone input with spoken summaries. Matplotlib and Plotly [@hunter2007matplotlib; @plotly2015] support charts; Streamlit [@streamlit2023] implements the web UI over the same Python APIs, consistent with Journal of Open Source Software guidance for web experiences built around a **core library**.

**Deployment.** Configuration is driven by environment variables (and optional YAML via `Config.from_file`), with example files in the repository; API keys for language and database services remain out of source control. Core MD runs locally without external services; LLM and database features are opt-in.

**Quality assurance.** The repository includes a `tests/` tree with integration tests that execute short real ASE simulations and verify RDF peaks against known lattice spacings, plus `Makefile` targets that invoke pytest [@pytest2023]; detailed API behavior and installation belong in repository documentation rather than in this paper, per JOSS expectations.

Design trade-offs include reliance on external language-model APIs for conversational and voice-dialogue features (e.g., OpenAI [@openai2024api]; users must supply keys and accept provider terms), cloud speech-recognition backends for microphone input, and the need for domain validation: the agent proposes plans, but scientific correctness still requires reviewer-level scrutiny of inputs, potentials, and convergence, exactly as in manually scripted workflows.

# Research impact statement

MaterialSim is developed in public on GitHub [Awwal41/MaterialSim](https://github.com/Awwal41/MaterialSim). The repository provides pytest-based automated tests [@pytest2023] that run real short MD trajectories and validate analysis outputs, installation instructions, a CLI (`materials-agent`), and runnable examples that connect natural-language and **voice** workflows to ASE-backed simulations and standard analysis outputs. The JARVIS voice mode demonstrates an end-to-end spoken pipeline—from utterance through model routing and tool execution to spoken and visual feedback—supporting independent verification on local machines without a cluster or LAMMPS install for the default engine path.

# AI usage disclosure

Generative AI tools were used to fix grammatical errors in this manuscript; authors reviewed all text.

# Acknowledgements

MaterialSim builds on LAMMPS [@plimpton1995fast], ASE [@larsen2017atomic], Pymatgen [@ong2013python], LangChain [@langchain2023], LangGraph [@langgraph2024], the OpenAI API [@openai2024api], arXiv [@arxiv2007arxiv], Materials Project [@jain2013commentary], NOMAD [@draxl2018nomad], Open Catalyst [@chanussot2021open], SpeechRecognition [@speechrecognition2024], edge-tts [@edgetts2024], and the wider Python scientific stack. We thank collaborators and contributors to the codebase. The authors declare no competing interests. No dedicated funding was received for this manuscript.

# References
