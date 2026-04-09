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

**MaterialSim** helps researchers drive such workflows through natural language: an agent plans and calls tools for molecular dynamics, property analysis, and optional machine learning, producing quantitative outputs and plots. It automates the setup and execution of molecular dynamics (MD) simulations, with system input configurations either generated within MaterialSim or imported from the Materials Project database. The platform recommends suitable interatomic models tailored to the system and analyzes the outputs to extract key properties, such as radial distribution functions and mean-squared displacements. MaterialSim also provides automated visualization, result interpretation, and accelerated predictions through machine learning models. 

The software is developed in public as the GitHub repository [Awwal41/MaterialSim](https://github.com/Awwal41/MaterialSim). The Python import path is `materials_ai_agent`, the main class is `MaterialsAgent`, and the console entry point is `materials-agent` (distribution metadata in `setup.py` uses the name `materialsim-ai-agent`). Implementations use LAMMPS [@plimpton1995fast] together with ASE [@larsen2017atomic] and Pymatgen [@ong2013python], LangChain [@langchain2023] with hosted large language models accessed through the OpenAI API [@openai2024api] (users supply their own keys), Matplotlib and Plotly [@hunter2007matplotlib; @plotly2015], scikit-learn and PyTorch [@pedregosa2011scikit; @paszke2019pytorch], plus additional dependencies listed in `requirements.txt` (including TensorFlow and Hugging Face `transformers`). Documented analysis targets include radial distribution functions, mean-squared displacement, elastic constants, and thermal-conductivity-style studies. **Materials Project** [@jain2013commentary] and **NOMAD** [@draxl2018nomad] queries are implemented in code; **Open Catalyst** [@chanussot2021open] is recorded in the same module as not yet implemented and should not be described as a live integration. A Streamlit [@streamlit2023] application in `gui_app.py` (see `launch_gui.py`) provides a conversational GUI; Crystal Toolkit supports structure-centric views. The project targets Python 3.8+ and runs LAMMPS via a configured executable (subprocess), which can be used on laptops or clusters without built-in scheduler coupling in that driver.

# Statement of need

Computational materials research depends on reproducible, composable pipelines from structure retrieval to simulation and post-processing. High-throughput frameworks such as FireWorks within the Atomate ecosystem [@jain2015fireworks] and analysis toolkits such as MatMiner [@ward2018matminer] have lowered barriers for expert users, but they still assume substantial familiarity with workflow configuration and Python glue code. Recent work has shown that large language models can act as planners that call external tools in scientific settings, for example, ChemCrow augments models with chemistry tools [@bran2023chemcrow]—but comparable, materials, focused orchestration that is packaged for simulation-scale tasks (MD engines, materials databases, cluster-capable execution of those engines) remains comparatively scarce.

MaterialSim targets researchers, educators, and interdisciplinary teams who want to iterate quickly on simulation ideas without rewriting boilerplate for every study. It is designed for exploratory studies, teaching demonstrations, and rapid prototyping where the cost of manual pipeline construction is high. The package emphasizes transparent composition of third-party tools rather than replacing domain codes: LAMMPS, ASE, and Pymatgen [@ong2013python] remain the sources of truth for numerics and data structures.

# State of the field

Workflow automation in materials informatics is mature in high-throughput and database-centric settings. Atomate/FireWorks [@jain2015fireworks] provides powerful, production-oriented workflow management for density-functional-theory-centric pipelines; MatMiner [@ward2018matminer] focuses on featurization and machine-learning datasets from structures and calculations. These systems excel when users already know which codes and presets to chain. General scientific tool-augmented language agents [@bran2023chemcrow] demonstrate natural-language steering of tool ecosystems but are not specialized for materials simulation stacks, HPC submission, or typical MD observables.

MaterialSim occupies a complementary niche: **natural-language orchestration of MD-centric materials simulations** with explicit bindings to LAMMPS/ASE, database-backed access where implemented (Materials Project and NOMAD today), optional ML layers (scikit-learn [@pedregosa2011scikit], PyTorch [@paszke2019pytorch], and other pinned ML dependencies), and plotting (Matplotlib [@hunter2007matplotlib], Plotly [@plotly2015]). A separate contribution to FireWorks or MatMiner would not, on its own, deliver the same end-user experience; conversely, MaterialSim intentionally builds *on* those ecosystems' components where possible (for example, Pymatgen interoperability) instead of reimplementing solvers or potentials. The scholarly contribution is the **integration architecture**: a maintainable agent layer (LangChain [@langchain2023]) that maps research intent to validated tool calls, logging, and configuration, while keeping the scientific heavy lifting in established open-source engines and user-provided LAMMPS installations.

# Software design

MaterialSim follows a modular layout so that new tools (potentials, analysis routines, schedulers) can be added without rewriting the core agent loop.

**Agent core.** A LangChain-style agent [@langchain2023] parses user intent, selects tools, and sequences steps such as database query, structure preparation, simulation, and post-processing. This separates *policy* (what to run next) from *mechanism* (how each tool runs), which aids testing and extension.

**Tool suite.** Wrappers expose MD drivers (LAMMPS invoked as a subprocess with ASE `Atoms` for structure handling), property calculators (e.g., RDF, MSD, elastic and thermal workflows described in the docs and CLI help), and ML tooling (`MLTool` and related dependencies). Heavy numerical work stays in LAMMPS, Pymatgen, and the declared ML stack; the agent layer orchestrates calls.

**Data access.** The `DatabaseTool` implementation queries the Materials Project (via `mp-api` / `MPRester`) and NOMAD [@jain2013commentary; @draxl2018nomad]. Database metadata in the same module lists Open Catalyst [@chanussot2021open] as **not yet implemented**; the paper describes current behavior rather than roadmap items as finished features.

**Visualization and UX.** Matplotlib and Plotly [@hunter2007matplotlib; @plotly2015] support charts; Streamlit [@streamlit2023] implements the web UI over the same Python APIs, consistent with Journal of Open Source Software guidance for web experiences built around a **core library**.

**Deployment.** Configuration is driven by environment variables (and optional YAML), with example files in the repository; API keys for language and database services remain out of source control. Simulations use a user-configured `LAMMPS_EXECUTABLE`, so deployments on shared infrastructure are supported to the extent that LAMMPS is available in the execution environment—the reference implementation does not encode scheduler, specific submission in the LAMMPS interface reviewed here.

**Quality assurance.** The repository includes a `tests/` tree, root-level test modules, and `Makefile` targets that invoke pytest [@pytest2023]; detailed API behavior and installation belong in repository documentation rather than in this paper, per JOSS expectations.

Design trade-offs include reliance on external language-model APIs for the agent (e.g., OpenAI [@openai2024api]; users must supply keys and accept provider terms) and the need for domain validation: the agent proposes plans, but scientific correctness still requires reviewer, level scrutiny of inputs, potentials, and convergence, exactly as in manually scripted workflows.

# Research impact statement

MaterialSim is developed in public on GitHub[Awwal41/MaterialSim](https://github.com/Awwal41/MaterialSim). The repository provides pytest-based automated tests [@pytest2023], installation instructions, and runnable examples that connect natural-language workflows to LAMMPS and ASE-backed simulations and standard analysis outputs, supporting independent verification on local or cluster environments.

# AI usage disclosure

Generative AI tools were used to fix grammatical errors in this manuscript; authors reviewed all text.

# Acknowledgements

MaterialSim builds on LAMMPS [@plimpton1995fast], ASE [@larsen2017atomic], Pymatgen [@ong2013python], LangChain [@langchain2023], the OpenAI API [@openai2024api], Materials Project [@jain2013commentary], NOMAD [@draxl2018nomad], Open Catalyst [@chanussot2021open], and the wider Python scientific stack. We thank collaborators and contributors to the codebase. The authors declare no competing interests. No dedicated funding was received for this manuscript.

# References
